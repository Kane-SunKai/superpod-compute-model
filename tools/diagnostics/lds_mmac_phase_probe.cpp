// Independent component experiment: LDS instruction mix + matrix accumulation.
// Private address patterns are runtime input, never embedded vendor code.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
using U4=unsigned __attribute__((ext_vector_type(4)));
using U2=unsigned __attribute__((ext_vector_type(2)));
using F4=float __attribute__((ext_vector_type(4)));
#define CHECK(e) do{auto rc=(e);if(rc!=hipSuccess){std::fprintf(stderr,"%s: %s\n",#e,hipGetErrorString(rc));std::exit(2);}}while(0)

template<int R> struct Operands{U4 a[R];U2 b[6];};
template<int R> __device__ __forceinline__ void load(Operands<R>& v,const unsigned (&offset)[R+6],unsigned base){
    #pragma unroll
    for(int j=0;j<R;j++){
        unsigned address=base+offset[j];
        asm volatile("ds_read_m32x16_b16 %0, %1" : "=v"(v.a[j]) : "v"(address) : "memory");
    }
    #pragma unroll
    for(int j=0;j<6;j++){
        unsigned address=base+offset[R+j];
        asm volatile("ds_read_b64 %0, %1" : "=v"(v.b[j]) : "v"(address) : "memory");
    }
}
template<int R> __device__ __forceinline__ void multiply(F4 (&c)[R*12],const Operands<R>& op){
    #pragma unroll
    for(int j=0;j<6;j++){
        #pragma unroll
        for(int i=0;i<R*2;i++){
            U2 a={op.a[i/2][2*(i%2)],op.a[i/2][2*(i%2)+1]};
            asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0" : "+v"(c[j*R*2+i]) : "v"(a),"v"(op.b[j]));
        }
    }
}
// mode 0: registers only; mode 1: read then compute; mode 2: one-panel prefetch.
template<int R,int Mode> __global__ __launch_bounds__(R==2?256:512,1)
void phase(const unsigned* input,float* output,unsigned long long* clocks,int iterations){
    __shared__ unsigned tile[16384];
    unsigned offsets[R+6];
    #pragma unroll
    for(int j=0;j<R+6;j++)offsets[j]=input[threadIdx.x*(R+6)+j];
    for(int j=threadIdx.x;j<16384;j+=blockDim.x)tile[j]=0x3c003c00u;
    __syncthreads();
    unsigned base=static_cast<unsigned>(reinterpret_cast<uintptr_t>(tile));
    asm volatile("s_mov_b32 m0, 0x10000" ::: "m0");
    Operands<R> operands[Mode==2?2:1];F4 c[R*12];
    #pragma unroll
    for(int j=0;j<R*12;j++)c[j]=F4{0,0,0,0};
    load<R>(operands[0],offsets,base);
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __syncthreads();auto start=clock64();
    for(int iteration=0;iteration<iterations/2;iteration++){
        #pragma unroll
        for(int step=0;step<2;step++){
            if constexpr(Mode==1){
                load<R>(operands[0],offsets,base);
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            }
            if constexpr(Mode==2){
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                load<R>(operands[1-step],offsets,base);
            }
            multiply<R>(c,operands[Mode==2?step:0]);
            __syncthreads();
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    auto end=clock64();
    if(threadIdx.x==0)clocks[blockIdx.x]=end-start;
    #pragma unroll
    for(int j=0;j<R*12;j++){
        #pragma unroll
        for(int z=0;z<4;z++)output[((blockIdx.x*blockDim.x+threadIdx.x)*(R*12)+j)*4+z]=c[j][z];
    }
}

template<int R,int Mode> void run(unsigned* addresses,int blocks,int iterations){
    constexpr int waves=R==2?4:8;constexpr int threads=waves*64;
    size_t count=size_t(blocks)*threads*R*12*4;
    float* output;unsigned long long* clocks;
    CHECK(hipMalloc(&output,count*sizeof(float)));CHECK(hipMalloc(&clocks,blocks*8));
    hipEvent_t begin,end;CHECK(hipEventCreate(&begin));CHECK(hipEventCreate(&end));
    auto launch=[&](){hipLaunchKernelGGL((phase<R,Mode>),dim3(blocks),dim3(threads),0,0,addresses,output,clocks,iterations);CHECK(hipGetLastError());};
    for(int j=0;j<3;j++)launch();CHECK(hipDeviceSynchronize());
    std::vector<float> times;
    for(int j=0;j<9;j++){
        CHECK(hipEventRecord(begin));launch();CHECK(hipEventRecord(end));CHECK(hipEventSynchronize(end));
        float ms;CHECK(hipEventElapsedTime(&ms,begin,end));times.push_back(ms*1000);
    }
    std::vector<float> values(count);std::vector<unsigned long long> ticks(blocks);
    CHECK(hipMemcpy(values.data(),output,count*sizeof(float),hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(ticks.data(),clocks,blocks*8,hipMemcpyDeviceToHost));
    bool correct=std::all_of(values.begin(),values.end(),[&](float v){return v==16.f*iterations;});
    std::sort(times.begin(),times.end());std::sort(ticks.begin(),ticks.end());
    hipFuncAttributes attr;CHECK(hipFuncGetAttributes(&attr,reinterpret_cast<const void*>(phase<R,Mode>)));
    std::printf("{\"waves\":%d,\"mode\":%d,\"blocks\":%d,\"iterations\":%d,\"median_us\":%.9f,\"ticks\":%llu,\"registers\":%d,\"local_bytes\":%zu,\"shared_bytes\":%zu,\"correct\":%s}\n",waves,Mode,blocks,iterations,times[4],ticks[ticks.size()/2],attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,correct?"true":"false");std::fflush(stdout);
    if(!correct){std::fprintf(stderr,"expected %g first %g\n",16.f*iterations,values[0]);std::exit(3);}
    CHECK(hipFree(output));CHECK(hipFree(clocks));CHECK(hipEventDestroy(begin));CHECK(hipEventDestroy(end));
}
int main(int argc,char** argv){
    if(argc!=2)return 1;
    std::ifstream in(argv[1]);int waves,reads;in>>waves>>reads;
    if(!((waves==4&&reads==2)||(waves==8&&reads==4)))return 1;
    std::vector<unsigned> offsets(waves*64*(reads+6));
    for(auto& v:offsets)if(!(in>>v)||v+16>65536)return 1;
    CHECK(hipSetDevice(0));unsigned* addresses;CHECK(hipMalloc(&addresses,offsets.size()*4));
    CHECK(hipMemcpy(addresses,offsets.data(),offsets.size()*4,hipMemcpyHostToDevice));
    for(int blocks:{1,72})for(int iterations:{1024,4096}){
        if(reads==2){run<2,0>(addresses,blocks,iterations);run<2,1>(addresses,blocks,iterations);run<2,2>(addresses,blocks,iterations);}
        else {run<4,0>(addresses,blocks,iterations);run<4,1>(addresses,blocks,iterations);}
    }
    CHECK(hipFree(addresses));
}
