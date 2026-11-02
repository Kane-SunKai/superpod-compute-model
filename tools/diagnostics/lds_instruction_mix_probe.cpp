// Independent LDS instruction-service microbenchmark. Access offsets are data
// supplied at run time, not embedded vendor kernel code. No GEMM timings used.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
using U4=unsigned __attribute__((ext_vector_type(4)));
using U2=unsigned __attribute__((ext_vector_type(2)));
#define CHECK(expr) do {auto rc=(expr); if(rc!=hipSuccess){std::fprintf(stderr,"%s: %s\n",#expr,hipGetErrorString(rc));std::exit(2);}} while(0)

template<int MatrixReads,bool Barrier>
__global__ __launch_bounds__(512) void read_mix(const unsigned* input,unsigned* output,unsigned long long* ticks,int iterations){
    __shared__ unsigned tile[16384];
    constexpr int reads=MatrixReads+6;
    unsigned offsets[reads];
    #pragma unroll
    for(int j=0;j<reads;j++)offsets[j]=input[threadIdx.x*reads+j];
    for(int j=threadIdx.x;j<16384;j+=blockDim.x)tile[j]=0x3c003c00u;
    __syncthreads();
    unsigned base=static_cast<unsigned>(reinterpret_cast<uintptr_t>(tile));
    unsigned sum=0;auto start=clock64();
    asm volatile("s_mov_b32 m0, 0x10000" ::: "m0");
    for(int outer=0;outer<iterations/8;outer++){
        #pragma unroll
        for(int u=0;u<8;u++){
            U4 matrix_values[MatrixReads];U2 scalar_values[6];
            #pragma unroll
            for(int j=0;j<MatrixReads;j++){
                unsigned address=base+offsets[j];
                asm volatile("ds_read_m32x16_b16 %0, %1" : "=v"(matrix_values[j]) : "v"(address) : "memory");
            }
            #pragma unroll
            for(int j=0;j<6;j++){
                unsigned address=base+offsets[MatrixReads+j];
                asm volatile("ds_read_b64 %0, %1" : "=v"(scalar_values[j]) : "v"(address) : "memory");
            }
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            #pragma unroll
            for(int j=0;j<MatrixReads;j++)sum+=matrix_values[j][0];
            #pragma unroll
            for(int j=0;j<6;j++)sum+=scalar_values[j][0];
            if constexpr(Barrier)__syncthreads();
        }
    }
    output[blockIdx.x*blockDim.x+threadIdx.x]=sum;
    __syncthreads();auto end=clock64();
    if(threadIdx.x==0)ticks[blockIdx.x]=end-start;
}

int main(int argc,char** argv){
    if(argc!=2)return 1;
    std::ifstream input(argv[1]);int waves,reads;
    input>>waves>>reads;
    if((waves!=4&&waves!=8)||(reads!=2&&reads!=4))return 1;
    int threads=waves*64;
    std::vector<unsigned> offsets(threads*(reads+6));
    for(auto& value:offsets){if(!(input>>value)||value+16>65536)return 1;}
    CHECK(hipSetDevice(0));unsigned* addresses;CHECK(hipMalloc(&addresses,offsets.size()*4));
    CHECK(hipMemcpy(addresses,offsets.data(),offsets.size()*4,hipMemcpyHostToDevice));
    for(int blocks:{1,72})for(int barrier:{0,1})for(int iterations:{2048,8192}){
        unsigned* output;unsigned long long* clock_output;
        CHECK(hipMalloc(&output,blocks*threads*4));CHECK(hipMalloc(&clock_output,blocks*8));
        hipEvent_t begin,end;CHECK(hipEventCreate(&begin));CHECK(hipEventCreate(&end));
        auto launch=[&](){
            if(reads==2){
                if(barrier)hipLaunchKernelGGL((read_mix<2,true>),dim3(blocks),dim3(threads),0,0,addresses,output,clock_output,iterations);
                else hipLaunchKernelGGL((read_mix<2,false>),dim3(blocks),dim3(threads),0,0,addresses,output,clock_output,iterations);
            }else{
                if(barrier)hipLaunchKernelGGL((read_mix<4,true>),dim3(blocks),dim3(threads),0,0,addresses,output,clock_output,iterations);
                else hipLaunchKernelGGL((read_mix<4,false>),dim3(blocks),dim3(threads),0,0,addresses,output,clock_output,iterations);
            }
            CHECK(hipGetLastError());
        };
        for(int i=0;i<3;i++)launch();CHECK(hipDeviceSynchronize());
        std::vector<float> times;
        for(int i=0;i<9;i++){
            CHECK(hipEventRecord(begin));launch();CHECK(hipEventRecord(end));CHECK(hipEventSynchronize(end));
            float ms;CHECK(hipEventElapsedTime(&ms,begin,end));times.push_back(ms*1000);
        }
        std::vector<unsigned> values(blocks*threads);std::vector<unsigned long long> ticks(blocks);
        CHECK(hipMemcpy(values.data(),output,values.size()*4,hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(ticks.data(),clock_output,blocks*8,hipMemcpyDeviceToHost));
        unsigned expected=unsigned(iterations)*unsigned(reads+6)*0x3c003c00u;
        bool correct=std::all_of(values.begin(),values.end(),[&](unsigned v){return v==expected;});
        std::sort(times.begin(),times.end());std::sort(ticks.begin(),ticks.end());
        std::printf("{\"waves\":%d,\"matrix_reads\":%d,\"blocks\":%d,\"barrier\":%d,\"iterations\":%d,\"median_us\":%.9f,\"ticks\":%llu,\"bytes_per_panel_per_block\":%d,\"correct\":%s}\n",waves,reads,blocks,barrier,iterations,times[4],ticks[ticks.size()/2],waves*(reads*1024+6*512),correct?"true":"false");std::fflush(stdout);
        if(!correct){std::fprintf(stderr,"checksum expected %u first %u\n",expected,values[0]);return 3;}
        CHECK(hipEventDestroy(begin));CHECK(hipEventDestroy(end));CHECK(hipFree(output));CHECK(hipFree(clock_output));
    }
    CHECK(hipFree(addresses));
}
