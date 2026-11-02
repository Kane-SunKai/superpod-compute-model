// Independent global-buffer -> LDS service with per-block scratch ownership.
// The final sum checks every LDS word; word permutation inside a vector does
// not affect the expected checksum. All source words are nonuniform.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using U4=unsigned __attribute__((ext_vector_type(4)));
#define CHECK(e) do{auto rc=(e);if(rc!=hipSuccess){std::fprintf(stderr,"%s: %s\n",#e,hipGetErrorString(rc));std::exit(2);}}while(0)

template<int Words> __global__ __launch_bounds__(256)
void probe(const unsigned char* input,unsigned* output,unsigned long long* clocks,
           unsigned bytes,unsigned stride,unsigned base,int iterations){
    __shared__ unsigned tile[16384];
    for(int i=threadIdx.x;i<16384;i+=256)tile[i]=0;
    __syncthreads();
    uintptr_t address=reinterpret_cast<uintptr_t>(input);
    U4 descriptor={unsigned(address),unsigned(address>>32),bytes,0x00020000u};
    unsigned offset=threadIdx.x*stride+base;
    unsigned shared_base=unsigned(reinterpret_cast<uintptr_t>(tile));
    unsigned wave=__builtin_amdgcn_readfirstlane(threadIdx.x/64);
    auto start=clock64();
    for(int i=0;i<iterations;i++){
        #pragma unroll
        for(int j=0;j<4;j++){
            unsigned destination=shared_base+(j*256+wave*64)*Words*4;
            unsigned source=offset+j*16;
            asm volatile("s_mov_b32 m0, %0" :: "s"(destination) : "m0","memory");
            if constexpr(Words==1)asm volatile("buffer_load_dword %0, %1, 0 offen lds" :: "v"(source),"s"(descriptor) : "memory");
            if constexpr(Words==2)asm volatile("buffer_load_dwordx2 %0, %1, 0 offen lds" :: "v"(source),"s"(descriptor) : "memory");
            if constexpr(Words==4)asm volatile("buffer_load_dwordx4 %0, %1, 0 offen lds" :: "v"(source),"s"(descriptor) : "memory");
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
    }
    auto stop=clock64();
    if(threadIdx.x==0)clocks[blockIdx.x]=stop-start;
    volatile unsigned* observed=tile;
    unsigned sum=0;
    for(int i=threadIdx.x;i<16384;i+=256)sum+=observed[i];
    output[blockIdx.x*256+threadIdx.x]=sum;
}

template<int Words> void measure(unsigned char* gpu,const std::vector<unsigned char>& host,
                                 int blocks,unsigned stride,unsigned base,int iterations){
    unsigned* output;unsigned long long* clocks;
    CHECK(hipMalloc(&output,blocks*256*4));CHECK(hipMalloc(&clocks,blocks*8));
    hipEvent_t begin,end;CHECK(hipEventCreate(&begin));CHECK(hipEventCreate(&end));
    auto launch=[&](){hipLaunchKernelGGL(probe<Words>,dim3(blocks),dim3(256),0,0,gpu,output,clocks,host.size(),stride,base,iterations);CHECK(hipGetLastError());};
    for(int i=0;i<3;i++)launch();CHECK(hipDeviceSynchronize());
    std::vector<float> times;
    for(int i=0;i<9;i++){
        CHECK(hipEventRecord(begin));launch();CHECK(hipEventRecord(end));CHECK(hipEventSynchronize(end));
        float ms;CHECK(hipEventElapsedTime(&ms,begin,end));times.push_back(ms*1000);
    }
    std::vector<unsigned> values(blocks*256);std::vector<unsigned long long> ticks(blocks);
    CHECK(hipMemcpy(values.data(),output,values.size()*4,hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(ticks.data(),clocks,blocks*8,hipMemcpyDeviceToHost));
    unsigned expected=0;
    for(int t=0;t<256;t++)for(int j=0;j<4;j++)for(int w=0;w<Words;w++){
        unsigned word;std::memcpy(&word,host.data()+t*stride+base+j*16+w*4,4);expected+=word;
    }
    bool correct=true;
    for(int b=0;b<blocks;b++){
        unsigned observed=0;for(int t=0;t<256;t++)observed+=values[b*256+t];
        if(observed!=expected){correct=false;std::fprintf(stderr,"block %d expected %u observed %u\n",b,expected,observed);break;}
    }
    std::sort(times.begin(),times.end());std::sort(ticks.begin(),ticks.end());
    hipFuncAttributes attr;CHECK(hipFuncGetAttributes(&attr,reinterpret_cast<const void*>(probe<Words>)));
    std::printf("{\"load_bytes\":%d,\"loads_per_wave_per_iteration\":4,\"blocks\":%d,\"stride_bytes\":%u,\"base_bytes\":%u,\"iterations\":%d,\"median_us\":%.9f,\"ticks\":%llu,\"local_bytes\":%zu,\"correct\":%s}\n",Words*4,blocks,stride,base,iterations,times[4],ticks[ticks.size()/2],attr.localSizeBytes,correct?"true":"false");std::fflush(stdout);
    if(!correct)std::exit(3);
    CHECK(hipEventDestroy(begin));CHECK(hipEventDestroy(end));CHECK(hipFree(output));CHECK(hipFree(clocks));
}

int main(){
    CHECK(hipSetDevice(0));std::vector<unsigned char> host(1024*1024);
    for(size_t i=0;i<host.size();i++)host[i]=unsigned((i*17+i/19)%251);
    unsigned char* gpu;CHECK(hipMalloc(&gpu,host.size()));
    CHECK(hipMemcpy(gpu,host.data(),host.size(),hipMemcpyHostToDevice));
    const unsigned cases[][2]={{2046,0},{2048,0},{2050,0},{2048,2},{2048,4},{2048,8}};
    for(auto& c:cases)for(int blocks:{1,72})for(int iterations:{1024,4096}){
        measure<1>(gpu,host,blocks,c[0],c[1],iterations);
        measure<2>(gpu,host,blocks,c[0],c[1],iterations);
        measure<4>(gpu,host,blocks,c[0],c[1],iterations);
    }
    CHECK(hipFree(gpu));
}
