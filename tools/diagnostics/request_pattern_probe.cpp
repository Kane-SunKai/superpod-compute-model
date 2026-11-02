// Independent address-pattern delivery test. Patterns are runtime input.
// Same final source addresses and LDS m0 control/layout as the supplied plan;
// source uses raw offen addressing (not the original indexed address setup).
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
using U4=unsigned __attribute__((ext_vector_type(4)));
#define CHECK(e) do{auto rc=(e);if(rc!=hipSuccess){std::fprintf(stderr,"%s: %s\n",#e,hipGetErrorString(rc));std::exit(2);}}while(0)

template<int Threads,int Panels> __global__ __launch_bounds__(Threads,1)
void delivery(const unsigned char* a,const unsigned char* b,unsigned bytes_a,unsigned bytes_b,
              const unsigned* patterns,unsigned* output,unsigned long long* clocks,int iterations){
    __shared__ unsigned tile[16384];
    unsigned offsets[Panels*3],destinations[Panels*3];
    #pragma unroll
    for(int stage=0;stage<Panels*3;stage++){
        offsets[stage]=patterns[(stage*Threads+threadIdx.x)*2];
        destinations[stage]=__builtin_amdgcn_readfirstlane(patterns[(stage*Threads+threadIdx.x)*2+1]);
    }
    uintptr_t aa=reinterpret_cast<uintptr_t>(a),bb=reinterpret_cast<uintptr_t>(b);
    U4 ra={unsigned(aa),unsigned(aa>>32),bytes_a,0x20000u};
    U4 rb={unsigned(bb),unsigned(bb>>32),bytes_b,0x20000u};
    for(int i=threadIdx.x;i<16384;i+=Threads)tile[i]=0;
    __syncthreads();auto start=clock64();
    for(int iteration=0;iteration<iterations;iteration++){
        #pragma unroll
        for(int stage=0;stage<Panels*3;stage++){
            asm volatile("s_mov_b32 m0, %0" :: "s"(destinations[stage]) : "m0","memory");
            if(stage%3==0)asm volatile("buffer_load_dwordx4 %0, %1, 0 offen lds" :: "v"(offsets[stage]),"s"(rb) : "memory");
            else if(stage%3==1)asm volatile("buffer_load_dwordx4 %0, %1, 0 offen lds" :: "v"(offsets[stage]),"s"(ra) : "memory");
            else asm volatile("buffer_load_dwordx2 %0, %1, 0 offen lds" :: "v"(offsets[stage]),"s"(ra) : "memory");
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");__syncthreads();
    }
    auto stop=clock64();
    if(threadIdx.x==0)clocks[blockIdx.x]=stop-start;
    volatile unsigned* observed=tile;unsigned sum=0;
    for(int i=threadIdx.x;i<16384;i+=Threads)sum+=observed[i];
    output[blockIdx.x*Threads+threadIdx.x]=sum;
}

template<int Threads,int Panels> void measure(const unsigned char* a,const unsigned char* b,unsigned bytes_a,unsigned bytes_b,
             const unsigned* patterns,unsigned expected,int blocks,int iterations){
    unsigned* output;unsigned long long* clocks;
    CHECK(hipMalloc(&output,blocks*Threads*4));CHECK(hipMalloc(&clocks,blocks*8));
    hipEvent_t start,end;CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&end));
    auto launch=[&](){hipLaunchKernelGGL((delivery<Threads,Panels>),dim3(blocks),dim3(Threads),0,0,a,b,bytes_a,bytes_b,patterns,output,clocks,iterations);CHECK(hipGetLastError());};
    for(int i=0;i<3;i++)launch();CHECK(hipDeviceSynchronize());
    std::vector<float> times;
    for(int i=0;i<9;i++){
        CHECK(hipEventRecord(start));launch();CHECK(hipEventRecord(end));CHECK(hipEventSynchronize(end));
        float ms;CHECK(hipEventElapsedTime(&ms,start,end));times.push_back(ms*1000);
    }
    std::vector<unsigned> values(blocks*Threads);std::vector<unsigned long long> ticks(blocks);
    CHECK(hipMemcpy(values.data(),output,values.size()*4,hipMemcpyDeviceToHost));CHECK(hipMemcpy(ticks.data(),clocks,blocks*8,hipMemcpyDeviceToHost));
    bool correct=true;
    for(int block=0;block<blocks;block++){
        unsigned got=0;for(int t=0;t<Threads;t++)got+=values[block*Threads+t];
        if(got!=expected){correct=false;std::fprintf(stderr,"block=%d expected=%u got=%u\n",block,expected,got);break;}
    }
    std::sort(times.begin(),times.end());std::sort(ticks.begin(),ticks.end());
    hipFuncAttributes attr;CHECK(hipFuncGetAttributes(&attr,reinterpret_cast<const void*>(delivery<Threads,Panels>)));
    std::printf("{\"threads\":%d,\"queued_panels\":%d,\"blocks\":%d,\"iterations\":%d,\"median_us\":%.9f,\"ticks\":%llu,\"registers\":%d,\"shared_bytes\":%zu,\"local_bytes\":%zu,\"correct\":%s}\n",Threads,Panels,blocks,iterations,times[4],ticks[ticks.size()/2],attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes,correct?"true":"false");std::fflush(stdout);
    if(!correct)std::exit(3);
    CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(end));CHECK(hipFree(output));CHECK(hipFree(clocks));
}

int main(int argc,char** argv){
    if(argc!=2)return 1;std::ifstream input(argv[1]);int threads,max_panels;unsigned bytes_a,bytes_b,offset_a,offset_b;
    input>>threads>>max_panels>>bytes_a>>bytes_b>>offset_a>>offset_b;
    if(!((threads==256&&max_panels==6)||(threads==512&&max_panels==3))||offset_a>=256||offset_b>=256)return 1;
    std::vector<unsigned> patterns(threads*max_panels*3*2);
    for(auto& value:patterns)if(!(input>>value))return 1;
    std::vector<unsigned char> host_a(bytes_a+256),host_b(bytes_b+256);
    for(size_t i=0;i<host_a.size();i++)host_a[i]=(i*17+i/19)%251;
    for(size_t i=0;i<host_b.size();i++)host_b[i]=(i*23+i/31)%251;
    CHECK(hipSetDevice(0));unsigned char *a,*b;unsigned* gpu_patterns;
    CHECK(hipMalloc(&a,host_a.size()));CHECK(hipMalloc(&b,host_b.size()));CHECK(hipMalloc(&gpu_patterns,patterns.size()*4));
    CHECK(hipMemcpy(a,host_a.data(),host_a.size(),hipMemcpyHostToDevice));CHECK(hipMemcpy(b,host_b.data(),host_b.size(),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(gpu_patterns,patterns.data(),patterns.size()*4,hipMemcpyHostToDevice));
    for(int panels:{1,2,3,6}){
        if(panels>max_panels)continue;unsigned expected=0;
        for(int stage=0;stage<panels*3;stage++)for(int thread=0;thread<threads;thread++){
            unsigned offset=patterns[(stage*threads+thread)*2],width=stage%3==2?8:16;
            bool use_b=stage%3==0;
            if(offset+width>(use_b?bytes_b:bytes_a))return 4;
            const unsigned char* src=(use_b?host_b.data()+offset_b:host_a.data()+offset_a)+offset;
            for(unsigned j=0;j<width;j+=4){unsigned word;std::memcpy(&word,src+j,4);expected+=word;}
        }
        for(int blocks:{1,72})for(int iterations:{512,2048}){
            #define RUN(T,P) measure<T,P>(a+offset_a,b+offset_b,bytes_a,bytes_b,gpu_patterns,expected,blocks,iterations)
            if(threads==256){switch(panels){case 1:RUN(256,1);break;case 2:RUN(256,2);break;case 3:RUN(256,3);break;case 6:RUN(256,6);break;}}
            else {switch(panels){case 1:RUN(512,1);break;case 2:RUN(512,2);break;case 3:RUN(512,3);break;}}
        }
    }
    CHECK(hipFree(a));CHECK(hipFree(b));CHECK(hipFree(gpu_patterns));
}
