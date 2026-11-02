// Independent raw-buffer read service. No vendor kernel code or GEMM times.
// All accesses are byte-addressed inline loads, so misaligned C++ dereference
// is avoided. Full per-thread checksums validate the actual returned bytes.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using U4=unsigned __attribute__((ext_vector_type(4)));
using U2=unsigned __attribute__((ext_vector_type(2)));
#define CHECK(e) do{auto rc=(e);if(rc!=hipSuccess){std::fprintf(stderr,"%s: %s\n",#e,hipGetErrorString(rc));std::exit(2);}}while(0)

template<int Words> __device__ __forceinline__ void read_words(U4 descriptor,unsigned offset,U4& value){
    if constexpr(Words==1){
        unsigned v;
        asm volatile("buffer_load_dword %0, %1, %2, 0 offen" : "=v"(v) : "v"(offset),"s"(descriptor) : "memory");
        value=U4{v,0,0,0};
    }else if constexpr(Words==2){
        U2 v;
        asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 offen" : "=v"(v) : "v"(offset),"s"(descriptor) : "memory");
        value=U4{v[0],v[1],0,0};
    }else{
        asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen" : "=v"(value) : "v"(offset),"s"(descriptor) : "memory");
    }
}

template<int Words> __global__ __launch_bounds__(256)
void probe(const unsigned char* input,unsigned* output,unsigned long long* clocks,
           unsigned bytes,unsigned stride,unsigned base,int iterations){
    uintptr_t address=reinterpret_cast<uintptr_t>(input);
    U4 descriptor={unsigned(address),unsigned(address>>32),bytes,0x00020000u};
    unsigned offset=threadIdx.x*stride+base;
    unsigned total=0;
    // The same 256-row region is shared by workgroups to keep the experiment
    // cache-resident; it is deliberately not a measurement of HBM bandwidth.
    __syncthreads();auto start=clock64();
    for(int i=0;i<iterations;i++){
        U4 values[4];
        #pragma unroll
        for(int j=0;j<4;j++)read_words<Words>(descriptor,offset+j*16,values[j]);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        #pragma unroll
        for(int j=0;j<4;j++){
            #pragma unroll
            for(int w=0;w<Words;w++)total+=values[j][w];
        }
    }
    __syncthreads();auto stop=clock64();
    output[blockIdx.x*blockDim.x+threadIdx.x]=total;
    if(threadIdx.x==0)clocks[blockIdx.x]=stop-start;
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
    bool correct=true;
    for(int t=0;t<blocks*256;t++){
        unsigned expected=0;
        for(int j=0;j<4;j++)for(int w=0;w<Words;w++){
            unsigned word;std::memcpy(&word,host.data()+(t%256)*stride+base+j*16+w*4,4);expected+=word;
        }
        if(values[t]!=expected*unsigned(iterations))correct=false;
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
