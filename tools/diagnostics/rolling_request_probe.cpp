// Independent rolling global-to-LDS request queue. No matrix computation.
// Private source/destination patterns are input data, not embedded instructions.
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

template<int T,int Q> __device__ __forceinline__
void issue_panel(int slot,unsigned panel,const unsigned (&offset)[Q*3],const unsigned (&dest)[Q*3],
                 unsigned b_step,U4 a,U4 b){
    #pragma unroll
    for(int j=0;j<3;j++){
        unsigned address=offset[slot*3+j]+panel*(j==0?b_step:32u);
        asm volatile("s_mov_b32 m0, %0" :: "s"(dest[slot*3+j]) : "m0","memory");
        if(j==0)asm volatile("buffer_load_dwordx4 %0, %1, 0 offen lds" :: "v"(address),"s"(b) : "memory");
        else if(j==1)asm volatile("buffer_load_dwordx4 %0, %1, 0 offen lds" :: "v"(address),"s"(a) : "memory");
        else asm volatile("buffer_load_dwordx2 %0, %1, 0 offen lds" :: "v"(address),"s"(a) : "memory");
    }
}

template<int T,int Q,bool Validate> __global__ __launch_bounds__(T,1)
void rolling(const unsigned char* a,const unsigned char* b,unsigned bytes_a,unsigned bytes_b,
             const unsigned* patterns,const unsigned* shifts,unsigned* output,unsigned long long* clocks,int iterations){
    __shared__ unsigned tile[16384];
    unsigned offset[Q*3],dest[Q*3];unsigned b_step=bytes_b/64u; // N*16*2, K=1024
    #pragma unroll
    for(int s=0;s<Q*3;s++){
        offset[s]=patterns[(s*T+threadIdx.x)*2]+shifts[blockIdx.x*2+(s%3==0?1:0)]-(s/3)*(s%3==0?b_step:32u);
        dest[s]=__builtin_amdgcn_readfirstlane(patterns[(s*T+threadIdx.x)*2+1]);
    }
    uintptr_t pa=reinterpret_cast<uintptr_t>(a),pb=reinterpret_cast<uintptr_t>(b);
    U4 ra={unsigned(pa),unsigned(pa>>32),bytes_a,0x20000u},rb={unsigned(pb),unsigned(pb>>32),bytes_b,0x20000u};
    for(int i=threadIdx.x;i<16384;i+=T)tile[i]=0;
    __syncthreads();unsigned consumed=0;auto begin=clock64();
    #pragma unroll
    for(int slot=0;slot<Q;slot++)issue_panel<T,Q>(slot,slot,offset,dest,b_step,ra,rb);
    for(int iteration=0;iteration<iterations;iteration++){
        #pragma unroll
        for(int slot=0;slot<Q;slot++){
            // Each panel contains three buffer-read instructions per wave.
            asm volatile("s_waitcnt vmcnt(%0)" :: "n"(3*(Q-1)) : "memory");
            __syncthreads();
            if constexpr(Validate){
                // Validate panel readiness, not merely the final drained data.
                constexpr int stride_words=(T==256?10752:20480)/4;
                volatile unsigned* observed=tile;
                for(int i=threadIdx.x;i<stride_words;i+=T)consumed+=observed[slot*stride_words+i];
                __syncthreads(); // all validation reads finish before slot reuse
            }
            unsigned next=((iteration+1)*Q+slot)&63u;
            issue_panel<T,Q>(slot,next,offset,dest,b_step,ra,rb);
        }
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");__syncthreads();auto end=clock64();
    if(threadIdx.x==0)clocks[blockIdx.x]=end-begin;
    unsigned final_sum=0;volatile unsigned* observed=tile;
    for(int i=threadIdx.x;i<16384;i+=T)final_sum+=observed[i];
    output[(blockIdx.x*T+threadIdx.x)*2]=final_sum;
    output[(blockIdx.x*T+threadIdx.x)*2+1]=consumed;
}

struct Input {
    int threads,panels;unsigned bytes_a,bytes_b,base_a,base_b;
    std::vector<unsigned> patterns,shifts;
    std::vector<unsigned char> a,b;
    unsigned char *da,*db;unsigned *dp,*ds;
    unsigned panel_sum(int group,int slot,int panel)const{
        unsigned sum=0;
        for(int j=0;j<3;j++)for(int t=0;t<threads;t++){
            bool use_b=j==0;unsigned stride=use_b?bytes_b/64u:32u;
            unsigned offset=patterns[((slot*3+j)*threads+t)*2]+shifts[group*2+(use_b?1:0)]-slot*stride+panel*stride;
            unsigned width=j==2?8:16;
            if(offset+width>(use_b?bytes_b:bytes_a)){std::fprintf(stderr,"source outside allocation\n");std::exit(4);}
            const auto* ptr=(use_b?b.data()+base_b:a.data()+base_a)+offset;
            for(unsigned k=0;k<width;k+=4){unsigned word;std::memcpy(&word,ptr+k,4);sum+=word;}
        }
        return sum;
    }
};

template<int T,int Q,bool Validate> void measure(const Input& input,int groups,int iterations){
    unsigned* output;unsigned long long* clocks;
    CHECK(hipMalloc(&output,groups*T*8));CHECK(hipMalloc(&clocks,groups*8));
    auto launch=[&](){hipLaunchKernelGGL((rolling<T,Q,Validate>),dim3(groups),dim3(T),0,0,
        input.da+input.base_a,input.db+input.base_b,input.bytes_a,input.bytes_b,input.dp,input.ds,output,clocks,iterations);CHECK(hipGetLastError());};
    std::vector<float> times;hipEvent_t start,stop;CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&stop));
    if constexpr(Validate){launch();CHECK(hipDeviceSynchronize());}
    else {
        for(int i=0;i<3;i++)launch();CHECK(hipDeviceSynchronize());
        for(int i=0;i<9;i++){
            CHECK(hipEventRecord(start));launch();CHECK(hipEventRecord(stop));CHECK(hipEventSynchronize(stop));
            float ms;CHECK(hipEventElapsedTime(&ms,start,stop));times.push_back(ms*1000);
        }
    }
    std::vector<unsigned> values(groups*T*2);std::vector<unsigned long long> ticks(groups);
    CHECK(hipMemcpy(values.data(),output,values.size()*4,hipMemcpyDeviceToHost));CHECK(hipMemcpy(ticks.data(),clocks,groups*8,hipMemcpyDeviceToHost));
    bool correct=true;
    for(int g=0;g<groups;g++){
        unsigned got_final=0,got_consumed=0,expected_final=0,expected_consumed=0;
        for(int t=0;t<T;t++){got_final+=values[(g*T+t)*2];got_consumed+=values[(g*T+t)*2+1];}
        for(int slot=0;slot<Q;slot++)expected_final+=input.panel_sum(g,slot,(iterations*Q+slot)&63);
        if constexpr(Validate)for(int i=0;i<iterations;i++)for(int slot=0;slot<Q;slot++)expected_consumed+=input.panel_sum(g,slot,(i*Q+slot)&63);
        if(got_final!=expected_final||got_consumed!=expected_consumed){
            correct=false;std::fprintf(stderr,"group=%d final %u/%u consumed %u/%u\n",g,got_final,expected_final,got_consumed,expected_consumed);break;
        }
    }
    hipFuncAttributes attr;CHECK(hipFuncGetAttributes(&attr,reinterpret_cast<const void*>(rolling<T,Q,Validate>)));
    std::sort(times.begin(),times.end());std::sort(ticks.begin(),ticks.end());
    std::printf("{\"threads\":%d,\"queued_panels\":%d,\"blocks\":%d,\"iterations\":%d,\"validation_only\":%s,\"median_us\":%.9f,\"ticks\":%llu,\"registers\":%d,\"shared_bytes\":%zu,\"local_bytes\":%zu,\"correct\":%s}\n",T,Q,groups,iterations,Validate?"true":"false",Validate?0:times[4],ticks[ticks.size()/2],attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes,correct?"true":"false");std::fflush(stdout);
    if(!correct)std::exit(3);
    CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(stop));CHECK(hipFree(output));CHECK(hipFree(clocks));
}

int main(int argc,char**argv){
    if(argc!=3)return 1;Input x;std::ifstream p(argv[1]),s(argv[2]);
    p>>x.threads>>x.panels>>x.bytes_a>>x.bytes_b>>x.base_a>>x.base_b;
    if(!((x.threads==256&&x.panels==6)||(x.threads==512&&x.panels==3)))return 1;
    x.patterns.resize(x.threads*x.panels*3*2);x.shifts.resize(144);
    for(auto&v:x.patterns)if(!(p>>v))return 1;for(auto&v:x.shifts)if(!(s>>v))return 1;
    x.a.resize(x.bytes_a+256);x.b.resize(x.bytes_b+256);
    for(size_t i=0;i<x.a.size();i++)x.a[i]=(i*17+i/19)%251;
    for(size_t i=0;i<x.b.size();i++)x.b[i]=(i*23+i/31)%251;
    CHECK(hipSetDevice(0));CHECK(hipMalloc(&x.da,x.a.size()));CHECK(hipMalloc(&x.db,x.b.size()));
    CHECK(hipMalloc(&x.dp,x.patterns.size()*4));CHECK(hipMalloc(&x.ds,x.shifts.size()*4));
    CHECK(hipMemcpy(x.da,x.a.data(),x.a.size(),hipMemcpyHostToDevice));CHECK(hipMemcpy(x.db,x.b.data(),x.b.size(),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(x.dp,x.patterns.data(),x.patterns.size()*4,hipMemcpyHostToDevice));CHECK(hipMemcpy(x.ds,x.shifts.data(),x.shifts.size()*4,hipMemcpyHostToDevice));
    #define RUN(T,Q) do {measure<T,Q,true>(x,72,(65+Q-1)/Q);for(int groups:{1,72})for(int iterations:{512,2048})measure<T,Q,false>(x,groups,iterations);} while(0)
    if(x.threads==256){RUN(256,1);RUN(256,2);RUN(256,3);RUN(256,6);}
    else {RUN(512,1);RUN(512,2);RUN(512,3);}
    CHECK(hipFree(x.da));CHECK(hipFree(x.db));CHECK(hipFree(x.dp));CHECK(hipFree(x.ds));
}
