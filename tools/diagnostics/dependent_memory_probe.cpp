#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>
#define CHECK(x) do{auto rc=(x);if(rc!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(rc));exit(2);}}while(0)
__global__ void chase(const volatile unsigned* ring,const unsigned* seeds,unsigned* outputs,unsigned long long* ticks,unsigned iterations){
 unsigned index=seeds[blockIdx.x];
 __syncthreads();unsigned long long start=clock64();
 if(threadIdx.x==0){
  #pragma unroll 1
  for(unsigned i=0;i<iterations;i++)index=ring[size_t(index)*16];
  outputs[blockIdx.x]=index;
 }
 __syncthreads();unsigned long long stop=clock64();
 if(threadIdx.x==0)ticks[blockIdx.x]=stop-start;
}
void run(size_t bytes,unsigned blocks){
 unsigned nodes=bytes/64;
 std::vector<unsigned> order(nodes),ring(nodes*16,0);std::iota(order.begin(),order.end(),0);
 std::mt19937 rng(20261102);std::shuffle(order.begin(),order.end(),rng);
 for(unsigned i=0;i<nodes;i++)ring[size_t(order[i])*16]=order[(i+1)%nodes];
 unsigned *device_ring,*device_seeds,*device_outputs;unsigned long long* device_ticks;
 CHECK(hipMalloc(&device_ring,bytes));CHECK(hipMalloc(&device_seeds,blocks*4));CHECK(hipMalloc(&device_outputs,blocks*4));CHECK(hipMalloc(&device_ticks,blocks*8));
 CHECK(hipMemcpy(device_ring,ring.data(),bytes,hipMemcpyHostToDevice));
 hipEvent_t begin,end;CHECK(hipEventCreate(&begin));CHECK(hipEventCreate(&end));
 std::vector<unsigned> seeds(blocks),positions(blocks),outputs(blocks);std::vector<unsigned long long> ticks(blocks);
 size_t cursor=0;
 auto launch=[&](unsigned iters){hipLaunchKernelGGL(chase,dim3(blocks),dim3(64),0,0,device_ring,device_seeds,device_outputs,device_ticks,iters);CHECK(hipGetLastError());};
 for(unsigned iters:{8192u,32768u}){
  std::vector<double> times;std::vector<unsigned long long> all_ticks;
  bool correct=true;
  for(unsigned sample=0;sample<9;sample++){
   for(unsigned b=0;b<blocks;b++){
    positions[b]=(cursor+size_t(b)*std::max(1u,nodes/blocks))%nodes;seeds[b]=order[positions[b]];
   }
   cursor+=iters;
   CHECK(hipMemcpy(device_seeds,seeds.data(),blocks*4,hipMemcpyHostToDevice));
   CHECK(hipEventRecord(begin));launch(iters);CHECK(hipEventRecord(end));CHECK(hipEventSynchronize(end));
   float ms;CHECK(hipEventElapsedTime(&ms,begin,end));
   CHECK(hipMemcpy(outputs.data(),device_outputs,blocks*4,hipMemcpyDeviceToHost));CHECK(hipMemcpy(ticks.data(),device_ticks,blocks*8,hipMemcpyDeviceToHost));
   for(unsigned b=0;b<blocks;b++)if(outputs[b]!=order[(positions[b]+iters)%nodes])correct=false;
   if(sample>=2){times.push_back(double(ms)*1000);all_ticks.insert(all_ticks.end(),ticks.begin(),ticks.end());}
  }
  std::sort(times.begin(),times.end());std::sort(all_ticks.begin(),all_ticks.end());
  printf("{\"footprint_bytes\":%zu,\"blocks\":%u,\"iterations\":%u,\"median_event_us\":%.9f,\"median_block_clock_ticks\":%llu,\"event_p10_us\":%.9f,\"event_p90_us\":%.9f,\"correct\":%s}\n",bytes,blocks,iters,times[times.size()/2],all_ticks[all_ticks.size()/2],times.front(),times.back(),correct?"true":"false");fflush(stdout);
  if(!correct)exit(3);
 }
 CHECK(hipEventDestroy(begin));CHECK(hipEventDestroy(end));CHECK(hipFree(device_ring));CHECK(hipFree(device_seeds));CHECK(hipFree(device_outputs));CHECK(hipFree(device_ticks));
}
int main(){CHECK(hipSetDevice(0));for(size_t bytes:{4096ul,262144ul,2097152ul,33554432ul})for(unsigned blocks:{1u,72u})run(bytes,blocks);}
