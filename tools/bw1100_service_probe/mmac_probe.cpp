#include <hip/hip_runtime.h>
#include <du_mma.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(2);} } while(0)

template<int Chains> __global__ __launch_bounds__(512) void probe(float* output,unsigned long long* clocks,int iterations) {
  using namespace du::dumma;
  DUFragment<matrix_a,16,16,16,__half,row_major> a;
  DUFragment<matrix_b,16,16,16,__half,col_major> b;
  DUFragment<accumulator,16,16,16,float> c[Chains];
  du_fill_fragment(a,__float2half(1.0f));du_fill_fragment(b,__float2half(1.0f));
  #pragma unroll
  for(int j=0;j<Chains;j++)du_fill_fragment(c[j],float(j)+float(threadIdx.x&1)*0.125f);
  __syncthreads();
  unsigned long long start=clock64();
  for(int i=0;i<iterations/128;i++) {
    #pragma unroll
    for(int u=0;u<128;u++) {
      #pragma unroll
      for(int j=0;j<Chains;j++)du_mma_sync(c[j],a,b,c[j]);
    }
  }
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  #pragma unroll
  for(int j=0;j<Chains;j++) {
    #pragma unroll
    for(int z=0;z<4;z++)output[(tid*Chains+j)*4+z]=c[j].x[z];
  }
  __syncthreads();
  unsigned long long stop=clock64();
  if(threadIdx.x==0)clocks[blockIdx.x]=stop-start;
  return;
}

template<int C> void run(int blocks,int waves,int iterations) {
  const int threads=waves*64;size_t size=size_t(blocks)*threads*C*4;
  float *gpu;unsigned long long *ticks;
  CHECK(hipMalloc(&gpu,size*sizeof(float)));CHECK(hipMalloc(&ticks,blocks*sizeof(unsigned long long)));
  hipEvent_t start,end;CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&end));
  auto launch=[&](){hipLaunchKernelGGL(probe<C>,dim3(blocks),dim3(threads),0,0,gpu,ticks,iterations);CHECK(hipGetLastError());};
  for(int i=0;i<4;i++)launch();CHECK(hipDeviceSynchronize());
  std::vector<double> times;
  for(int i=0;i<9;i++) {
    CHECK(hipEventRecord(start));launch();CHECK(hipEventRecord(end));CHECK(hipEventSynchronize(end));
    float ms;CHECK(hipEventElapsedTime(&ms,start,end));times.push_back(double(ms)*1000);
  }
  std::sort(times.begin(),times.end());
  std::vector<float> values(size);std::vector<unsigned long long> cycles(blocks);
  CHECK(hipMemcpy(values.data(),gpu,size*sizeof(float),hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(cycles.data(),ticks,blocks*sizeof(unsigned long long),hipMemcpyDeviceToHost));
  bool correct=true;
  for(size_t t=0;t<size_t(blocks)*threads;t++)for(int j=0;j<C;j++)for(int z=0;z<4;z++) {
    float expected=16.f*iterations+float(j)+float(t%threads&1)*0.125f;
    if(values[(t*C+j)*4+z]!=expected)correct=false;
  }
  std::sort(cycles.begin(),cycles.end());
  printf("{\"blocks\":%d,\"waves_per_block\":%d,\"chains\":%d,\"iterations\":%d,\"median_us\":%.9f,\"min_us\":%.9f,\"max_us\":%.9f,\"median_block_clock64_ticks\":%llu,\"correct\":%s}\n",blocks,waves,C,iterations,times[4],times[0],times.back(),cycles[cycles.size()/2],correct?"true":"false");
  fflush(stdout);
  CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(end));CHECK(hipFree(gpu));CHECK(hipFree(ticks));
  if(!correct)exit(3);
}

int main() {
  CHECK(hipSetDevice(0));
  // Warm the matrix pipelines before measuring low-occupancy cases; no DVFS writes.
  float* warm;unsigned long long* warm_ticks;
  CHECK(hipMalloc(&warm,size_t(72)*512*8*4*sizeof(float)));
  CHECK(hipMalloc(&warm_ticks,72*sizeof(unsigned long long)));
  for(int i=0;i<300;i++)hipLaunchKernelGGL(probe<8>,dim3(72),dim3(512),0,0,warm,warm_ticks,16384);
  CHECK(hipGetLastError());CHECK(hipDeviceSynchronize());CHECK(hipFree(warm));CHECK(hipFree(warm_ticks));
  for(int blocks: {1,72,288})for(int waves:{1,4,8})for(int iters:{4096,16384}) {
    run<1>(blocks,waves,iters);run<4>(blocks,waves,iters);run<8>(blocks,waves,iters);
  }
}
