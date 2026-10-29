#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
using V4=float __attribute__((ext_vector_type(4)));
#define CHECK(x) do{auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(2);}}while(0)
__global__ __launch_bounds__(512) void lds_write(float* out,unsigned long long* ticks,int iterations){
  __shared__ volatile V4 tile[1024];
  for(int j=threadIdx.x;j<1024;j+=blockDim.x)tile[j]=V4{0,0,0,0};
  __syncthreads();unsigned long long begin=clock64();
  V4 value{float(threadIdx.x+1),2,3,4};
  for(int i=0;i<iterations/64;i++){
    #pragma unroll
    for(int u=0;u<64;u++)tile[(threadIdx.x+(u&3)*blockDim.x)&1023]=value;
  }
  __syncthreads();auto end=clock64();
  if(threadIdx.x==0)ticks[blockIdx.x]=end-begin;
  // All readback and global writes are outside the clock interval.
  for(int j=threadIdx.x;j<1024;j+=blockDim.x){
    V4 v=tile[j];
    bool touched=j<4*blockDim.x;
    float expected=touched?float(j%blockDim.x+1)+9:0;
    out[blockIdx.x*1024+j]=(v[0]+v[1]+v[2]+v[3]==expected)?1:0;
  }
  return;
}
int main(){
  CHECK(hipSetDevice(0));
  for(int blocks:{1,72,288})for(int waves:{1,4,8})for(int iterations:{4096,16384}){
    int threads=64*waves;float* out;unsigned long long* clocks;
    CHECK(hipMalloc(&out,blocks*1024*sizeof(float)));CHECK(hipMalloc(&clocks,blocks*sizeof(unsigned long long)));
    hipEvent_t a,b;CHECK(hipEventCreate(&a));CHECK(hipEventCreate(&b));
    auto launch=[&](){hipLaunchKernelGGL(lds_write,dim3(blocks),dim3(threads),0,0,out,clocks,iterations);CHECK(hipGetLastError());};
    for(int i=0;i<8;i++)launch();CHECK(hipDeviceSynchronize());
    std::vector<float> times;std::vector<unsigned long long> medians;
    for(int i=0;i<9;i++){
      CHECK(hipEventRecord(a));launch();CHECK(hipEventRecord(b));CHECK(hipEventSynchronize(b));float ms;CHECK(hipEventElapsedTime(&ms,a,b));times.push_back(ms*1000);
      std::vector<unsigned long long> cycles(blocks);CHECK(hipMemcpy(cycles.data(),clocks,blocks*sizeof(unsigned long long),hipMemcpyDeviceToHost));
      std::sort(cycles.begin(),cycles.end());medians.push_back(cycles[cycles.size()/2]);
    }
    std::vector<float> values(blocks*1024);CHECK(hipMemcpy(values.data(),out,values.size()*sizeof(float),hipMemcpyDeviceToHost));
    bool correct=std::all_of(values.begin(),values.end(),[](float x){return x==1;});
    std::sort(times.begin(),times.end());std::sort(medians.begin(),medians.end());
    printf("{\"blocks\":%d,\"waves\":%d,\"iterations\":%d,\"median_us\":%.9f,\"block_ticks\":%llu,\"bytes_per_iteration_per_block\":%d,\"correct\":%s}\n",blocks,waves,iterations,times[4],medians[4],threads*16,correct?"true":"false");fflush(stdout);
    CHECK(hipFree(out));CHECK(hipFree(clocks));CHECK(hipEventDestroy(a));CHECK(hipEventDestroy(b));if(!correct)return 3;
  }
}
