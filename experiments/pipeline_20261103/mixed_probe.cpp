// Standalone synthetic MMAC/LDS scheduling probe; not the GEMM kernel.
#include <hip/hip_runtime.h>
#include <du_mma.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do{auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(2);}}while(0)
using V4=float __attribute__((ext_vector_type(4)));
template<int Chains,int Mode> __global__ void mixed(float* output,int iterations){
 using namespace du::dumma;
 __shared__ volatile V4 data[1024];
 for(int i=threadIdx.x;i<1024;i+=blockDim.x)data[i]=V4{1,1,1,1};
 DUFragment<matrix_a,16,16,16,__half,row_major> a;
 DUFragment<matrix_b,16,16,16,__half,col_major> b;
 DUFragment<accumulator,16,16,16,float> c[Chains];
 du_fill_fragment(a,__float2half(1));du_fill_fragment(b,__float2half(1));
 for(int j=0;j<Chains;j++)du_fill_fragment(c[j],0.f);
 __syncthreads();V4 sum{0,0,0,0};
 for(int i=0;i<iterations/64;i++){
  #pragma unroll
  for(int u=0;u<64;u++){
   if(Mode!=0){V4 v=data[(threadIdx.x+(u&3)*blockDim.x)&1023];sum+=v;
     // Mode 3 has a genuine LDS -> operand dependency, with conversion charged.
     if(Mode==3){du_fill_fragment(a,__float2half(v[0]));}}
   if(Mode!=1){
    #pragma unroll
    for(int j=0;j<Chains;j++)du_mma_sync(c[j],a,b,c[j]);
   }
  }
 }
 int tid=blockIdx.x*blockDim.x+threadIdx.x;
 for(int j=0;j<Chains;j++)output[tid*(Chains+1)+j]=c[j].x[0];
 output[tid*(Chains+1)+Chains]=sum[0]+sum[1]+sum[2]+sum[3];
}
template<int C,int Mode> void run(int waves,int iterations,int round){
 int blocks=72,threads=waves*64;size_t size=size_t(blocks)*threads*(C+1);float* out;CK(hipMalloc(&out,size*4));
 auto launch=[&](){hipLaunchKernelGGL((mixed<C,Mode>),dim3(blocks),dim3(threads),0,0,out,iterations);CK(hipGetLastError());};
 for(int i=0;i<8;i++)launch();CK(hipDeviceSynchronize());hipEvent_t a,b;CK(hipEventCreate(&a));CK(hipEventCreate(&b));std::vector<double> ts;
 for(int i=0;i<15;i++){CK(hipEventRecord(a));launch();CK(hipEventRecord(b));CK(hipEventSynchronize(b));float ms;CK(hipEventElapsedTime(&ms,a,b));ts.push_back(ms*1000);}
 std::vector<float> values(size);CK(hipMemcpy(values.data(),out,size*4,hipMemcpyDeviceToHost));bool ok=true;
 for(size_t tid=0;tid<size_t(blocks)*threads;tid++){
  for(int j=0;j<C;j++)if(values[tid*(C+1)+j]!=(Mode==1?0.f:iterations*16.f))ok=false;
  if(values[tid*(C+1)+C]!=(Mode==0?0.f:iterations*4.f))ok=false;
 }
 std::sort(ts.begin(),ts.end());printf("{\"round\":%d,\"chains\":%d,\"mode\":%d,\"waves\":%d,\"iterations\":%d,\"blocks\":72,\"median_us\":%.6f,\"p10_us\":%.6f,\"p90_us\":%.6f,\"correct\":%s}\n",round,C,Mode,waves,iterations,ts[7],ts[1],ts[13],ok?"true":"false");fflush(stdout);
 CK(hipFree(out));CK(hipEventDestroy(a));CK(hipEventDestroy(b));if(!ok)exit(3);
}
template<int C>void modes(int w,int i,int r){run<C,0>(w,i,r);run<C,1>(w,i,r);run<C,2>(w,i,r);run<C,3>(w,i,r);}
int main(){CK(hipSetDevice(0));for(int r:{1,2})for(int w:{4,8})for(int i:{4096,16384}){modes<1>(w,i,r);modes<4>(w,i,r);modes<8>(w,i,r);}}
