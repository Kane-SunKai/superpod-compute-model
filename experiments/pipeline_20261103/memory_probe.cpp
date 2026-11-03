// Controlled full-array reads/copies: each element exactly once. No vendor code.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#define CK(x) do{auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(2);}}while(0)
template<bool Copy> __global__ void linear(const uint4* a,uint4* b,unsigned* sums,size_t packs){
 unsigned acc=0;
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<packs;i+=size_t(gridDim.x)*blockDim.x){
   uint4 v=a[i];if(Copy)b[i]=v;else acc+=v.x+v.y+v.z+v.w;
 }
 if(!Copy)sums[blockIdx.x*blockDim.x+threadIdx.x]=acc;
}
template<bool Copy> __global__ void panel(const uint4* a,uint4* b,unsigned* sums,int rows,int npacks,int width,int kparts){
 int nt=npacks/width,nb=blockIdx.x%nt,kb=blockIdx.x/nt;
 int first=rows*kb/kparts,last=rows*(kb+1)/kparts;unsigned acc=0;
 for(size_t j=threadIdx.x;j<size_t(last-first)*width;j+=blockDim.x){
   size_t idx=(first+j/width)*size_t(npacks)+nb*width+j%width;
   uint4 v=a[idx];if(Copy)b[idx]=v;else acc+=v.x+v.y+v.z+v.w;
 }
 if(!Copy)sums[blockIdx.x*blockDim.x+threadIdx.x]=acc;
}
void run(int K,int N,int width,int concurrency,bool copy,int round){
 size_t bytes=size_t(K)*N*2,packs=bytes/16;int npacks=N/8;
 int blocks=width?(npacks/width)*concurrency:concurrency;
 uint4 *a,*b;unsigned* sums;CK(hipMalloc(&a,bytes));CK(hipMalloc(&b,bytes));CK(hipMemset(a,1,bytes));
 CK(hipMalloc(&sums,size_t(blocks)*256*4));CK(hipMemset(b,0,bytes));
 auto launch=[&](){
   if(width){
     if(copy)hipLaunchKernelGGL(panel<true>,dim3(blocks),dim3(256),0,0,a,b,sums,K,npacks,width,concurrency);
     else hipLaunchKernelGGL(panel<false>,dim3(blocks),dim3(256),0,0,a,b,sums,K,npacks,width,concurrency);
   }else{
     if(copy)hipLaunchKernelGGL(linear<true>,dim3(blocks),dim3(256),0,0,a,b,sums,packs);
     else hipLaunchKernelGGL(linear<false>,dim3(blocks),dim3(256),0,0,a,b,sums,packs);
   }CK(hipGetLastError());
 };
 for(int i=0;i<10;i++)launch();CK(hipDeviceSynchronize());
 hipEvent_t start,end;CK(hipEventCreate(&start));CK(hipEventCreate(&end));std::vector<double> ts;
 for(int i=0;i<21;i++){CK(hipEventRecord(start));launch();CK(hipEventRecord(end));CK(hipEventSynchronize(end));float ms;CK(hipEventElapsedTime(&ms,start,end));ts.push_back(ms*1000);}
 bool ok=true;
 if(copy){std::vector<unsigned> out(bytes/4);CK(hipMemcpy(out.data(),b,bytes,hipMemcpyDeviceToHost));for(unsigned x:out)if(x!=0x01010101u)ok=false;}
 else {std::vector<unsigned> out(size_t(blocks)*256);CK(hipMemcpy(out.data(),sums,out.size()*4,hipMemcpyDeviceToHost));unsigned total=0;for(unsigned x:out)total+=x;ok=(total==unsigned(uint64_t(bytes/4)*0x01010101u));}
 std::sort(ts.begin(),ts.end());
 printf("{\"round\":%d,\"K\":%d,\"N\":%d,\"column_half_elements\":%d,\"kparts_or_blocks\":%d,\"blocks\":%d,\"copy\":%s,\"read_bytes\":%zu,\"write_bytes\":%zu,\"median_us\":%.6f,\"p10_us\":%.6f,\"p90_us\":%.6f,\"correct\":%s}\n",round,K,N,width*8,concurrency,blocks,copy?"true":"false",bytes,copy?bytes:0,ts[10],ts[2],ts[18],ok?"true":"false");fflush(stdout);
 CK(hipFree(a));CK(hipFree(b));CK(hipFree(sums));CK(hipEventDestroy(start));CK(hipEventDestroy(end));if(!ok)exit(3);
}
int main(){CK(hipSetDevice(0));for(int round:{1,2}){
 for(int n:{6144,12288,24576}){
   for(int blocks:{36,72,144,288})run(6144,n,0,blocks,false,round);
   for(int width:{16,32,64,256})for(int kp:{1,4})run(6144,n,width,kp,false,round);
   run(6144,n,0,288,true,round);run(6144,n,32,4,true,round);
 }
}}
