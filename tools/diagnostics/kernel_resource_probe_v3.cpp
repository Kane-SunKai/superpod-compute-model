#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
extern "C" hipError_t hipModuleGetFunction(hipFunction_t* fn, hipModule_t mod, const char* name) {
 using Get = hipError_t (*)(hipFunction_t*,hipModule_t,const char*);
 static auto next = reinterpret_cast<Get>(dlsym(RTLD_NEXT,"hipModuleGetFunction"));
 if (!next) return hipErrorUnknown;
 auto rc=next(fn,mod,name);
 if (rc==hipSuccess && name && (strstr(name,"_MT") || strstr(name,"PostGSU"))) {
  using Query=hipError_t (*)(int*,hipFunction_attribute,hipFunction_t);
  static auto query=reinterpret_cast<Query>(dlsym(RTLD_NEXT,"hipFuncGetAttribute"));
  int regs=0,shared=0,local=0,threads=0;
  auto qr=query ? query(&regs,HIP_FUNC_ATTRIBUTE_NUM_REGS,*fn) : hipErrorUnknown;
  if (qr==hipSuccess) {
   query(&shared,HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,*fn);
   query(&local,HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,*fn);
   query(&threads,HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,*fn);
  }
  using Occ=hipError_t (*)(int*,hipFunction_t,int,size_t);
  static auto occupancy=reinterpret_cast<Occ>(dlsym(RTLD_NEXT,"hipModuleOccupancyMaxActiveBlocksPerMultiprocessor"));
  unsigned w0=1,w1=1,w2=1;int actual_threads=threads;
  const char* wg=strstr(name,"_WG");
  if(wg && std::sscanf(wg,"_WG%u_%u_%u",&w0,&w1,&w2)==3) actual_threads=w0*w1*w2;
  int blocks=0;
  auto oq=occupancy ? occupancy(&blocks,*fn,actual_threads,0) : hipErrorUnknown;
  std::fprintf(stderr,"RESOURCE\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",name,int(qr),regs,shared,local,threads,int(oq),blocks,actual_threads);
 }
 return rc;
}
