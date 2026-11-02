// Observe only selected GEMM launch arguments. No tensor contents are read.
// Replace all device pointers with synthetic bases preserving modulo 256.
#include <hip/hip_runtime_api.h>
#include <hip/hip_ext.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>

static std::mutex lock;
static std::unordered_map<hipFunction_t,std::string> names;
static std::set<hipFunction_t> seen;

static void observe(hipFunction_t f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,void** extra){
    std::lock_guard<std::mutex> guard(lock);
    if(!names.count(f)||seen.count(f))return;
    seen.insert(f);
    const unsigned char* buffer=nullptr;size_t size=0;
    if(extra)for(unsigned i=0;i<16&&extra[i]!=HIP_LAUNCH_PARAM_END;i+=2){
        if(extra[i]==HIP_LAUNCH_PARAM_BUFFER_POINTER)buffer=static_cast<const unsigned char*>(extra[i+1]);
        if(extra[i]==HIP_LAUNCH_PARAM_BUFFER_SIZE&&extra[i+1])std::memcpy(&size,extra[i+1],sizeof(size));
    }
    if(!buffer||size<156||size>160){std::fprintf(stderr,"ARG_UNSUPPORTED\t%s\t%zu\n",names[f].c_str(),size);return;}
    std::array<unsigned char,160> data{};std::memcpy(data.data(),buffer,size);
    uint64_t original[4];for(int i=0;i<4;i++)std::memcpy(&original[i],data.data()+24+8*i,8);
    for(int i=0;i<4;i++){
        int identity=i;for(int j=0;j<i;j++)if(original[j]==original[i]){identity=j;break;}
        uint64_t synthetic=(uint64_t(identity+1)<<32)+(original[i]%256);
        std::memcpy(data.data()+24+8*i,&synthetic,8);
    }
    std::fprintf(stderr,"GEMM_ARGS\t%s\t%u,%u,%u\t%u,%u,%u\t",names[f].c_str(),gx,gy,gz,bx,by,bz);
    for(auto byte:data)std::fprintf(stderr,"%02x",unsigned(byte));std::fprintf(stderr,"\n");
}

extern "C" hipError_t hipModuleGetFunction(hipFunction_t* f,hipModule_t module,const char* name){
    using Get=hipError_t(*)(hipFunction_t*,hipModule_t,const char*);
    static auto next=reinterpret_cast<Get>(dlsym(RTLD_NEXT,"hipModuleGetFunction"));
    if(!next)return hipErrorUnknown;auto status=next(f,module,name);
    if(status==hipSuccess&&name&&(std::strstr(name,"_MT128x192x16_")||std::strstr(name,"_MT256x384x16_"))){
        std::lock_guard<std::mutex> guard(lock);names[*f]=name;
    }
    return status;
}
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,
                                             unsigned shared,hipStream_t stream,void** params,void** extra){
    using Launch=hipError_t(*)(hipFunction_t,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,hipStream_t,void**,void**);
    static auto next=reinterpret_cast<Launch>(dlsym(RTLD_NEXT,"hipModuleLaunchKernel"));
    if(!next)return hipErrorUnknown;observe(f,gx*bx,gy*by,gz*bz,bx,by,bz,extra);
    return next(f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
}
extern "C" hipError_t hipExtModuleLaunchKernel(hipFunction_t f,uint32_t gx,uint32_t gy,uint32_t gz,uint32_t bx,uint32_t by,uint32_t bz,
                                                size_t shared,hipStream_t stream,void** params,void** extra,hipEvent_t start,hipEvent_t stop,uint32_t flags){
    using Launch=hipError_t(*)(hipFunction_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,size_t,hipStream_t,void**,void**,hipEvent_t,hipEvent_t,uint32_t);
    static auto next=reinterpret_cast<Launch>(dlsym(RTLD_NEXT,"hipExtModuleLaunchKernel"));
    if(!next)return hipErrorUnknown;observe(f,gx,gy,gz,bx,by,bz,extra);
    return next(f,gx,gy,gz,bx,by,bz,shared,stream,params,extra,start,stop,flags);
}
