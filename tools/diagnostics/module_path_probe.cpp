// Record filenames passed to the runtime loader, without reading/copying ELF.
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <array>
#include <cstring>
#include <unordered_map>

static std::unordered_map<hipModule_t,std::array<unsigned char,16>>& headers() {
    static std::unordered_map<hipModule_t,std::array<unsigned char,16>> entries;
    return entries;
}

extern "C" hipError_t hipModuleLoad(hipModule_t* module, const char* filename) {
    using Load = hipError_t (*)(hipModule_t*, const char*);
    static auto next = reinterpret_cast<Load>(dlsym(RTLD_NEXT, "hipModuleLoad"));
    if (!next) return hipErrorUnknown;
    auto status = next(module, filename);
    std::fprintf(stderr, "MODULE_PATH\t%d\t%s\n", int(status), filename ? filename : "<null>");
    std::fprintf(stderr, "MODULE_HANDLE\t%p\t%s\n", status==hipSuccess?*module:nullptr, filename ? filename : "<null>");
    return status;
}

extern "C" hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
    using Load = hipError_t (*)(hipModule_t*, const void*);
    static auto next = reinterpret_cast<Load>(dlsym(RTLD_NEXT,"hipModuleLoadData"));
    if (!next) return hipErrorUnknown;
    auto status=next(module,image);
    Dl_info caller{};dladdr(__builtin_return_address(0),&caller);
    if(status==hipSuccess)std::memcpy(headers()[*module].data(),image,16);
    std::fprintf(stderr,"MODULE_DATA\t%d\t%p\t%s\n",int(status),status==hipSuccess?*module:nullptr,caller.dli_fname?caller.dli_fname:"<unknown>");
    return status;
}

extern "C" hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                                        unsigned count, hipJitOption* options, void** values) {
    using Load = hipError_t (*)(hipModule_t*,const void*,unsigned,hipJitOption*,void**);
    static auto next = reinterpret_cast<Load>(dlsym(RTLD_NEXT,"hipModuleLoadDataEx"));
    if (!next) return hipErrorUnknown;
    auto status=next(module,image,count,options,values);
    std::fprintf(stderr,"MODULE_DATA_EX\t%d\t%p\n",int(status),status==hipSuccess?*module:nullptr);
    return status;
}

extern "C" hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* name) {
    using Get = hipError_t (*)(hipFunction_t*,hipModule_t,const char*);
    static auto next = reinterpret_cast<Get>(dlsym(RTLD_NEXT,"hipModuleGetFunction"));
    if (!next) return hipErrorUnknown;
    auto status=next(function,module,name);
    std::fprintf(stderr,"MODULE_FUNCTION\t%d\t%p\t%s\n",int(status),module,name?name:"<null>");
    if(status==hipSuccess && name && std::strstr(name,"_MT") && headers().count(module)) {
        std::fprintf(stderr,"MODULE_HEADER\t%p\t",module);
        for(auto byte:headers()[module])std::fprintf(stderr,"%02x",unsigned(byte));
        std::fprintf(stderr,"\n");
    }
    return status;
}
