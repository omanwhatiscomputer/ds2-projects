

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
    #include <windows.h>
    #define LIB_HANDLE HMODULE
    #define LIB_LOAD(name) LoadLibrary(name)
    #define LIB_GETSYM(lib, name) GetProcAddress(lib, name)
    #define LIB_CLOSE(lib) FreeLibrary(lib)
#else
    #include <dlfcn.h>
    #define LIB_HANDLE void*
    #define LIB_LOAD(name) dlopen(name, RTLD_LAZY)
    #define LIB_GETSYM(lib, name) dlsym(lib, name)
    #define LIB_CLOSE(lib) dlclose(lib)
#endif

// Function pointer types for the CUDA Runtime API functions we need
typedef int    (*cudaGetDeviceCount_t)(int*);
typedef const char* (*cudaGetErrorString_t)(int);

// Helper: try to load the CUDA runtime library and get symbols
static bool load_cuda_library(LIB_HANDLE* handle,
                              cudaGetDeviceCount_t* cudaGetDeviceCount_fn,
                              cudaGetErrorString_t* cudaGetErrorString_fn) {
    const char* lib_names[] = {
#ifdef _WIN32
        "nvcuda.dll",
        "cudart64_12.dll",
        "cudart64_11.dll",
        "cudart64_10.dll",
#else
        "libcudart.so",         // Default symlink (points to latest)
        "libcudart.so.12",
        "libcudart.so.11",
        "libcudart.so.10",
#endif
        NULL
    };

    for (int i = 0; lib_names[i] != NULL; ++i) {
        LIB_HANDLE h = LIB_LOAD(lib_names[i]);
        if (h) {
            // Try to get all required function pointers
            *cudaGetDeviceCount_fn       = (cudaGetDeviceCount_t)LIB_GETSYM(h, "cudaGetDeviceCount");
            *cudaGetErrorString_fn       = (cudaGetErrorString_t)LIB_GETSYM(h, "cudaGetErrorString");

            if (*cudaGetDeviceCount_fn && *cudaGetErrorString_fn) {
                *handle = h;
                return true;
            }
            // Missing symbols – close and try next library name
            LIB_CLOSE(h);
        }
    }
    return false;
}

bool cuda_is_available(void) {
    LIB_HANDLE cuda_lib = NULL;
    cudaGetDeviceCount_t cudaGetDeviceCount = NULL;
    cudaGetErrorString_t cudaGetErrorString = NULL;

    if (!load_cuda_library(&cuda_lib, &cudaGetDeviceCount, &cudaGetErrorString)) {
        return false;   // CUDA runtime not found
    }

    int deviceCount = 0;
    int err = cudaGetDeviceCount(&deviceCount);

    // Typical “no CUDA” errors
    if (err == 100 /* cudaErrorNoDevice */ || err == 101 /* cudaErrorInsufficientDriver */) {
        LIB_CLOSE(cuda_lib);
        return false;
    }

    bool available = (err == 0 && deviceCount > 0);

    LIB_CLOSE(cuda_lib);
    return available;
}



