```cpp
/*
Query and print the maximum layered 1D texture size and number of layers.

Thinking:
1. The CUDA Runtime API provides device properties via cudaDeviceProp.
2. The struct contains a field called `maxTexture1DLayered`, which is an array of two ints:
   - `maxTexture1DLayered[0]` holds the maximum width of a 1D layered texture.
   - `maxTexture1DLayered[1]` holds the maximum number of layers for a 1D layered texture.
3. The program should query all CUDA devices on the system, retrieve their properties,
   and print the two values above for each device.
4. Error handling will be added via a simple macro that checks the return status of CUDA API calls.
5. The program uses standard C++ I/O (`std::cout`) for readability.
6. The code is self‑contained in a single .cu file and can be compiled with `nvcc`.

*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__     \
                      << " : " << cudaGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

int main()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return EXIT_FAILURE;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        std::cout << "Device " << dev << " (" << prop.name << "):\n";
        std::cout << "  Max 1D layered texture width:   " 
                  << prop.maxTexture1DLayered[0] << " texels\n";
        std::cout << "  Max 1D layered texture layers:  " 
                  << prop.maxTexture1DLayered[1] << "\n\n";
    }

    return EXIT_SUCCESS;
}
```