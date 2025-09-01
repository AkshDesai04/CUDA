```cuda
/*
Aim of the program: Calculate and print the theoretical maximum number of concurrent warps the entire GPU can run.

Thinking:
- On a CUDA-capable GPU, the total theoretical concurrent warps that can be active at a time is determined by the number of Streaming Multiprocessors (SMs) and the maximum number of threads that can be resident on each SM.
- Each warp consists of 32 threads (warpSize). The maximum number of threads that can reside on an SM is given by the device property maxThreadsPerMultiProcessor.
- Therefore, the maximum number of warps that can be resident on one SM is:
        maxWarpsPerSM = maxThreadsPerMultiProcessor / warpSize
  (Since warpSize is always a divisor of maxThreadsPerMultiProcessor for supported GPUs, integer division is safe.)
- The theoretical maximum number of concurrent warps for the whole GPU is then:
        maxConcurrentWarps = numMultiProcessors * maxWarpsPerSM
- We will query the device properties using cudaGetDeviceProperties and compute this value for the default device (device 0). The program will handle errors gracefully and print the result in a human‑readable format.
- No kernel launch is necessary; we only need the device properties. The program is written in standard C++ with CUDA runtime API.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return 1;
    }

    // Use the first device (device 0)
    int device = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    int numSMs = prop.multiProcessorCount;
    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int warpSize = prop.warpSize; // Usually 32

    int maxWarpsPerSM = maxThreadsPerSM / warpSize;
    int maxConcurrentWarps = numSMs * maxWarpsPerSM;

    std::cout << "Device " << device << " : " << prop.name << std::endl;
    std::cout << "  Multi-Processor Count (SMs): " << numSMs << std::endl;
    std::cout << "  Max Threads per SM: " << maxThreadsPerSM << std::endl;
    std::cout << "  Warp Size: " << warpSize << std::endl;
    std::cout << "  Max Warps per SM: " << maxWarpsPerSM << std::endl;
    std::cout << "  Theoretical maximum number of concurrent warps across the GPU: "
              << maxConcurrentWarps << std::endl;

    return 0;
}
```