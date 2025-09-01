/*
Calculate and print the theoretical maximum number of concurrent threads the entire GPU can run (`maxThreadsPerMultiProcessor` * `multiProcessorCount`).

Thoughts:
- The goal is to query CUDA device properties to retrieve `maxThreadsPerMultiProcessor` and `multiProcessorCount`.
- The theoretical maximum concurrent threads for the whole GPU is simply the product of these two values.
- We'll use `cudaGetDeviceProperties` on the default device (device 0). 
- Error handling: check the CUDA API calls for failures.
- Output the result using `std::cout`.
- The program will compile with `nvcc` as a `.cu` file.
- This program assumes at least one CUDA-capable device is present.
- We'll keep the code minimal and straightforward.
*/

#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        std::cerr << "Error retrieving device count: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return 1;
    }

    // Use the first device
    int device = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        std::cerr << "Error retrieving device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Calculate theoretical maximum concurrent threads
    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int mpCount = prop.multiProcessorCount;
    long long totalMaxThreads = static_cast<long long>(maxThreadsPerSM) * mpCount;

    std::cout << "Device " << device << ": " << prop.name << std::endl;
    std::cout << "maxThreadsPerMultiProcessor: " << maxThreadsPerSM << std::endl;
    std::cout << "multiProcessorCount: " << mpCount << std::endl;
    std::cout << "Theoretical maximum concurrent threads for the entire GPU: "
              << totalMaxThreads << std::endl;

    return 0;
}
