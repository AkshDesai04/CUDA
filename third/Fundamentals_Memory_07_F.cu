```cpp
/*
Aim of the program:
Write a __host__ function that takes an int* and dereferences it. Call it with a device pointer. Observe the segfault.

Thinking:
1. Allocate an integer on the device using cudaMalloc.
2. Pass that device pointer to a host function that attempts to dereference it.
3. Because the pointer refers to device memory, the host cannot access it; this should result in a segmentation fault when the host tries to read or write the memory.
4. The program includes minimal error checking to illustrate the failure point.
5. The host function is explicitly annotated with __host__ (though this is the default) to emphasize that it runs on the CPU.
6. The program is a single .cu file that can be compiled with nvcc and run on a machine with a CUDA-capable GPU.
*/

#include <iostream>
#include <cuda_runtime.h>

__host__ void hostDereference(int* ptr)
{
    // Attempt to read the value at the device pointer
    // This will cause a segmentation fault because ptr points to device memory
    std::cout << "Attempting to dereference device pointer on host." << std::endl;
    int val = *ptr;  // <-- Faulty dereference
    std::cout << "Value: " << val << std::endl;
}

int main()
{
    int *d_ptr = nullptr;
    size_t size = sizeof(int);

    // Allocate memory on the device
    cudaError_t err = cudaMalloc((void**)&d_ptr, size);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Optional: Initialize the device memory to a known value
    int hostVal = 42;
    err = cudaMemcpy(d_ptr, &hostVal, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy to device failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_ptr);
        return 1;
    }

    // Call the host function with the device pointer
    // This will cause a segmentation fault
    hostDereference(d_ptr);

    // Cleanup (will likely not be reached if segfault occurs)
    cudaFree(d_ptr);

    return 0;
}
```