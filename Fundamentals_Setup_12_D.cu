/*
Define a kernel that takes an `int*` argument, but try to launch it without providing any arguments.
---

**Thinking and Design**

1. **Goal**  
   The user wants a CUDA kernel that formally declares an `int*` parameter but is launched without explicitly passing that argument from the host side.  
   In standard CUDA, calling a kernel without providing the required arguments results in a compiler error.  
   To satisfy the requirement while still allowing the program to compile and run, we can give the kernel parameter a default value (`nullptr`).  
   This means the kernel signature still contains the `int*` argument, but callers may omit it and the compiler will supply the default.

2. **Kernel Implementation**  
   - The kernel will check if the pointer is `nullptr`.  
   - If it is not `nullptr`, it will write a value to the location it points to.  
   - If it is `nullptr`, it will simply print a message indicating that the argument was omitted.

3. **Host Code**  
   - Allocate a device integer with `cudaMalloc`.  
   - Launch the kernel **without** passing any arguments: `myKernel<<<1,1>>>();`.  
   - Synchronize the device to ensure completion.  
   - Copy back the result from device to host to verify whether the kernel attempted to write to the memory (it should not because we didn't pass a pointer).  
   - Print the host value to confirm it remains unchanged (initialized to zero).  

4. **Error Checking**  
   A simple macro `CHECK_CUDA` will wrap CUDA API calls to report errors promptly.

5. **Compilation**  
   The code is self‑contained in a single `.cu` file and can be compiled with `nvcc`.

*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple macro for CUDA error checking
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Kernel that takes an int* argument, but provides a default value
__global__ void myKernel(int *ptr = nullptr)
{
    // Thread index (not really needed here, but good practice)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // If the pointer is not null, write a value; otherwise, print a message
    if (ptr)
    {
        // Safely write to the device memory
        *ptr = 42;
    }
    else
    {
        // Using device printf requires compute capability >= 2.0
        printf("Kernel launched without providing the int* argument (ptr == nullptr). Thread %d\n", tid);
    }
}

int main(void)
{
    // Allocate device memory for an integer
    int *d_int = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_int, sizeof(int)));

    // Initialize the device memory to zero
    CHECK_CUDA(cudaMemset(d_int, 0, sizeof(int)));

    // Launch the kernel **without** passing any arguments
    // The kernel parameter has a default value (nullptr), so this is legal
    myKernel<<<1, 1>>>();
    // Check for any launch errors
    CHECK_CUDA(cudaGetLastError());

    // Synchronize to make sure the kernel has finished
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy the result back to host
    int h_int = 0;
    CHECK_CUDA(cudaMemcpy(&h_int, d_int, sizeof(int), cudaMemcpyDeviceToHost));

    // Print the value retrieved from device
    printf("Value in device memory after kernel launch: %d\n", h_int);

    // Clean up
    CHECK_CUDA(cudaFree(d_int));

    return 0;
}
