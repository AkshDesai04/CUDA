```cpp
/*
Use `cudaGetSymbolAddress` to get a device pointer to the `__device__` variable, which you can then pass to other API functions like `cudaMemset`.

The idea behind this example is to illustrate how CUDA allows host code to obtain a raw device pointer for a global device variable declared with `__device__`. Once we have this pointer, we can manipulate the device memory using standard runtime API calls such as `cudaMemset` or `cudaMemcpy`. 
In this example we:

1. Declare a global device variable `devVar` and initialize it.
2. In host code, call `cudaGetSymbolAddress` to retrieve a device pointer to `devVar`.
3. Copy the initial value of `devVar` back to the host and print it.
4. Use `cudaMemset` to set the device memory to zero.
5. Copy the updated value back to the host and print it again to confirm the operation succeeded.

We also include basic error checking to ensure each CUDA call succeeds. The program can be compiled with `nvcc` and executed on a CUDA-capable device.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = (call);                                \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

// Global device variable initialized to 42
__device__ int devVar = 42;

int main(void) {
    // Obtain device pointer to the __device__ variable
    void *devPtr = nullptr;
    CHECK_CUDA(cudaGetSymbolAddress(&devPtr, devVar));

    // Copy initial value from device to host and print
    int hostVal = 0;
    CHECK_CUDA(cudaMemcpy(&hostVal, devPtr, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Initial device variable value: %d\n", hostVal);

    // Use cudaMemset to set the device memory to zero
    CHECK_CUDA(cudaMemset(devPtr, 0, sizeof(int)));

    // Copy updated value from device to host and print
    CHECK_CUDA(cudaMemcpy(&hostVal, devPtr, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Device variable value after cudaMemset: %d\n", hostVal);

    // Clean up and exit
    return 0;
}
```