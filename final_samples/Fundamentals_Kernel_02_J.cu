```cpp
/* 
Pass a string (char*) from the host to the kernel and try to print it. Observe what happens and why it doesn't work as expected. (Hint: pointer points to host memory).

The idea of this small program is to demonstrate a common pitfall when using CUDA: device kernels cannot dereference pointers that point to host memory. In C/C++ a `char*` that is allocated or points to a literal in host memory is not automatically accessible from device code. If we simply pass that pointer to a kernel and try to print the string using `printf`, the kernel will either produce garbage, crash, or do nothing, because the device has no valid mapping for that host address.

To show this, the program:
1. Declares a host string literal and a pointer to it.
2. Launches a kernel that receives this pointer and attempts to print it with `printf`.
3. Synchronizes the device and checks for errors after the launch.
4. Prints the error code (if any) to the console, allowing us to see that the kernel failed due to invalid memory access.

The expected outcome on most systems is that the kernel will print nothing or will produce a segmentation fault, and the host will report an error such as `invalid device pointer` or `invalid value`. We also demonstrate the correct approach by allocating device memory, copying the string there, and then printing from the kernel, showing that the string is correctly printed only after proper transfer to device memory.

This code is intentionally minimal and self‑contained; it compiles with any recent CUDA toolkit using `nvcc`. It also contains extensive comments to explain each step, why it fails, and how to fix it.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Kernel that attempts to print a string passed from the host
__global__ void printString(const char *s)
{
    // Device printf requires the format string to be known at compile time
    // The string `s` is expected to point to device memory.
    // If `s` actually points to host memory, this will cause an error.
    printf("Device sees string: %s\n", s);
}

// Helper macro to check CUDA errors
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

int main()
{
    // Host string literal (stored in read-only memory on many systems)
    const char *hostStr = "Hello from host!";

    printf("Launching kernel with host string pointer: %p\n", (void*)hostStr);

    // Launch kernel: pass the host pointer directly
    printString<<<1, 1>>>(hostStr);

    // Synchronize and check for launch errors
    CHECK_CUDA(cudaDeviceSynchronize());

    // Attempt to print device error state after launch
    cudaError_t syncErr = cudaGetLastError();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(syncErr));
    } else {
        printf("Kernel executed successfully (unexpected).\n");
    }

    // Now show the correct way: copy string to device memory
    char *deviceStr;
    size_t len = strlen(hostStr) + 1; // include null terminator
    CHECK_CUDA(cudaMalloc((void**)&deviceStr, len));
    CHECK_CUDA(cudaMemcpy(deviceStr, hostStr, len, cudaMemcpyHostToDevice));

    printf("Launching kernel with device string pointer: %p\n", (void*)deviceStr);
    printString<<<1, 1>>>(deviceStr);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Clean up
    CHECK_CUDA(cudaFree(deviceStr));

    return 0;
}
```