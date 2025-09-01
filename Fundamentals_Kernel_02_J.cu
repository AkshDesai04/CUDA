```cpp
/*
Aim of the program:
Pass a string (char*) from the host to the kernel and try to print it. Observe what happens and why it doesn't work as expected. (Hint: pointer points to host memory).

Thinking:
- In CUDA, device code cannot directly access host memory unless it is pinned or using unified memory.
- A simple char* allocated on host (either a literal or heap) points to host address space.
- Passing that pointer to a kernel will give the device a pointer that refers to host memory, which is invalid on the device.
- Attempting to dereference such a pointer in the kernel (e.g., via printf("%s", str)) will lead to undefined behavior: either garbage output, nothing, or a device fault.
- To correctly pass a string, we must copy it to device memory first (cudaMalloc + cudaMemcpy) and then pass the device pointer to the kernel.
- The program below demonstrates both: first, it passes the host pointer (bad) and prints from the kernel (will not show expected string), then it copies the string to device memory (good) and passes the device pointer, which prints correctly.
- The output will show the difference and illustrate why the host pointer does not work.
*/

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

__global__ void printStringKernel(const char *str)
{
    // Print the string passed from host (or device)
    printf("Kernel received string: %s\n", str);
}

int main()
{
    // Host string
    const char *h_str = "Hello from host!";

    // ----------------------------------------------------
    // 1. Pass the host pointer directly to the kernel
    // ----------------------------------------------------
    printf("=== Attempting to pass host pointer directly to kernel ===\n");
    printStringKernel<<<1, 1>>>(h_str);
    cudaDeviceSynchronize(); // Wait for kernel to finish

    // ----------------------------------------------------
    // 2. Copy the string to device memory and pass device pointer
    // ----------------------------------------------------
    size_t len = strlen(h_str) + 1; // Include null terminator
    char *d_str = nullptr;
    cudaMalloc((void **)&d_str, len);
    cudaMemcpy(d_str, h_str, len, cudaMemcpyHostToDevice);

    printf("\n=== Passing device pointer to kernel ===\n");
    printStringKernel<<<1, 1>>>(d_str);
    cudaDeviceSynchronize(); // Wait for kernel to finish

    // Clean up
    cudaFree(d_str);
    return 0;
}
```