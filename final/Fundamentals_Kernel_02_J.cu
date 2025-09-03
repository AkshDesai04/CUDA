```cuda
/*
Aim of the program (word to word as requested):
Pass a string (char*) from the host to the kernel and try to print it. Observe what happens and why it doesn't work as expected. (Hint: pointer points to host memory).

Thinking:
- In CUDA, memory is separated into host (CPU) and device (GPU) spaces.
- A pointer allocated on the host points to host memory; it is not valid in device code.
- If we pass such a pointer directly to a kernel, the device will interpret it as an address in its own address space, leading to undefined behavior (often no output or garbage).
- To correctly use a string on the device, we need to copy the string to device memory first (e.g., with cudaMemcpy or cudaMallocManaged).
- For demonstration, we will allocate a string in host memory, pass the pointer to the kernel, attempt to print it, and observe that the output is not as expected.
- After that, we will show the correct way: allocate device memory, copy the string, and pass the device pointer.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printStringKernel(const char *dev_str)
{
    // Attempt to print the string pointed to by dev_str
    // This may print garbage or nothing because dev_str points to host memory.
    printf("Kernel: The string is: %s\n", dev_str);
}

int main()
{
    const char *hostString = "Hello from host string!";
    const char *deviceStringPtr = hostString; // Intentionally wrong: pointer points to host memory

    printf("Host: About to launch kernel with host pointer %p\n", (void*)deviceStringPtr);

    // Launch kernel with incorrect pointer
    printStringKernel<<<1, 1>>>(deviceStringPtr);
    cudaDeviceSynchronize();

    // -------------------------------------------------------------
    // Correct approach: allocate device memory, copy the string, and pass device pointer

    // Determine string length
    size_t len = strlen(hostString) + 1; // +1 for null terminator

    // Allocate device memory
    char *devString;
    cudaMalloc((void**)&devString, len);

    // Copy string from host to device
    cudaMemcpy(devString, hostString, len, cudaMemcpyHostToDevice);

    printf("Host: About to launch kernel with device pointer %p\n", (void*)devString);

    // Launch kernel with correct device pointer
    printStringKernel<<<1, 1>>>(devString);
    cudaDeviceSynchronize();

    // Clean up
    cudaFree(devString);

    return 0;
}
```