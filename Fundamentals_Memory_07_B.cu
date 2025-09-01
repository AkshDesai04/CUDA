```cuda
/*
Aim of the program: 
Now do the reverse: allocate host memory with `malloc`, pass the host pointer to a kernel, and have the kernel try to dereference it. What happens? (This can cause an error or unspecified behavior).

My thinking:
- In CUDA, device code runs on the GPU and can only directly access device memory. A pointer obtained by `malloc` on the host points to host virtual memory that the GPU cannot reach unless it is explicitly made visible via a special API (e.g., `cudaHostAlloc` or `cudaMemcpy` into device memory).
- Passing a raw host pointer to a kernel will give the kernel a value that it will interpret as a device address. Since this address does not correspond to any valid location in the device address space, any attempt to dereference it will most likely lead to an illegal memory access on the GPU.
- On modern GPUs, the kernel launch will succeed (the driver will not reject the pointer at launch time), but during execution, when the kernel tries to read/write to that address, it will trigger an error. The error is typically reported as "invalid device pointer" or "invalid memory access".
- We can capture this by checking `cudaGetLastError()` after kernel launch and `cudaDeviceSynchronize()` to force execution completion and propagate the error.
- The program demonstrates that host memory cannot be accessed directly from device code without proper steps (pinning, mapping, or copying).

The code below allocates an array on the host with `malloc`, passes the pointer to a trivial kernel that writes to the first element, and then prints any error that occurs. It shows that the kernel fails because it dereferences a host pointer that is not visible to the GPU.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel that attempts to write to a location pointed to by a host pointer
__global__ void writeToHostPtr(int *hostPtr)
{
    // Attempt to write a value to the first element
    // This will likely cause an invalid memory access on the GPU
    hostPtr[0] = 42;
}

int main()
{
    const int N = 10;
    // Allocate host memory with standard malloc
    int *h_array = (int*)malloc(N * sizeof(int));
    if (!h_array) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }
    // Initialize host array (not strictly necessary)
    for (int i = 0; i < N; ++i) {
        h_array[i] = i;
    }

    // Launch kernel with the host pointer
    writeToHostPtr<<<1, 1>>>(h_array);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to force the kernel to run and capture any runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution error (synchronize): %s\n", cudaGetErrorString(err));
    }

    // Free host memory
    free(h_array);

    // Reset device before exiting
    cudaDeviceReset();

    return 0;
}
```