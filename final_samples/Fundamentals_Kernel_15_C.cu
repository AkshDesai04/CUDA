```cuda
/*
Aim of the program:
Modify the infinite loop kernel to take a pointer `int* flag`. The loop should be `while(*flag == 0);`. From the host, after a delay (`sleep(1)`), copy a value of `1` to the flag on the device to stop the kernel. (This demonstrates device-side polling).

Thinking:
The task requires a kernel that loops until a flag on the device changes from 0 to 1. The flag will be allocated on the device, initialized to 0, and passed by pointer to the kernel. The kernel will busy‑wait in a `while (*flag == 0);` loop. After launching the kernel, the host will sleep for one second using `sleep(1)` from `<unistd.h>`. Then it will set the host copy of the flag to 1 and copy this value back to the device memory. When the kernel reads the updated flag it will exit the loop. We use `cudaDeviceSynchronize()` to wait for the kernel to finish and then clean up resources. For simplicity we launch a single thread, but any grid configuration works. Device side printf is optional; the kernel prints a message when exiting the loop to confirm termination. Error checking is added for robustness.
*/
#include <cstdio>
#include <cstdlib>
#include <unistd.h>      // for sleep
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__,                     \
                    cudaGetErrorString(err));                      \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel that polls the device flag
__global__ void pollingKernel(int *flag)
{
    // Busy-wait until flag becomes non-zero
    while (*flag == 0)
        ;
    // Optional: indicate completion (requires device printf support)
    printf("Kernel detected flag change, exiting loop.\n");
}

int main()
{
    int *d_flag = nullptr;
    int h_flag = 0;

    // Allocate device memory for flag
    CUDA_CHECK(cudaMalloc((void**)&d_flag, sizeof(int)));

    // Initialize flag to 0 on device
    CUDA_CHECK(cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice));

    // Launch the kernel with one thread
    pollingKernel<<<1, 1>>>(d_flag);
    CUDA_CHECK(cudaGetLastError());

    // Host sleeps for 1 second before signaling the kernel
    sleep(1);

    // Update flag to 1 on host and copy to device
    h_flag = 1;
    CUDA_CHECK(cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice));

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up
    CUDA_CHECK(cudaFree(d_flag));

    printf("Host: kernel finished, program exiting.\n");
    return 0;
}
```