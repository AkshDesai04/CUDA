```cuda
/* 
Aim of the program: 
Modify the infinite loop kernel to take a pointer `int* flag`. The loop should be `while(*flag == 0);`. From the host, after a delay (`sleep(1)`), copy a value of `1` to the flag on the device to stop the kernel. (This demonstrates device-side polling).

Thinking: 
We want to show how a kernel can poll a flag in device memory and be stopped by the host. The kernel will simply loop until the flag changes. Since the kernel runs on the GPU, we use a simple `while(*flag == 0);` which will spin until the host writes a non‑zero value. After launching the kernel we pause the host for one second (using `sleep(1)` from `<unistd.h>`), then copy `1` to the flag using `cudaMemcpy` from host to device. Finally we call `cudaDeviceSynchronize()` to ensure the kernel has finished before the program exits. We also print a message from the kernel after the loop to confirm it stopped. This code includes basic error checking for CUDA calls and cleanly frees device memory.
*/
#include <stdio.h>
#include <unistd.h>   // for sleep()
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void infinite_loop_kernel(int *flag) {
    // Simple spin-wait until flag becomes non-zero
    while (*flag == 0);
    printf("Kernel finished: flag=%d\n", *flag);
}

int main(void) {
    int h_flag = 0;
    int *d_flag = NULL;

    // Allocate flag on device
    CHECK_CUDA(cudaMalloc((void**)&d_flag, sizeof(int)));
    // Initialize flag to 0 on device
    CHECK_CUDA(cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel with 1 block and 1 thread
    infinite_loop_kernel<<<1, 1>>>(d_flag);
    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Wait for 1 second before changing the flag
    sleep(1);

    // Set flag to 1 on host
    h_flag = 1;
    // Copy new flag value to device
    CHECK_CUDA(cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice));

    // Wait for kernel to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Host: Kernel has terminated.\n");

    // Clean up
    CHECK_CUDA(cudaFree(d_flag));

    return 0;
}
```