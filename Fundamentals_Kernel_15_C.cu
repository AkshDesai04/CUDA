```cuda
/* 
Aim of the program:
Modify the infinite loop kernel to take a pointer `int* flag`. The loop should be `while(*flag == 0);`. From the host, after a delay (`sleep(1)`), copy a value of `1` to the flag on the device to stop the kernel. (This demonstrates device-side polling).

Thinking process:
1. Allocate a single integer on the device to serve as the flag.
2. Initialize it to 0 before launching the kernel.
3. The kernel will simply spin in a while loop until the flag becomes non-zero.
4. After launching the kernel, the host sleeps for one second to simulate a delay.
5. The host then copies the value 1 to the device flag, which will cause all threads to exit the loop.
6. Use `cudaDeviceSynchronize()` to wait for the kernel to finish.
7. Add a small device printf to confirm that the kernel exited the loop.
8. Include basic CUDA error checking and compile the code with device printf support.
*/

#include <stdio.h>
#include <unistd.h>   // for sleep()
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel that polls the flag until it becomes non-zero
__global__ void poll_kernel(int *flag)
{
    // Each thread spins until flag != 0
    while (*flag == 0)
    {
        // Do nothing; just wait
    }

    // Let only one thread print a message
    if (threadIdx.x == 0)
    {
        printf("Kernel finished polling. flag=%d\n", *flag);
    }
}

int main(void)
{
    int *d_flag = NULL;
    int h_flag = 0;

    // Allocate flag on device
    CUDA_CHECK(cudaMalloc((void**)&d_flag, sizeof(int)));

    // Initialize flag to 0 on device
    CUDA_CHECK(cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel with 1 block and 1 thread (for simplicity)
    poll_kernel<<<1, 1>>>(d_flag);

    // Give the kernel time to start and begin polling
    sleep(1);

    // Set flag to 1 to stop the kernel
    h_flag = 1;
    CUDA_CHECK(cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice));

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up
    CUDA_CHECK(cudaFree(d_flag));

    printf("Host: Program completed.\n");

    return 0;
}
```