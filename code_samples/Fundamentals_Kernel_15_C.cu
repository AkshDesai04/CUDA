```cpp
/*
Modify the infinite loop kernel to take a pointer `int* flag`. The loop should be `while(*flag == 0);`. From the host, after a delay (`sleep(1)`), copy a value of `1` to the flag on the device to stop the kernel. (This demonstrates device-side polling).

Thought process:
- The kernel must run indefinitely until a flag on the device memory becomes non-zero.
- We'll allocate an integer on the device (`d_flag`), initialize it to 0, and pass its pointer to the kernel.
- The kernel will spin with `while (*flag == 0);`. This is a busy-wait loop that will exit once the host writes a 1 to the flag.
- On the host side, after launching the kernel, we wait for 1 second using `sleep(1)`.
- Then we set a host variable `h_flag` to 1 and copy it to `d_flag` with `cudaMemcpy`.
- Finally, we synchronize the device to ensure the kernel has finished, then clean up resources.
- The program will print messages to indicate progress and successful termination.
*/

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA Error [%s:%d]: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Kernel that spins until *flag becomes non-zero
__global__ void infiniteLoopKernel(int *flag) {
    // Each thread can perform the same busy wait; we use only one thread here
    while (*flag == 0) {
        // Optionally, we could use __nanosleep or __threadfence to reduce load,
        // but for demonstration a simple loop suffices.
    }
    // Optionally, print from the device (requires device printf support)
    // printf("Kernel exiting.\n");
}

int main() {
    // Allocate device flag
    int *d_flag;
    CUDA_CHECK(cudaMalloc((void**)&d_flag, sizeof(int)));

    // Initialize flag to 0 on the device
    int h_flag = 0;
    CUDA_CHECK(cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice));

    // Launch the kernel with a single thread
    infiniteLoopKernel<<<1, 1>>>(d_flag);
    CUDA_CHECK(cudaGetLastError());

    printf("Kernel launched; flag set to 0.\n");

    // Wait for 1 second
    sleep(1);

    // Set flag to 1 to stop the kernel
    h_flag = 1;
    CUDA_CHECK(cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice));
    printf("Host set flag to 1; waiting for kernel to finish.\n");

    // Synchronize to ensure kernel completion
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Kernel has terminated.\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_flag));

    printf("Program completed successfully.\n");
    return 0;
}
```