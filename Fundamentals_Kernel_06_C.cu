```cuda
/*
Aim of the program: Modify the kernel to increment the value at `data[0]`. 
The host should initialize the value to 100, copy it to the device, launch the 
kernel to increment it, copy it back, and verify the result is 101.

Thinking:
- We need a simple CUDA kernel that accesses a device array and increments 
  the first element. The kernel will be invoked with a single thread, as we 
  only modify a single value. 
- The host code will:
  1. Initialize a host integer to 100.
  2. Allocate device memory for a single int.
  3. Copy the host value to the device.
  4. Launch the kernel with 1 block of 1 thread.
  5. Copy the result back to the host.
  6. Verify that the value is 101 and print a success/failure message.
- We will add a small error checking macro to wrap CUDA API calls.
- The program is self‑contained and can be compiled with `nvcc` and run on a 
  system with a CUDA-capable GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",    \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),         \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel that increments data[0]
__global__ void incrementKernel(int *data)
{
    // Only one thread needed
    data[0] += 1;
}

int main(void)
{
    int host_val = 100;
    int *dev_val = NULL;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&dev_val, sizeof(int)));

    // Copy host value to device
    CUDA_CHECK(cudaMemcpy(dev_val, &host_val, sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel: 1 block, 1 thread
    incrementKernel<<<1, 1>>>(dev_val);

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&host_val, dev_val, sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(dev_val));

    // Verify the result
    if (host_val == 101) {
        printf("Success: value incremented to %d\n", host_val);
    } else {
        printf("Failure: expected 101, got %d\n", host_val);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```