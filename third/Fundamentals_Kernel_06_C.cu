```cuda
/*
Aim of the program:
Modify the kernel to increment the value at `data[0]`. The host should initialize the value to 100, copy it to the device, launch the kernel to increment it, copy it back, and verify the result is 101.

Thinking:
- The program is a minimal CUDA example that demonstrates memory transfer between host and device and kernel execution.
- We declare a kernel that receives a pointer to an int array on the device. The kernel will simply increment the first element: `data[0]++`. Since we only need to modify a single element, we can launch a single thread.
- The host allocates a single int, sets it to 100, and allocates device memory for one int. We then copy the host value to the device using `cudaMemcpy`.
- Launch the kernel with a single block of a single thread: `increment<<<1, 1>>>(dev_data);`.
- After kernel execution, copy the result back to the host and verify that it equals 101. We print success or failure accordingly.
- Basic CUDA error checking is included for robustness, but is kept simple to focus on the main flow.
- The entire code is self-contained in a single .cu file and can be compiled with `nvcc` and executed on a CUDA-capable GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that increments the first element of an int array
__global__ void increment(int *data)
{
    // Only one thread needed
    data[0]++;    // Increment the value at index 0
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    #call, __FILE__, __LINE__,                    \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void)
{
    int h_data = 100;         // Host data initialized to 100
    int *d_data = NULL;       // Device pointer

    // Allocate device memory for one int
    CUDA_CHECK(cudaMalloc((void **)&d_data, sizeof(int)));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel: single block, single thread
    increment<<<1, 1>>>(d_data);

    // Wait for GPU to finish before accessing on host
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    if (h_data == 101)
        printf("Success: data[0] incremented to %d\n", h_data);
    else
        printf("Failure: expected 101 but got %d\n", h_data);

    // Clean up
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
```