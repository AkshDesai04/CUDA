```cpp
/*
Aim: Write a kernel `increment_counter()` that has each thread perform
`atomicAdd(&my_global_counter, 1);`. Launch it, then launch a second kernel
`read_counter(int* out)` that has thread 0 write the value of
`my_global_counter` to an output variable for verification.

Thoughts:
1. Use a device-wide global variable `my_global_counter` initialized to 0.
   We can set it explicitly via `cudaMemcpyToSymbol` to ensure a clean start.
2. `increment_counter` kernel: each thread calls `atomicAdd(&my_global_counter, 1);`.
   For safety, we can launch enough threads and let all of them execute the
   atomic operation; no need for index bounds checking if the grid size
   exactly equals the desired number of increments.
3. `read_counter` kernel: thread 0 writes the global counter value to a
   device pointer passed from host. This is the only thread that accesses
   the output pointer.
4. In `main`, set up launch configuration, reset the counter, launch
   `increment_counter`, synchronize, then launch `read_counter` and copy
   the result back to host. Print the final counter value to verify correctness.
5. Basic error checking after kernel launches and memory copies will be added
   for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__device__ int my_global_counter = 0;

// Kernel that increments the global counter atomically
__global__ void increment_counter()
{
    // Each thread performs an atomic addition of 1
    atomicAdd(&my_global_counter, 1);
}

// Kernel that writes the global counter value to an output variable
// Only thread 0 writes to the output pointer
__global__ void read_counter(int* out)
{
    if (threadIdx.x == 0)
        *out = my_global_counter;
}

int main()
{
    // Number of threads to launch for the increment kernel
    const int threadsPerBlock = 256;
    const int numBlocks = 10; // Total threads = 2560

    // Reset the global counter to zero
    int zero = 0;
    cudaError_t err = cudaMemcpyToSymbol(my_global_counter, &zero, sizeof(int),
                                         0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Launch the increment kernel
    increment_counter<<<numBlocks, threadsPerBlock>>>();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "increment_counter launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for the kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize after increment_counter failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Allocate device memory for the output
    int *d_out;
    err = cudaMalloc((void**)&d_out, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Launch the read_counter kernel to copy the counter value to d_out
    read_counter<<<1, 1>>>(d_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "read_counter launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return -1;
    }

    // Wait for the read kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize after read_counter failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return -1;
    }

    // Copy the result back to host
    int h_out;
    err = cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy from device to host failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return -1;
    }

    // Free device memory
    cudaFree(d_out);

    // Print the result
    printf("Final counter value: %d (expected %d)\n", h_out, threadsPerBlock * numBlocks);

    return 0;
}
```