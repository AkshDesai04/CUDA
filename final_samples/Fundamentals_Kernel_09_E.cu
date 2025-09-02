```cuda
/*
 * Aim of the program: Create a __host__ __device__ function (a function that can be called from both host and device). Call it from both main and your kernel.
 *
 * Thinking:
 * 1. In CUDA, a function that can be called from both host (CPU) and device (GPU) must be declared with both __host__ and __device__ qualifiers.
 * 2. For demonstration, we implement a simple addition function `int add(int a, int b)` which returns the sum of two integers.
 * 3. In the `main()` function we call `add` on the host and print the result to verify host usage.
 * 4. We also launch a CUDA kernel that calls `add` on each thread, stores the result into a device array, and copy it back to the host to verify device usage.
 * 5. The kernel is defined with `__global__`, and it takes a pointer to an int array and the number of elements. Each thread computes the sum of two constants (e.g., 10 and 20) using `add` and writes the result.
 * 6. Memory allocation and deallocation are handled with `cudaMalloc`/`cudaMemcpy`/`cudaFree`.
 * 7. Error checking is minimal but sufficient for this small demo.
 * 8. Compile with: `nvcc -o demo demo.cu` and run `./demo`.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// __host__ __device__ function that can be called from both host and device
__host__ __device__ int add(int a, int b) {
    return a + b;
}

// Kernel that calls the __host__ __device__ function
__global__ void kernel_add(int *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Call the same function on the device
        out[idx] = add(10, 20);  // example: sum of 10 and 20
    }
}

int main() {
    // Host call to the __host__ __device__ function
    int hostResult = add(3, 4);
    printf("Host call to add(3, 4) = %d\n", hostResult);

    // Device call via kernel
    const int N = 8;
    int *d_out = NULL;
    int h_out[N];

    cudaError_t err = cudaMalloc((void**)&d_out, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel: 1 block of N threads (or adjust as needed)
    kernel_add<<<1, N>>>(d_out, N);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    // Copy results back to host
    err = cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    // Print results
    printf("Device call to add(10, 20) via kernel (each thread):\n");
    for (int i = 0; i < N; ++i) {
        printf("  Thread %d: %d\n", i, h_out[i]);
    }

    cudaFree(d_out);
    return 0;
}
```