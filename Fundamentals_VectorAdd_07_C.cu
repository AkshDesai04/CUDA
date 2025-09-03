```cuda
/*
Implement an in-place version: `data[i] = data[i] * data[i]`.

I want to write a small CUDA program that demonstrates how to perform an in‑place
element‑wise squaring of an array. The goal is to multiply each element by itself
directly in the same memory location, rather than writing to a different array.

Plan:
1. Create a simple host array of floats (or ints) and initialize it.
2. Allocate device memory and copy the array to the device.
3. Define a CUDA kernel `squareInPlace` that takes a pointer to the data
   and its size. Each thread will compute the index it is responsible for
   and perform `data[i] = data[i] * data[i]`.
4. Launch the kernel with a reasonable number of threads per block.
5. Copy the result back to the host and print it to verify the operation.
6. Clean up device memory.

Considerations:
- I'll use `float` for simplicity, but the same code would work with `int`.
- Boundary checks are necessary in the kernel to avoid out‑of‑bounds
  memory accesses when the number of elements is not a multiple of the block size.
- Error checking for CUDA calls will be added to make the program robust.

This code is self‑contained and can be compiled with `nvcc`:

    nvcc -o square_inplace square_inplace.cu

Then run it:

    ./square_inplace

It will output the original and squared arrays to the console.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that squares each element of the array in-place
__global__ void squareInPlace(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        data[idx] = val * val;
    }
}

// Helper function to check CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    // Example data
    const int N = 10;
    float h_data[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = (float)(i + 1);  // 1.0, 2.0, ..., 10.0
    }

    // Print original data
    printf("Original data:\n");
    for (int i = 0; i < N; ++i) {
        printf("%.1f ", h_data[i]);
    }
    printf("\n");

    // Allocate device memory
    float *d_data;
    checkCudaError(cudaMalloc((void **)&d_data, N * sizeof(float)), "cudaMalloc");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // Launch kernel: use 256 threads per block as a typical choice
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    squareInPlace<<<blocks, threadsPerBlock>>>(d_data, N);

    // Wait for GPU to finish before accessing on host
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    // Print squared data
    printf("Squared data (in-place):\n");
    for (int i = 0; i < N; ++i) {
        printf("%.1f ", h_data[i]);
    }
    printf("\n");

    // Clean up
    checkCudaError(cudaFree(d_data), "cudaFree");

    return 0;
}
```