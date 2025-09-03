```cpp
/*
Aim of the program:
Implement vector-scalar addition: B[i] = A[i] + 5; where 5 is a scalar value passed to the kernel.

Thinking process:

1. Define the problem clearly – we need to add a scalar value to every element of an input vector A and store the result in vector B.
2. Decide data type – using float for simplicity.
3. Set a fixed vector size (e.g., N = 10) for demonstration purposes.
4. Allocate host memory for A and B, initialize A with known values.
5. Allocate device memory for A, B, and copy the input vector from host to device.
6. Write a CUDA kernel that:
   - Computes the global thread index.
   - Checks bounds (i < n).
   - Performs the addition: B[i] = A[i] + scalar.
7. Launch the kernel with a sufficient number of blocks and threads to cover all elements.
8. Synchronize and check for errors using cudaGetLastError() and cudaDeviceSynchronize().
9. Copy the result vector B back to the host.
10. Print a few elements to verify correctness.
11. Free device and host memory, reset the device.

The program is kept simple, focusing on clarity and correctness. It includes basic error checking and demonstrates the key CUDA concepts: memory allocation, data transfer, kernel launch, and cleanup. */

#include <stdio.h>
#include <stdlib.h>

// Kernel: adds a scalar to each element of array A and writes to B
__global__ void vectorAddScalar(const float* A, float* B, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        B[idx] = A[idx] + scalar;
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main(void)
{
    const int N = 10;                 // Number of elements
    const float scalar = 5.0f;        // Scalar to add

    // Host memory allocation
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));

    if (!h_A || !h_B) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host input array
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;           // Example values: 0, 1, 2, ...
    }

    // Device memory allocation
    float *d_A = NULL;
    float *d_B = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, N * sizeof(float)));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAddScalar<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, scalar, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Display results
    printf("Result of vector-scalar addition (A[i] + %.1f):\n", scalar);
    for (int i = 0; i < N; ++i) {
        printf("B[%d] = %f\n", i, h_B[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    free(h_A);
    free(h_B);

    // Reset device
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```