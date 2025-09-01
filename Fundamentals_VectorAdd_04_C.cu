/*
Implement `C[i] = A[i] - scalar`. 

I started by parsing the user request. The task is to create a CUDA program that performs an element‑wise subtraction of a scalar from each element of an input array `A` and stores the result in array `C`. The simplest approach is to write a CUDA kernel that takes the input array, the output array, the scalar, and the number of elements `n`. Each thread handles one index: it reads `A[idx]`, subtracts `scalar`, and writes to `C[idx]`. 

The host code will:
1. Allocate and initialize a small example array on the host.
2. Allocate corresponding device memory using `cudaMalloc`.
3. Copy the host input array to the device.
4. Launch the kernel with a grid configuration that covers all elements.
5. Copy the result back to the host.
6. Print the output for verification.
7. Clean up device memory.

I will include a small helper macro for error checking after each CUDA API call. The example will use `float` for simplicity and a fixed array size of 10. This code is self‑contained and can be compiled with `nvcc`. 

No additional text or files will be produced; only the complete .cu source code follows.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel that subtracts a scalar from each element of array A
__global__ void subtractScalar(const float *A, float *C, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] - scalar;
    }
}

int main(void) {
    const int N = 10;
    const int SIZE = N * sizeof(float);
    const float scalar = 1.5f;

    // Host arrays
    float h_A[N];
    float h_C[N];

    // Initialize host input array
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)(i + 1); // 1.0, 2.0, ..., 10.0
    }

    // Device arrays
    float *d_A = NULL;
    float *d_C = NULL;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_A, SIZE));
    CHECK_CUDA(cudaMalloc((void **)&d_C, SIZE));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int THREADS_PER_BLOCK = 256;
    const int BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    subtractScalar<<<BLOCKS, THREADS_PER_BLOCK>>>(d_A, d_C, scalar, N);
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, SIZE, cudaMemcpyDeviceToHost));

    // Print results
    printf("Input array A:\n");
    for (int i = 0; i < N; ++i) {
        printf("%f ", h_A[i]);
    }
    printf("\n\nScalar: %f\n\n", scalar);

    printf("Result array C (A[i] - scalar):\n");
    for (int i = 0; i < N; ++i) {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
