/*
Initialize vector A with the values `i` and vector B with `100-i`. Verify that `C[i]` is always 100.

Thought process:
- We will create three integer arrays A, B, and C of a fixed size N (e.g., 1024).
- Host code will allocate memory for A and B, initialize them so that A[i] = i and B[i] = 100 - i.
- Allocate memory on the device for A_d, B_d, C_d.
- Copy the initialized host arrays to the device.
- Launch a CUDA kernel that performs element‑wise addition: C_d[i] = A_d[i] + B_d[i].
- Copy C_d back to host and verify each element equals 100.
- Include error checking after each CUDA API call to ensure robustness.
- The program will print "Verification succeeded!" if all elements are 100, otherwise it will report the mismatches.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 1024  // Size of the vectors

// Macro for checking CUDA API calls
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Kernel for vector addition
__global__ void vectorAdd(const int *A, const int *B, int *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    int *h_A = (int *)malloc(N * sizeof(int));
    int *h_B = (int *)malloc(N * sizeof(int));
    int *h_C = (int *)malloc(N * sizeof(int));
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = 100 - i;
    }

    // Device pointers
    int *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, N * sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify results
    int verificationPassed = 1;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != 100) {
            printf("Verification failed at index %d: A=%d, B=%d, C=%d\n",
                   i, h_A[i], h_B[i], h_C[i]);
            verificationPassed = 0;
            // Break after first error to avoid clutter
            break;
        }
    }
    if (verificationPassed) {
        printf("Verification succeeded! All C[i] == 100.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
