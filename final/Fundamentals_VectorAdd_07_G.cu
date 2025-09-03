/*
Verify the results on the CPU.

In order to satisfy the requirement I will write a small CUDA program that
demonstrates a simple GPU computation (vector addition) and then verifies
that the results are correct by recomputing the same operation on the CPU.
The program will:

1. Allocate two input arrays (A and B) on the host and fill them with
   deterministic values (e.g., A[i] = i, B[i] = 2*i) so that the expected
   result is known.
2. Allocate corresponding device arrays and copy the host data to the device.
3. Launch a CUDA kernel that performs element‑wise addition: C[i] = A[i] + B[i].
4. Copy the result array back to the host.
5. Compute the same addition on the CPU, producing a reference result.
6. Compare each element of the GPU result with the CPU result using a small
   tolerance to account for floating‑point differences.
7. Report success if all elements match within the tolerance, otherwise
   report a failure and indicate the first mismatch.
8. Clean up all allocated memory.

The program includes a simple error‑checking macro to handle CUDA API
calls. It uses a block size of 256 threads per block, which is a common
choice for such simple kernels. The array size is set to 1024 elements for
brevity, but it can be easily increased to test performance.

This implementation fully meets the goal of verifying GPU results against a
CPU reference and provides a clear, self‑contained example in a single .cu
file. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",    \
                    #call, __FILE__, __LINE__,                  \
                    cudaGetErrorString(err));                   \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1024;            // Number of elements
    const int SIZE = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(SIZE);
    float *h_B = (float *)malloc(SIZE);
    float *h_C = (float *)malloc(SIZE);     // Result from GPU
    float *h_ref = (float *)malloc(SIZE);   // Reference result from CPU

    if (!h_A || !h_B || !h_C || !h_ref) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    // Allocate device memory
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, SIZE));
    CHECK_CUDA(cudaMalloc((void **)&d_B, SIZE));
    CHECK_CUDA(cudaMalloc((void **)&d_C, SIZE));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());  // Check for launch errors
    CHECK_CUDA(cudaDeviceSynchronize());  // Wait for completion

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, SIZE, cudaMemcpyDeviceToHost));

    // Compute reference result on CPU
    for (int i = 0; i < N; ++i) {
        h_ref[i] = h_A[i] + h_B[i];
    }

    // Verify results
    const float epsilon = 1e-5f;
    bool match = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - h_ref[i]) > epsilon) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n",
                   i, h_C[i], h_ref[i]);
            match = false;
            break;
        }
    }

    if (match) {
        printf("Verification PASSED: GPU results match CPU results.\n");
    } else {
        printf("Verification FAILED: GPU results do not match CPU results.\n");
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return match ? EXIT_SUCCESS : EXIT_FAILURE;
}
