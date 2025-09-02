```cpp
/*
 * Aim of the program:
 * Re-implement vector addition using Unified Memory. Allocate all vectors using
 * cudaMallocManaged. Notice that you no longer need explicit cudaMemcpy calls.
 * The CUDA runtime will migrate data on-demand.
 *
 * Thinking:
 * The task requires a straightforward vector addition kernel, but instead of
 * using separate device memory allocations and explicit host-to-device / device-to-host
 * copies, we use Unified Memory (cudaMallocManaged) to allocate the vectors A, B, and C.
 * This allows us to initialize A and B directly on the host, launch the kernel, and
 * read back C without any explicit memcpy calls. The CUDA runtime handles data
 * migration between CPU and GPU as needed.
 *
 * Implementation steps:
 * 1. Include <cstdio> for printing.
 * 2. Define a simple error-checking macro to capture CUDA API failures.
 * 3. Implement the vector addition kernel (vectorAdd) which writes C[i] = A[i] + B[i].
 * 4. In main:
 *    - Set a vector size N.
 *    - Allocate A, B, C using cudaMallocManaged.
 *    - Initialize A and B on the host with some values.
 *    - Choose grid/block dimensions and launch the kernel.
 *    - Call cudaDeviceSynchronize to ensure the kernel completes.
 *    - Optionally print a few elements of C to verify correctness.
 *    - Free the unified memory allocations.
 *
 * No cudaMemcpy calls are required because Unified Memory automatically migrates
 * pages on demand between host and device during kernel execution or data access.
 */

#include <cstdio>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            return EXIT_FAILURE;                                            \
        }                                                                    \
    } while (0)

__global__
void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    float *A, *B, *C;

    // Allocate unified memory
    CUDA_CHECK(cudaMallocManaged(&A, size));
    CUDA_CHECK(cudaMallocManaged(&B, size));
    CUDA_CHECK(cudaMallocManaged(&C, size));

    // Initialize vectors A and B
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify a few results
    printf("C[0] = %f (expected %f)\n", C[0], A[0] + B[0]);
    printf("C[N/2] = %f (expected %f)\n", C[N/2], A[N/2] + B[N/2]);
    printf("C[N-1] = %f (expected %f)\n", C[N-1], A[N-1] + B[N-1]);

    // Clean up
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));

    return EXIT_SUCCESS;
}
```