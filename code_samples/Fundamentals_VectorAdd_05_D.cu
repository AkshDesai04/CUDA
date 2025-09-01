```cuda
/* 
Aim of the program: Implement `C[i] = A[i] * B[i] * D[i]` (three-vector multiplication). 

Thinking process:
1. The goal is to perform an element-wise multiplication of three vectors A, B, and D, storing the result in C. 
2. For a CUDA implementation, the work should be distributed across many GPU threads so that each thread handles one element of the vectors.
3. We need to:
   - Allocate host arrays for A, B, C, D.
   - Initialize A, B, D with some values (for demonstration we use random floats).
   - Allocate corresponding device arrays.
   - Copy A, B, D from host to device.
   - Launch a kernel where each thread computes C[i] = A[i] * B[i] * D[i] for its index.
   - Copy the result C back to the host.
   - Optionally verify a few elements or compute a checksum to ensure correctness.
4. CUDA error checking is important, so we wrap CUDA API calls in a macro that prints the error and aborts if an error occurs.
5. For simplicity, we use a fixed vector size (e.g., 1,048,576 elements) and a simple grid/block configuration (256 threads per block).
6. The kernel will use a global index calculation: `int idx = blockIdx.x * blockDim.x + threadIdx.x;` and guard against out-of-bounds indices.
7. After computation, we free all allocated GPU and CPU memory.
8. The code compiles with nvcc and runs on any CUDA-capable device.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// Error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",      \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),          \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel: element-wise multiplication of three vectors
__global__ void vecMul3Kernel(const float *A, const float *B, const float *D, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx] * D[idx];
    }
}

int main()
{
    const int N = 1 << 20;            // 1,048,576 elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    float *h_D = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C || !h_D) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors with random values
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_D[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_D, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D, h_D, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vecMul3Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, d_C, N);
    CUDA_CHECK(cudaGetLastError());   // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: compute checksum
    double checksum = 0.0;
    for (int i = 0; i < N; ++i) {
        checksum += h_C[i];
    }
    printf("Checksum of result vector C: %f\n", checksum);

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    return EXIT_SUCCESS;
}
```