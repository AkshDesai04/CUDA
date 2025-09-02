/*
Implement `C[i] = (A[i] == B[i]) ? 1 : 0`.

Thinking: 
To demonstrate a simple element‑wise comparison in CUDA, we implement a kernel that
takes two input integer arrays `A` and `B` and writes to a third array `C` such that
`C[i]` is 1 when `A[i]` equals `B[i]`, otherwise 0.  
Each thread handles one array index.  
The host code allocates sample data, copies it to the device, launches the kernel,
copies the result back, and prints it.  
Error checking is performed via a helper macro.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call)                                            \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",             \
                    __FUNCTION__, __FILE__, __LINE__,                    \
                    cudaGetErrorString(err));                            \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

__global__ void compareKernel(const int *A, const int *B, int *C, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = (A[idx] == B[idx]) ? 1 : 0;
    }
}

int main(void)
{
    const size_t N = 10;
    size_t bytes = N * sizeof(int);

    int hA[N], hB[N], hC[N];

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i)
    {
        hA[i] = i;
        hB[i] = (i % 2 == 0) ? i : i + 1; // some elements will differ
    }

    // Device pointers
    int *dA = NULL, *dB = NULL, *dC = NULL;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&dA, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&dB, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&dC, bytes));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    compareKernel<<<blocks, threadsPerBlock>>>(dA, dB, dC, N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    // Print results
    printf("Index\tA\tB\tC\n");
    for (size_t i = 0; i < N; ++i)
    {
        printf("%zu\t%d\t%d\t%d\n", i, hA[i], hB[i], hC[i]);
    }

    // Clean up
    CHECK_CUDA_ERROR(cudaFree(dA));
    CHECK_CUDA_ERROR(cudaFree(dB));
    CHECK_CUDA_ERROR(cudaFree(dC));

    return 0;
}
