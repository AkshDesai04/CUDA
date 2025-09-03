/*
Explain why calling this check function only at the end of the program is not as useful for debugging.

The check function in CUDA is typically used to verify that kernel launches, memory copies, and other CUDA API calls succeed. Calling it only at the end of the program means that if an error occurs early (e.g., a kernel launch fails, a memory allocation fails, or an asynchronous memory copy errors out), the program will not be notified immediately. Instead, the program may continue executing, possibly reaching the end of the main function and calling the check. By that point, the error may have already corrupted program state, caused data corruption, or resulted in undefined behavior that makes it difficult to pinpoint the original source of the failure. Early error detection allows the developer to quickly identify and fix the problem, add necessary debugging information, or cleanly abort the program before further side effects occur. Delayed error checking can obscure the true cause, make debugging more time-consuming, and increase the risk of cascading failures that mask the original error.

Thinking process:
1. Identify the key points of why early error detection is important in CUDA programming.
2. Outline how asynchronous execution and CUDA streams can hide errors if not checked immediately.
3. Discuss the potential consequences of waiting until the end: corrupted state, misleading error messages, etc.
4. Summarize these insights succinctly in a comment block as requested.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple CUDA kernel that performs a trivial computation
__global__ void dummyKernel(int *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = idx * 2;
    }
}

// CUDA error checking helper
void checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    int *d_data = NULL;

    // Allocate device memory
    cudaMalloc((void **)&d_data, N * sizeof(int));
    checkCudaError("cudaMalloc");

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dummyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    // Deliberately check error only at the end (bad practice)
    // checkCudaError("dummyKernel launch");

    // Copy result back to host
    int *h_data = (int *)malloc(N * sizeof(int));
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy");

    // Final error check (too late for kernel launch errors)
    checkCudaError("end of program");

    // Verify result (optional)
    int valid = 1;
    for (int i = 0; i < N; ++i) {
        if (h_data[i] != i * 2) {
            valid = 0;
            break;
        }
    }
    printf("Verification %s\n", valid ? "passed" : "failed");

    // Clean up
    free(h_data);
    cudaFree(d_data);

    return 0;
}
