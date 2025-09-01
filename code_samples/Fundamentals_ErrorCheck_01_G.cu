/*
Aim of the program: Change the macro to also print the original CUDA call as a string using the `#call` preprocessor directive.

Thoughts:
- The program demonstrates a macro that wraps CUDA API calls, checks for errors, and prints the original call as a string.
- It uses the preprocessor stringification operator `#` to turn the macro argument into a string.
- A simple vector addition kernel is provided to illustrate the macro usage.
- Each CUDA call is wrapped with `CUDA_CHECK` which prints the error message, the file, line number, and the stringified call if an error occurs.
- The macro is safe to use in all contexts by using the `do { ... } while(0)` pattern.
- The code compiles as a standard CUDA .cu file and includes the necessary headers.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro that checks the return value of a CUDA call.
 * If the call fails, it prints the error string, the file, line,
 * and the original call as a string using the #call stringification operator. */
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s\n",      \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                           \
    } while (0)

/* Simple vector addition kernel */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

int main(void) {
    const int N = 10;
    const size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float h_A[N], h_B[N], h_C[N];
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    /* Copy host memory to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());
    /* Wait for GPU to finish */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Print results */
    printf("Vector addition results:\n");
    for (int i = 0; i < N; ++i) {
        printf("C[%d] = %f + %f = %f\n", i, h_A[i], h_B[i], h_C[i]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
