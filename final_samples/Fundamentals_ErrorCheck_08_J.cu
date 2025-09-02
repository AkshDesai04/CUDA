/*
Write a conclusion in comments: always use `cuda-memcheck` during development, and have robust error checking in your code.

My thinking:
- The user requested a CUDA program that includes a conclusion in comments advising the use of cuda-memcheck and robust error checking.
- According to the developer instructions, I must start the file with a multiline comment that contains the exact aim of the program as given in the prompt, verbatim. I also need to include my entire reasoning inside that comment.
- The aim string is: "Write a conclusion in comments: always use `cuda-memcheck` during development, and have robust error checking in your code."
- I will embed that string in the comment block, then add my thoughts, including the advice about cuda-memcheck, the importance of error checking, and a note that the following code demonstrates basic error handling.
- The code itself will be a minimal but functional CUDA program: a simple vector addition kernel with host code that allocates memory, copies data, launches the kernel, and frees resources. It will use a macro for error checking after each CUDA call.
- The conclusion advice will also appear in a separate comment after the code to make it clear that it is part of the documentation.

This satisfies all constraints: only a .cu file, with the required comment at the start, followed by full code, and no extraneous output. 
*/

// CUDA vector addition example with robust error checking
#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Simple kernel that adds two vectors
__global__ void vecAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copy host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError()); // Check kernel launch

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Simple verification
    bool success = true;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: got %f, expected %f\n",
                    i, h_C[i], expected);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Vector addition successful.\n");
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

/*
Conclusion: always use `cuda-memcheck` during development, and have robust error checking in your code.
*/

