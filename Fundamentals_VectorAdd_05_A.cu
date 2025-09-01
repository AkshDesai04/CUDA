/*
Aim: Implement `C[i] = A[i] * A[i]` (squaring).

The user request is to write a CUDA C program that takes an array A, computes the square of each element, and stores the result in array C. 
The program must be a single .cu file. According to the developer instruction, the first thing in the file must be a multi‑line comment that contains:
1. The exact aim statement from the prompt, word for word: `Implement \`C[i] = A[i] * A[i]\` (squaring).`
2. My entire reasoning and design choices.

Below is my thinking:

- Use a simple kernel that launches one thread per element.  
- Use float arrays for simplicity, but could be any numeric type.  
- Allocate host memory with malloc, initialize with some data.  
- Allocate device memory with cudaMalloc.  
- Copy host data to device with cudaMemcpy.  
- Launch kernel with a reasonable block size (e.g., 256 threads).  
- Copy results back to host.  
- Verify correctness for a small number of elements.  
- Include basic error checking after each CUDA call for robustness.  
- Keep the program self‑contained and straightforward to compile and run.

The final code below follows this plan and is ready to compile with `nvcc`.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel that squares each element
__global__ void square_kernel(const float *A, float *C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] * A[idx];
}

int main(void)
{
    const size_t N = 1 << 20;          // Number of elements (1M)
    const size_t bytes = N * sizeof(float);

    // Host allocations
    float *h_A = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (h_A == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array A with some values
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)i / N;   // example values between 0 and 1
    }

    // Device allocations
    float *d_A = NULL;
    float *d_C = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (int)((N + threadsPerBlock - 1) / threadsPerBlock);

    // Launch kernel
    square_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Optional: verify a few results
    int errors = 0;
    for (size_t i = 0; i < 10; ++i) {
        float expected = h_A[i] * h_A[i];
        if (fabsf(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %zu: got %f, expected %f\n",
                    i, h_C[i], expected);
            errors++;
        }
    }
    if (errors == 0) {
        printf("Verification passed for first 10 elements.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Clean up
    free(h_A);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
