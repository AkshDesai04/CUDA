/*
Implement `C[i] = (A[i] > threshold) ? 1 : 0` (thresholding).

**Thought process and design decisions:**

1. **Goal**  
   The program must demonstrate a simple element‑wise thresholding operation on an array using CUDA.  
   For each element `A[i]`, the output `C[i]` should be `1` if `A[i] > threshold`, otherwise `0`.

2. **Language and target**  
   The code is written in C++ with CUDA extensions (.cu file).  
   We will use the standard CUDA runtime API and compile with `nvcc`.

3. **Kernel design**  
   A very small kernel is sufficient: each thread handles one element.  
   The kernel will read the element from `A` on device memory, compare it to the threshold (passed as a scalar), and write the result to `C`.  
   The result type is `int` to store 0 or 1.

4. **Memory allocation**  
   - Host arrays: `float *h_A`, `int *h_C`.  
   - Device arrays: `float *d_A`, `int *d_C`.  
   - We choose a moderate size `N = 1024` for demonstration.

5. **Error handling**  
   Wrap CUDA API calls in a helper macro `CUDA_CHECK` that prints an error and exits if the call fails.  
   This keeps the main code clean.

6. **Execution configuration**  
   Use 256 threads per block, compute the number of blocks as `(N + threadsPerBlock - 1) / threadsPerBlock`.  
   This covers all elements without excessive blocks.

7. **Testing & output**  
   After copying `C` back to the host, print the first 10 results to verify correctness.  
   We also compute a simple verification on the CPU to compare.

8. **Build instructions**  
   The program can be compiled with:  
   `nvcc -o thresholding thresholding.cu`  
   and run: `./thresholding`

9. **Extensions**  
   This skeleton can be extended to handle larger arrays, different data types, or to read input from files.

The program below implements all of the above. It is fully self‑contained and ready to compile with `nvcc`.  
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 256

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

/* CUDA kernel performing element-wise thresholding */
__global__ void thresholdKernel(const float *A, int *C, float threshold, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        C[idx] = (A[idx] > threshold) ? 1 : 0;
    }
}

int main(void)
{
    /* Host allocations */
    float *h_A = (float*)malloc(N * sizeof(float));
    int   *h_C = (int*)  malloc(N * sizeof(int));
    if (!h_A || !h_C)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input array with random values between 0 and 1 */
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)rand() / RAND_MAX;
    }

    /* Threshold value */
    const float threshold = 0.5f;

    /* Device allocations */
    float *d_A = NULL;
    int   *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(int)));

    /* Copy input to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Launch kernel */
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    thresholdKernel<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_C, threshold, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Simple CPU verification */
    int mismatches = 0;
    for (int i = 0; i < N; ++i)
    {
        int expected = (h_A[i] > threshold) ? 1 : 0;
        if (h_C[i] != expected)
        {
            ++mismatches;
        }
    }

    /* Print results */
    printf("Threshold = %f\n", threshold);
    printf("First 10 input values and thresholded output:\n");
    for (int i = 0; i < 10 && i < N; ++i)
    {
        printf("A[%d] = %f -> C[%d] = %d\n", i, h_A[i], i, h_C[i]);
    }
    printf("Total mismatches between GPU and CPU: %d\n", mismatches);

    /* Cleanup */
    free(h_A);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));

    return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
