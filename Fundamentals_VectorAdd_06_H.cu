```cuda
/* 
Aim of the program:
Write a kernel for a generalized AXPY: `Y[i] = a*X[i] + b*Y[i] + c`, where a, b, c are scalars.

Thought process:
1. Understand the requirement: We need a CUDA kernel that takes two input arrays X and Y of length n, and three scalar parameters a, b, c. For each element i, it computes Y[i] = a*X[i] + b*Y[i] + c. This is a simple element-wise operation that can be parallelized across GPU threads.

2. Decide on kernel interface:
   - Input: const float* X, float* Y
   - Scalars: float a, b, c
   - Size: int n

3. Write the kernel:
   - Each thread handles one element i = blockIdx.x * blockDim.x + threadIdx.x
   - Check that i < n before accessing memory
   - Perform the computation and write back to Y[i]

4. Host code:
   - Allocate host arrays X_h and Y_h with sample data.
   - Allocate device arrays X_d, Y_d.
   - Copy X_h to X_d, Y_h to Y_d.
   - Launch kernel with a suitable block and grid size.
   - Copy result Y_d back to Y_h.
   - Print a few elements to verify.

5. Error checking:
   - Implement a macro to check CUDA API calls.

6. Make the code self‑contained:
   - Include <cstdio> and <cstdlib>.
   - Use standard memory allocation and CUDA runtime API.
   - Provide a simple main that demonstrates the kernel.

7. Code style:
   - Clear variable names.
   - Comments in code for readability.
   - Minimal but complete program that can be compiled with `nvcc`.

No additional supporting files or output are included; the entire implementation is inside this single .cu file.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors following a CUDA API call */
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                     \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/* Kernel for generalized AXPY operation */
__global__ void generalized_axpy_kernel(int n, const float *X, float *Y,
                                        float a, float b, float c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        Y[idx] = a * X[idx] + b * Y[idx] + c;
    }
}

/* Host function to launch the kernel */
void generalized_axpy(int n, const float *X_h, float *Y_h,
                      float a, float b, float c)
{
    /* Device pointers */
    float *X_d = nullptr;
    float *Y_d = nullptr;

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc((void**)&X_d, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&Y_d, n * sizeof(float)));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(X_d, X_h, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Y_d, Y_h, n * sizeof(float), cudaMemcpyHostToDevice));

    /* Define block and grid sizes */
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    generalized_axpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, X_d, Y_d, a, b, c);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(Y_h, Y_d, n * sizeof(float), cudaMemcpyDeviceToHost));

    /* Free device memory */
    CUDA_CHECK(cudaFree(X_d));
    CUDA_CHECK(cudaFree(Y_d));
}

/* Main function to test the generalized AXPY kernel */
int main()
{
    const int N = 1024;          // Number of elements
    const float a = 2.0f;        // Example scalar a
    const float b = 0.5f;        // Example scalar b
    const float c = 1.0f;        // Example scalar c

    /* Allocate host memory */
    float *X_h = (float*)malloc(N * sizeof(float));
    float *Y_h = (float*)malloc(N * sizeof(float));
    if (!X_h || !Y_h) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input arrays */
    for (int i = 0; i < N; ++i) {
        X_h[i] = static_cast<float>(i);       // X[i] = i
        Y_h[i] = static_cast<float>(N - i);   // Y[i] = N - i
    }

    /* Perform the generalized AXPY on GPU */
    generalized_axpy(N, X_h, Y_h, a, b, c);

    /* Print a few results to verify */
    printf("Result after generalized AXPY (a=%f, b=%f, c=%f):\n", a, b, c);
    for (int i = 0; i < 10; ++i) {
        printf("Y[%3d] = %f\n", i, Y_h[i]);
    }

    /* Clean up host memory */
    free(X_h);
    free(Y_h);

    return EXIT_SUCCESS;
}
```