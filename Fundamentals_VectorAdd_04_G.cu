/*
Aim: Implement for character arrays: `C[i] = A[i] - B[i]`.

Thinking process:
1. We want a CUDA program that subtracts two character arrays element-wise and stores the result in a third array.
2. Character arrays are stored as `char` type; subtraction will be performed in signed integer domain then cast back to `char`.
3. Kernel: `__global__ void subtractKernel(char *A, char *B, char *C, int n)`  
   Inside:  
   ```cpp
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < n) C[idx] = A[idx] - B[idx];
   ```
4. Host side:  
   * Allocate host arrays, initialize with sample data.  
   * Allocate device memory with `cudaMalloc`.  
   * Copy data to device (`cudaMemcpy`).  
   * Launch kernel with an appropriate grid/block configuration.  
   * Copy result back to host.  
   * Print a few elements to verify correctness.  
5. Include a simple CUDA error‑checking macro `cudaCheckError`.  
6. Use a small size (e.g., `N = 10`) for demonstration.  
7. Free all allocated memory before exiting.  
8. Provide a complete `.cu` file with all necessary includes, the kernel, error checking, and a `main` function that ties everything together.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* CUDA error checking macro */
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* Kernel to subtract two character arrays */
__global__ void subtractKernel(const char *A, const char *B, char *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        /* Perform subtraction in signed integer to avoid overflow issues */
        int temp = static_cast<int>(A[idx]) - static_cast<int>(B[idx]);
        C[idx] = static_cast<char>(temp);
    }
}

int main(void)
{
    const int N = 10;
    size_t bytes = N * sizeof(char);

    /* Host arrays */
    char *h_A = (char*)malloc(bytes);
    char *h_B = (char*)malloc(bytes);
    char *h_C = (char*)malloc(bytes);

    /* Initialize host arrays with sample values */
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<char>(i + 10);     /* 10, 11, ..., 19 */
        h_B[i] = static_cast<char>(i);          /* 0, 1, ..., 9 */
    }

    /* Device arrays */
    char *d_A, *d_B, *d_C;
    cudaCheckError( cudaMalloc((void**)&d_A, bytes) );
    cudaCheckError( cudaMalloc((void**)&d_B, bytes) );
    cudaCheckError( cudaMalloc((void**)&d_C, bytes) );

    /* Copy data from host to device */
    cudaCheckError( cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice) );

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    subtractKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaCheckError( cudaGetLastError() );
    cudaCheckError( cudaDeviceSynchronize() );

    /* Copy result back to host */
    cudaCheckError( cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost) );

    /* Print results */
    printf("Index\tA\tB\tC=A-B\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d\t%d\t%d\t%d\n", i, (int)h_A[i], (int)h_B[i], (int)h_C[i]);
    }

    /* Clean up */
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
