/*
The CUDA compiler is often smart enough to detect simple cases of this error. Does your example compile without warnings?
I was asked to demonstrate a CUDA program that compiles cleanly, without any compiler warnings.  
To satisfy that, I write a minimal but functional CUDA program that:
- Allocates two input arrays and one output array on the device.
- Copies input data from host to device.
- Launches a simple element‑wise addition kernel.
- Copies the result back to the host.
- Prints the first few results to verify correctness.
All variables are fully initialized, every function is correctly declared, and the
code follows the CUDA C conventions so that a standard `nvcc` build produces no
warnings.  
The kernel is intentionally simple to avoid any corner cases that might
trigger compiler diagnostics.  All error checking is performed with
`cudaGetLastError()` and the return value from `cudaDeviceSynchronize()` to
ensure that any runtime error is reported.  The program uses only the
standard CUDA runtime API and standard C library functions, making it
portable and straightforward to compile on any CUDA‑enabled system.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel that adds two arrays element‑wise */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

/* Helper function to check CUDA errors */
static void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    const int N = 1 << 20;          // 1M elements
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    /* Initialize host input arrays */
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    checkCudaError(cudaMalloc((void**)&d_A, size), "cudaMalloc d_A");
    checkCudaError(cudaMalloc((void**)&d_B, size), "cudaMalloc d_B");
    checkCudaError(cudaMalloc((void**)&d_C, size), "cudaMalloc d_C");

    /* Copy host data to device */
    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "cudaMemcpy B");

    /* Determine block and grid sizes */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    /* Copy result back to host */
    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "cudaMemcpy C");

    /* Verify a few results */
    printf("Verification of first 10 results:\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    /* Clean up */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
