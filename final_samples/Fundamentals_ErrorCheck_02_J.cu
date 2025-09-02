/*
Add the `CHECK` macro around a call to `cudaGetLastError()`.
The goal is to demonstrate how to wrap CUDA API error checks with a convenient macro that will abort the program with a clear message if any CUDA call fails. In this simple example we perform a vector addition kernel and use the `CHECK` macro after the kernel launch to catch launch failures or any other errors reported by `cudaGetLastError()`. The `CHECK` macro is implemented using a helper function `gpuAssert` that prints the error string and terminates execution if an error occurs. The program is fully self‑contained and can be compiled with `nvcc`. It will allocate two input vectors on the host, copy them to the device, launch the addition kernel, copy the result back, and finally verify the output. The macro is placed immediately after the kernel invocation and before any further synchronization to ensure that launch‑time errors are caught early.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper macro to check CUDA errors */
#define CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/* Error checking function */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* Simple vector addition kernel */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
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

    /* Initialize input vectors */
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    /* Allocate device memory */
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CHECK(cudaMalloc((void**)&d_A, size));
    CHECK(cudaMalloc((void**)&d_B, size));
    CHECK(cudaMalloc((void**)&d_C, size));

    /* Copy inputs to device */
    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Add CHECK macro around cudaGetLastError() */
    CHECK(cudaGetLastError());

    /* Synchronize to catch runtime errors */
    CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Verify results */
    bool success = true;
    for (int i = 0; i < N; ++i)
    {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5)
        {
            printf("Mismatch at index %d: host=%f, device=%f\n", i, h_A[i]+h_B[i], h_C[i]);
            success = false;
            break;
        }
    }

    if (success)
        printf("Vector addition successful!\n");
    else
        printf("Vector addition failed.\n");

    /* Free resources */
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    return 0;
}
