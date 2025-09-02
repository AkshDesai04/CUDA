/*
Instrument your code with `printf` before and after sync points (from thread 0 only) to trace the execution flow.

The goal of this program is to demonstrate how to place device-side printf statements before and after __syncthreads() calls so that only thread 0 of each block reports its progress.  This is useful for debugging or understanding the flow of execution in parallel kernels.  The kernel performs a simple element‑wise addition of two arrays, but between the two stages of the addition it synchronizes the threads within a block.  By printing from thread 0 before and after each __syncthreads(), we can observe the exact order in which blocks enter and exit the synchronization points.

The host code allocates three small integer arrays on the device, copies input data from the host, launches the kernel with a modest number of blocks and threads, and then copies the result back.  Simple error checking macros are used to catch CUDA API failures.  After the kernel completes, the host verifies that the output matches the expected result.  Device printf output will appear in the console as the kernel executes, with each block’s thread 0 printing before and after each sync point.

Because device printf requires a GPU architecture of sm_20 or higher, the code should be compiled with a suitable compute capability (e.g., -arch=sm_50).  This example is intentionally simple to keep the focus on the instrumentation rather than performance or memory optimization.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Error checking macro */
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

/* Kernel that adds two arrays element-wise and prints before/after sync points */
__global__ void vectorAddWithSync(const int *a, const int *b, int *c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* First synchronization point */
    if (threadIdx.x == 0)
        printf("[Block %d] Thread 0 before first __syncthreads() at tid %d\n", blockIdx.x, tid);
    __syncthreads();
    if (threadIdx.x == 0)
        printf("[Block %d] Thread 0 after first __syncthreads() at tid %d\n", blockIdx.x, tid);

    /* Dummy work: each thread calculates its part of the addition */
    if (tid < n)
    {
        /* Simulate some work with a small loop */
        int sum = 0;
        for (int i = 0; i < 10; ++i)
        {
            sum += a[tid] + b[tid];
        }
        c[tid] = sum / 10;  // average
    }

    /* Second synchronization point */
    if (threadIdx.x == 0)
        printf("[Block %d] Thread 0 before second __syncthreads() at tid %d\n", blockIdx.x, tid);
    __syncthreads();
    if (threadIdx.x == 0)
        printf("[Block %d] Thread 0 after second __syncthreads() at tid %d\n", blockIdx.x, tid);

    /* Final dummy work */
    if (tid < n)
    {
        c[tid] += 1;  // simple post-sum increment
    }
}

int main()
{
    const int N = 16;           // Number of elements
    const int THREADS_PER_BLOCK = 4;
    const int NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    /* Host allocations */
    int *h_a = (int *)malloc(N * sizeof(int));
    int *h_b = (int *)malloc(N * sizeof(int));
    int *h_c = (int *)malloc(N * sizeof(int));

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = i;
        h_b[i] = 2 * i;
        h_c[i] = 0;
    }

    /* Device allocations */
    int *d_a, *d_b, *d_c;
    checkCudaErrors(cudaMalloc((void **)&d_a, N * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_b, N * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_c, N * sizeof(int)));

    /* Copy data to device */
    checkCudaErrors(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice));

    /* Launch kernel */
    vectorAddWithSync<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    /* Check for kernel launch errors */
    checkCudaErrors(cudaGetLastError());

    /* Synchronize to ensure all prints are flushed */
    checkCudaErrors(cudaDeviceSynchronize());

    /* Copy result back to host */
    checkCudaErrors(cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Verify results */
    bool success = true;
    for (int i = 0; i < N; ++i)
    {
        int expected = h_a[i] + h_b[i] + 1;
        if (h_c[i] != expected)
        {
            printf("Mismatch at index %d: expected %d, got %d\n", i, expected, h_c[i]);
            success = false;
        }
    }

    if (success)
        printf("Success! All results are correct.\n");
    else
        printf("Failure! Results did not match expected values.\n");

    /* Clean up */
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    /* Reset device */
    checkCudaErrors(cudaDeviceReset());

    return 0;
}
