```cuda
/*
Time the matrix addition kernel.

The goal is to measure the execution time of a simple element-wise matrix addition
kernel on the GPU.  To achieve this we follow the typical CUDA workflow:

1. **Host Setup**  
   - Determine matrix dimensions (default 1024x1024, optionally overridden by command line).
   - Allocate pinned host memory for the input matrices A and B, and the result matrix C.
   - Initialize A and B with arbitrary values (e.g., random floats).

2. **Device Setup**  
   - Allocate device memory for A, B, and C.
   - Copy A and B from host to device using cudaMemcpy.

3. **Kernel Launch**  
   - Define a simple kernel that adds corresponding elements of A and B into C.
   - Choose a block size (16x16 threads) and compute a grid that covers the matrix.
   - Use `cudaEvent_t` objects to time only the kernel execution:
     - Record a start event.
     - Launch the kernel.
     - Record an end event.
     - Synchronize and compute elapsed time with `cudaEventElapsedTime`.

4. **Result Retrieval**  
   - Copy the computed matrix C back to host.
   - Optionally verify correctness by comparing against a CPU implementation.

5. **Cleanup**  
   - Free all host and device memory and destroy CUDA events.

Error handling is performed with a helper macro that checks the return codes of CUDA API calls.
The timing measurement focuses exclusively on the kernel, excluding memory transfers,
which provides a clear picture of the kernel’s performance.

The program is fully self‑contained, compilable with `nvcc`, and prints the elapsed kernel
time in milliseconds to standard output.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

/* Helper macro for CUDA error checking */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* CUDA kernel: element-wise matrix addition */
__global__ void matrixAdd(const float *A, const float *B, float *C, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // row index

    if (row < height && col < width)
    {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char *argv[])
{
    /* Matrix dimensions */
    int width = 1024;
    int height = 1024;

    /* Optional command line arguments to override dimensions */
    if (argc >= 3)
    {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }

    size_t size = width * height * sizeof(float);

    /* Allocate pinned host memory for better transfer performance */
    float *h_A, *h_B, *h_C;
    gpuErrchk(cudaMallocHost((void**)&h_A, size));
    gpuErrchk(cudaMallocHost((void**)&h_B, size));
    gpuErrchk(cudaMallocHost((void**)&h_C, size));

    /* Initialize host matrices with random data */
    srand((unsigned)time(NULL));
    for (int i = 0; i < width * height; ++i)
    {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    /* Device memory allocation */
    float *d_A, *d_B, *d_C;
    gpuErrchk(cudaMalloc((void**)&d_A, size));
    gpuErrchk(cudaMalloc((void**)&d_B, size));
    gpuErrchk(cudaMalloc((void**)&d_C, size));

    /* Copy data from host to device */
    gpuErrchk(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Define block and grid dimensions */
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    /* Record the start event */
    gpuErrchk(cudaEventRecord(start, 0));

    /* Launch the kernel */
    matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width, height);

    /* Record the stop event */
    gpuErrchk(cudaEventRecord(stop, 0));
    gpuErrchk(cudaEventSynchronize(stop));

    /* Calculate elapsed time in milliseconds */
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    /* Copy result back to host */
    gpuErrchk(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Optional: Verify correctness */
    int errors = 0;
    for (int i = 0; i < width * height; ++i)
    {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f)
        {
            ++errors;
            if (errors <= 10) // print first few errors
                printf("Mismatch at %d: %f != %f\n", i, h_C[i], expected);
        }
    }
    if (errors == 0)
        printf("Result verified: all entries match.\n");
    else
        printf("Total mismatches: %d\n", errors);

    /* Print timing result */
    printf("Matrix addition kernel time: %f ms\n", milliseconds);

    /* Clean up */
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_C));
    gpuErrchk(cudaFreeHost(h_A));
    gpuErrchk(cudaFreeHost(h_B));
    gpuErrchk(cudaFreeHost(h_C));

    return 0;
}
```