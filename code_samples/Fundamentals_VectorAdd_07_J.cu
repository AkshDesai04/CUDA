```cuda
/* 
Aim of the program: Time the squaring kernel.

Thinking and design:
- The program will create a CUDA kernel that squares each element of an input array.
- It will allocate host and device memory, initialize the input array on the host,
  copy it to the device, launch the kernel, copy the result back, and finally free memory.
- Timing will be performed only for the kernel execution using CUDA events (cudaEvent_t).
  Two events, start and stop, will be recorded immediately before and after the kernel
  launch. After the kernel completes, the elapsed time in milliseconds will be computed
  and printed.
- Basic error checking is incorporated via a macro (CUDA_CHECK) that verifies the
  return value of CUDA API calls and the kernel launch.
- The kernel uses a 1D grid of blocks, with each thread handling a single array element.
  Boundary checks ensure that threads beyond the array size do nothing.
- A block size of 256 threads is chosen as a common default that works well on most GPUs.
- The example uses floating-point arrays, but the same pattern would work for integers
  or other data types by adjusting the kernel signature.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

/* Kernel that squares each element of the input array */
__global__ void squareKernel(float *d_out, const float *d_in, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = d_in[idx];
        d_out[idx] = val * val;
    }
}

int main(void)
{
    const int N = 1 << 24; // 16 million elements
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_in  = (float*)malloc(size);
    float *h_out = (float*)malloc(size);
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize input array */
    for (int i = 0; i < N; ++i) {
        h_in[i] = (float)i;
    }

    /* Allocate device memory */
    float *d_in  = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in,  size));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size));

    /* Copy input data to device */
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    /* Timing events */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaEventRecord(start, 0));
    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));

    /* Wait for kernel to finish */
    CUDA_CHECK(cudaEventSynchronize(stop));

    /* Compute elapsed time */
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    /* Verify a few results (optional) */
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = h_in[i] * h_in[i];
        if (h_out[i] != expected) {
            fprintf(stderr, "Mismatch at index %d: host=%f, device=%f\n",
                    i, expected, h_out[i]);
            errors++;
        }
    }
    if (errors == 0) {
        printf("First 10 outputs verified correctly.\n");
    }

    printf("Kernel execution time: %f ms\n", milliseconds);

    /* Cleanup */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return EXIT_SUCCESS;
}
```