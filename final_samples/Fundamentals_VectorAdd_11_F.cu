```cuda
/* 
Aim of the program: Can you still get out-of-bounds errors with a grid-stride loop if `N` is not a multiple of anything? (No, the loop condition `i < N` handles it perfectly). 
The goal of this CUDA C program is to demonstrate that a grid‑stride loop correctly handles array lengths that are not a multiple of the total number of threads. 
We will:
  1. Allocate an array of size N (chosen such that N is not divisible by the total thread count).
  2. Launch a kernel that uses a grid‑stride loop: `for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)`.
  3. Inside the loop, each thread writes to its index in a safe manner.
  4. After kernel execution, copy the array back to host and print a few values to confirm correct execution.
  5. Include CUDA error checking to catch any out‑of‑bounds errors (there should be none).
This program embodies the reasoning that the loop condition `i < N` automatically prevents any thread from accessing indices beyond the array bounds, regardless of whether `N` is a multiple of the total thread count or not.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",         \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),            \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* Kernel that doubles each element using a grid-stride loop */
__global__ void double_elements(float *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        /* Safe access guaranteed by i < N */
        d_arr[i] *= 2.0f;
    }
}

int main(void) {
    /* Choose array size N that is not divisible by total thread count */
    const int N = 10007; /* Example: not divisible by many common thread counts */

    /* Host allocation */
    float *h_arr = (float *)malloc(N * sizeof(float));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize host array */
    for (int i = 0; i < N; ++i) {
        h_arr[i] = (float)i;
    }

    /* Device allocation */
    float *d_arr = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_arr, N * sizeof(float)));

    /* Copy data to device */
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Kernel launch parameters */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; /* not required for safety but typical */

    /* Launch kernel */
    double_elements<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
    CUDA_CHECK(cudaGetLastError()); /* Check for launch errors */

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Verify a few elements */
    printf("Sample of doubled elements (first 10):\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_arr[%d] = %f\n", i, h_arr[i]);
    }

    /* Verify last few elements to ensure no out-of-bounds occurred */
    printf("Sample of doubled elements (last 10):\n");
    for (int i = N - 10; i < N; ++i) {
        printf("h_arr[%d] = %f\n", i, h_arr[i]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);

    return EXIT_SUCCESS;
}
```