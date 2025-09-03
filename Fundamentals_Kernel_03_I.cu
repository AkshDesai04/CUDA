/*
Launch with one block of N threads. Pass an integer array `int* out` to the kernel. Have each thread write its `threadIdx.x` to `out[threadIdx.x]`.

My thinking: The goal is to demonstrate a simple kernel that writes the thread index into an output array. We'll launch with a single block containing N threads. The kernel will take a pointer to an integer array allocated on device memory. Each thread will store its threadIdx.x into the corresponding index of that array. After launch, the host will copy the array back and print the values to verify correctness. We'll include basic error checking and resource cleanup.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple error checking macro */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* Kernel: each thread writes its threadIdx.x into out[threadIdx.x] */
__global__ void writeThreadIdx(int *out) {
    out[threadIdx.x] = threadIdx.x;
}

int main(void) {
    const int N = 256; /* Number of threads in the single block */

    /* Allocate host array */
    int *h_out = (int *)malloc(N * sizeof(int));
    if (h_out == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Allocate device array */
    int *d_out = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_out, N * sizeof(int)));

    /* Launch kernel: one block of N threads */
    dim3 blockDim(N);
    dim3 gridDim(1);
    writeThreadIdx<<<gridDim, blockDim>>>(d_out);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Wait for the GPU to finish */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Print the results */
    printf("Thread indices written to array:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    /* Clean up */
    CUDA_CHECK(cudaFree(d_out));
    free(h_out);

    return 0;
}
