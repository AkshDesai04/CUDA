```cuda
/* 
Aim: Instead of `exit(-1)`, use `cudaDeviceReset()` before exiting to clean up the context.

Thinking:
1. The requirement is to demonstrate how to properly clean up a CUDA context before the program exits.
2. We'll create a minimal CUDA program that launches a simple kernel.
3. For error handling, a macro (`CUDA_CHECK`) will wrap all CUDA API calls.
4. Whenever an error occurs, the macro prints the error, calls `cudaDeviceReset()` to release all resources, and then exits the program with `EXIT_FAILURE`.
5. After the kernel completes successfully, we still call `cudaDeviceReset()` to ensure the context is destroyed gracefully before the program terminates normally.
6. The code is fully self‑contained and can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

/* Simple kernel that writes 1 to each element of an array */
__global__ void setOnes(int *d_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_array[idx] = 1;
}

/* Error checking macro */
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            cudaDeviceReset(); /* Clean up context before exit */  \
            exit(EXIT_FAILURE);                                   \
        }                                                          \
    } while (0)

int main(void) {
    const int N = 1 << 20;  /* 1M elements */
    int *d_array = NULL;
    int *h_array = NULL;

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc((void **)&d_array, N * sizeof(int)));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    setOnes<<<blocksPerGrid, threadsPerBlock>>>(d_array, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Allocate host memory and copy result back */
    h_array = (int *)malloc(N * sizeof(int));
    if (h_array == NULL) {
        fprintf(stderr, "Host allocation failed\n");
        cudaDeviceReset(); /* Clean up context before exit */
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Verify a few elements */
    for (int i = 0; i < 10; ++i) {
        if (h_array[i] != 1) {
            fprintf(stderr, "Verification failed at index %d\n", i);
            cudaDeviceReset(); /* Clean up context before exit */
            free(h_array);
            return EXIT_FAILURE;
        }
    }
    printf("Kernel executed successfully, first 10 elements verified.\n");

    /* Free device and host memory */
    CUDA_CHECK(cudaFree(d_array));
    free(h_array);

    /* Clean up CUDA context before normal exit */
    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
```