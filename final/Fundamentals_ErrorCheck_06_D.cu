/*
Aim: Demonstrate its use by checking a successful kernel launch.

Thought process:
1. The program must be a single .cu file that compiles with nvcc and demonstrates a successful CUDA kernel launch.
2. To keep the example minimal yet illustrative, the kernel will perform a simple element‑wise operation on an array (increment each element by 1).
3. After launching the kernel, we need to check for launch failures using cudaGetLastError() and also ensure that the kernel finished executing successfully with cudaDeviceSynchronize().
4. If both checks succeed, we will print a confirmation message. If any error occurs, the program will report the error string.
5. Finally, the program will copy the result back to host memory and print the first few elements to confirm the operation.
6. All CUDA API calls will be wrapped with a simple error‑checking macro for clarity.
7. The code will be self‑contained, use only standard headers, and compile with nvcc without external dependencies.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

/* Simple kernel that increments each element of an array */
__global__ void increment_kernel(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1;
    }
}

int main(void) {
    const int N = 256;               // Size of array
    const int THREADS_PER_BLOCK = 64;
    const int NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    /* Allocate host memory */
    int *h_arr = (int *)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }
    /* Initialize host array */
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    /* Allocate device memory */
    int *d_arr = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_arr, N * sizeof(int)));

    /* Copy data from host to device */
    CHECK_CUDA(cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice));

    /* Launch kernel */
    increment_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_arr, N);

    /* Check for kernel launch errors */
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(launch_err));
        CHECK_CUDA(cudaFree(d_arr));
        free(h_arr);
        return EXIT_FAILURE;
    }

    /* Wait for kernel to finish and check for execution errors */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Display a few results */
    printf("Kernel executed successfully. Sample results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_arr[%d] = %d\n", i, h_arr[i]);
    }

    /* Clean up */
    CHECK_CUDA(cudaFree(d_arr));
    free(h_arr);

    return EXIT_SUCCESS;
}
