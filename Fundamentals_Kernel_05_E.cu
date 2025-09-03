```c
/*
Aim: Write the global ID to an output array at the corresponding index: `output[id] = id;`. Verify the result on the host.

Thinking:
1. Define a CUDA kernel that calculates the global thread index (using blockIdx, blockDim, threadIdx) and writes that index into the output array.
2. In the host code:
   - Choose a reasonable number of elements (e.g., 1<<20 for a million elements).
   - Allocate device memory for the output array.
   - Configure the kernel launch parameters (number of threads per block and number of blocks) to cover all elements.
   - Launch the kernel.
   - Copy the result back to host memory.
   - Verify each element equals its index; report success or failure.
3. Use simple error checking for CUDA API calls to ensure robust execution.
4. Keep the code minimal and self‑contained, ready to compile with `nvcc`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Kernel: each thread writes its global ID into the output array */
__global__ void write_global_id(int *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = idx;
    }
}

int main(void)
{
    const int N = 1 << 20;          // Number of elements (1,048,576)
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int *d_output = NULL;
    int *h_output = NULL;

    /* Allocate device memory */
    CHECK_CUDA(cudaMalloc((void**)&d_output, N * sizeof(int)));

    /* Allocate host memory */
    h_output = (int*)malloc(N * sizeof(int));
    if (h_output == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Launch kernel */
    write_global_id<<<blocks, threadsPerBlock>>>(d_output, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Verify results */
    int success = 1;
    for (int i = 0; i < N; ++i) {
        if (h_output[i] != i) {
            fprintf(stderr, "Verification failed at index %d: expected %d, got %d\n",
                    i, i, h_output[i]);
            success = 0;
            break;
        }
    }

    if (success) {
        printf("Verification succeeded: all %d elements are correct.\n", N);
    } else {
        printf("Verification failed.\n");
    }

    /* Clean up */
    free(h_output);
    CHECK_CUDA(cudaFree(d_output));

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```