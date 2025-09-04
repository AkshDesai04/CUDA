/*
Aim: Create a parent kernel that launches a dynamically sized grid for a child kernel based on data computed within the parent.

Thinking:
- Use CUDA dynamic parallelism to launch child kernel from parent kernel.
- Parent computes desired grid size (e.g., based on an input parameter or some runtime data).
- For demonstration, parent uses a single thread to compute child grid size and launches child kernel.
- Child kernel performs a simple operation: write index values to an output array.
- Host sets up data, launches parent, synchronizes, then reads back results.
- Include error checking for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Child kernel: writes the global thread index into the output array
__global__ void childKernel(int *out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = idx;
    }
}

// Parent kernel: computes child grid size and launches childKernel
__global__ void parentKernel(int *out, int N, int *blocksWritten)
{
    // For simplicity, use only one thread in the parent to compute and launch child
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Compute desired child size (e.g., double the input size)
        int childN = N * 2;

        // Determine child grid dimensions
        int threadsPerBlock = 256;
        int childBlocks = (childN + threadsPerBlock - 1) / threadsPerBlock;

        // Launch child kernel
        childKernel<<<childBlocks, threadsPerBlock>>>(out, childN);

        // Record the number of blocks for host inspection
        if (blocksWritten) {
            blocksWritten[0] = childBlocks;
        }
    }
}

int main(void)
{
    const int N = 1000;          // Base input size
    const int childN = N * 2;    // Expected child size
    const size_t outSize = childN * sizeof(int);

    int *d_out = NULL;
    int *d_blocksWritten = NULL;

    // Allocate device memory for output and block count
    CUDA_CHECK(cudaMalloc((void**)&d_out, outSize));
    CUDA_CHECK(cudaMalloc((void**)&d_blocksWritten, sizeof(int)));

    // Launch parent kernel with a single thread block
    parentKernel<<<1, 1>>>(d_out, N, d_blocksWritten);
    CUDA_CHECK(cudaGetLastError());      // Check parent launch

    // Wait for all kernels (parent and child) to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back the number of blocks launched by child
    int h_blocksWritten = 0;
    CUDA_CHECK(cudaMemcpy(&h_blocksWritten, d_blocksWritten, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Child kernel launched with %d blocks (threads per block: 256)\n", h_blocksWritten);

    // Copy back output from device
    int *h_out = (int*)malloc(outSize);
    if (!h_out) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(h_out, d_out, outSize, cudaMemcpyDeviceToHost));

    // Print first few values to verify
    printf("First 10 values from child kernel output:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Cleanup
    free(h_out);
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_blocksWritten));

    return EXIT_SUCCESS;
}
