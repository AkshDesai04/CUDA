/*
Aim: Write a kernel to reverse an array within a single block. Each thread `i` loads `input[i]` into `shared_mem[i]`. After a `__syncthreads()`, thread `i` reads from `shared_mem[BLOCK_SIZE - 1 - i]` and writes to `output[i]`. This is much faster than global memory reversal.

Thinking: The reversal operation can be accelerated by staging the data in fast shared memory instead of performing many strided global memory accesses. Since a single block can fully participate in this operation, we simply copy the input array into a shared array (one element per thread), synchronize, and then write out the reversed data from the shared array. The kernel assumes that the number of elements `N` does not exceed `BLOCK_SIZE` and that all threads in the block participate. The code includes basic CUDA error checking, host-side memory allocation, data transfer, kernel launch, and result verification. The program demonstrates the speed‑up benefit by printing the original and reversed arrays. This self‑contained .cu file can be compiled with `nvcc` and executed on a CUDA‑capable device. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// CUDA error checking macro
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that reverses an array using shared memory
__global__ void reverseArrayKernel(const int *input, int *output, int N)
{
    __shared__ int sdata[BLOCK_SIZE];
    int idx = threadIdx.x;

    // Load into shared memory if within bounds
    if (idx < N) {
        sdata[idx] = input[idx];
    }

    __syncthreads();

    // Write reversed element to output
    if (idx < N) {
        output[idx] = sdata[N - 1 - idx];
    }
}

int main(void)
{
    const int N = BLOCK_SIZE;   // Number of elements to reverse
    size_t bytes = N * sizeof(int);

    // Allocate host memory
    int *h_in  = (int*)malloc(bytes);
    int *h_out = (int*)malloc(bytes);

    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input array with sample data
    for (int i = 0; i < N; ++i) {
        h_in[i] = i + 1;  // Fill with 1..N
    }

    // Print original array
    printf("Original array:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_in[i]);
    }
    printf("\n");

    // Allocate device memory
    int *d_in = NULL;
    int *d_out = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_in,  bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_out, bytes));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Launch kernel: one block of BLOCK_SIZE threads
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(1);
    reverseArrayKernel<<<gridDim, blockDim>>>(d_in, d_out, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Print reversed array
    printf("Reversed array:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Clean up
    free(h_in);
    free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return EXIT_SUCCESS;
}
