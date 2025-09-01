```cu
/*
Use `atomicAdd` to safely have multiple threads increment a counter member within the single shared struct instance.

The program demonstrates how to use CUDA's atomicAdd function to safely increment a counter that is stored in a struct located in shared memory. Each thread in a block increments the same counter atomically, preventing race conditions. The shared struct is written back to global memory after all threads in the block have finished. The main function launches a kernel with multiple blocks and prints the final counter value for each block, which should match the number of threads per block.
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct SharedStruct {
    int counter;
};

/* Kernel that uses atomicAdd on a counter inside a shared struct instance */
__global__ void atomicAddKernel(SharedStruct* d_struct)
{
    /* Shared memory instance of the struct */
    __shared__ SharedStruct s_struct;

    /* Initialize the counter once per block */
    if (threadIdx.x == 0) {
        s_struct.counter = 0;
    }
    __syncthreads();            // Ensure all threads see the initialized value

    /* Each thread increments the counter atomically */
    atomicAdd(&(s_struct.counter), 1);

    __syncthreads();            // Ensure all increments are done before writing back

    /* Write the result back to global memory once per block */
    if (threadIdx.x == 0) {
        d_struct[blockIdx.x] = s_struct;
    }
}

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                    \
    do {                                                    \
        cudaError_t err = call;                             \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__,             \
                    cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

int main(void)
{
    const int blocks = 2;              // Number of blocks
    const int threadsPerBlock = 256;   // Threads per block
    const int totalThreads = blocks * threadsPerBlock;

    /* Allocate memory for the struct array in global memory */
    SharedStruct* d_struct;
    CUDA_CHECK(cudaMalloc((void**)&d_struct, blocks * sizeof(SharedStruct)));

    /* Launch kernel */
    atomicAddKernel<<<blocks, threadsPerBlock>>>(d_struct);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy results back to host */
    SharedStruct h_struct[blocks];
    CUDA_CHECK(cudaMemcpy(h_struct, d_struct, blocks * sizeof(SharedStruct),
                          cudaMemcpyDeviceToHost));

    /* Print the results */
    for (int i = 0; i < blocks; ++i) {
        printf("Block %d counter: %d (expected %d)\n",
               i, h_struct[i].counter, threadsPerBlock);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_struct));

    return 0;
}
```