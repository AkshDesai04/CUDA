/*
Aim of the program: Modify the program to launch the kernel with one block containing two threads.
Thoughts:
1. The original prompt asks for a CUDA .cu file that launches a kernel with a single block of two threads.
2. The kernel itself can be very simple; for example, each thread writes its index to an array.
3. We'll allocate a small array on the device, launch the kernel, copy the result back, and print it.
4. The launch configuration will be kernel<<<1, 2>>>(); where the first parameter is the number of blocks and the second is the number of threads per block.
5. We'll include error checking for kernel launch and memory operations.
6. Since device printf requires certain CUDA versions, to keep it portable we will avoid printf from the device and instead write results to a host array.
7. The main flow:
   - Allocate device array of size 2.
   - Launch kernel<<<1,2>>>; each thread writes its threadIdx.x into the array.
   - Copy the array back to host.
   - Print the values to confirm the two threads ran.
8. This fulfills the requirement of launching one block with two threads and demonstrates a working CUDA program.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel that writes the thread index into the output array
__global__ void write_thread_idx(int *out)
{
    int idx = threadIdx.x;
    out[idx] = idx;
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void)
{
    const int num_threads = 2;
    const int num_blocks = 1;
    const size_t array_size = num_threads * sizeof(int);

    // Allocate device memory
    int *d_out;
    CHECK_CUDA(cudaMalloc((void **)&d_out, array_size));

    // Launch kernel with one block of two threads
    write_thread_idx<<<num_blocks, num_threads>>>(d_out);

    // Check for launch errors
    CHECK_CUDA(cudaGetLastError());

    // Allocate host memory and copy results back
    int h_out[num_threads];
    CHECK_CUDA(cudaMemcpy(h_out, d_out, array_size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Thread outputs:\n");
    for (int i = 0; i < num_threads; ++i) {
        printf("  h_out[%d] = %d\n", i, h_out[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
