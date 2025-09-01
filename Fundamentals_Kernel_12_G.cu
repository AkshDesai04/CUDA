```cpp
/*
Aim of the program (word for word):
Read the desired number of threads from the command line and dynamically calculate the grid/block `dim3` variables at runtime.

Thinking:
The requirement is to read a command-line argument that specifies how many threads the user wants to launch in a CUDA kernel. 
From that number we must compute the block and grid dimensions at runtime, rather than using fixed compile-time constants.
A common pattern is to set a maximum block size (e.g., 256 threads per block) and then compute the number of blocks needed to cover all
requested threads: gridDim.x = (threads + blockDim.x - 1) / blockDim.x.
The kernel itself can be very simple – for example, writing its global thread index into an array – just to demonstrate that the threads
are being launched correctly. After the kernel completes, we copy the data back to the host and print a few entries to verify execution.
This example also includes basic error checking for CUDA API calls and argument parsing.

The program flow:
1. Parse the desired thread count from argv[1] (or print usage if missing).
2. Set blockDim.x to a reasonable value (e.g., 256) but cap it at the requested thread count if that is smaller.
3. Compute gridDim.x from the formula above.
4. Allocate an array of size `threads` on the device.
5. Launch a simple kernel that writes the global thread index into the array.
6. Copy the array back to host memory.
7. Print the first few elements (or all if the array is small) to confirm the kernel ran.
8. Clean up memory and exit.

This code is written for CUDA C++ and can be compiled with `nvcc -o dyn_grid_dyn_block dyn_grid_dyn_block.cu`.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro for checking CUDA API call results
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                              \
        }                                                                     \
    } while (0)

// Kernel: each thread writes its global index into the output array
__global__ void fill_indices(int *out, int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
        out[idx] = idx;
}

int main(int argc, char *argv[])
{
    // Check command line arguments
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <desired_thread_count>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Parse desired thread count
    int desired_threads = atoi(argv[1]);
    if (desired_threads <= 0) {
        fprintf(stderr, "Error: desired_thread_count must be a positive integer.\n");
        return EXIT_FAILURE;
    }

    // Decide block size (choose a reasonable maximum, e.g., 256)
    const int max_block_size = 256;
    int block_size = (desired_threads < max_block_size) ? desired_threads : max_block_size;

    // Compute grid size to cover all desired threads
    int grid_size = (desired_threads + block_size - 1) / block_size;

    // Report the configuration
    printf("Launching kernel with:\n");
    printf("  Desired threads: %d\n", desired_threads);
    printf("  Block size    : %d\n", block_size);
    printf("  Grid size     : %d\n", grid_size);
    printf("  Total launched: %d\n\n", grid_size * block_size);

    // Allocate host memory
    int *h_output = (int*)malloc(desired_threads * sizeof(int));
    if (!h_output) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_output, desired_threads * sizeof(int)));

    // Launch kernel
    fill_indices<<<grid_size, block_size>>>(d_output, desired_threads);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, desired_threads * sizeof(int), cudaMemcpyDeviceToHost));

    // Print first few results (or all if small)
    int print_count = (desired_threads < 20) ? desired_threads : 20;
    printf("First %d elements of output array:\n", print_count);
    for (int i = 0; i < print_count; ++i) {
        printf("  h_output[%d] = %d\n", i, h_output[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_output));
    free(h_output);

    printf("\nCompleted successfully.\n");
    return EXIT_SUCCESS;
}
```