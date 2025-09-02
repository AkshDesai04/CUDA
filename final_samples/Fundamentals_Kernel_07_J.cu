/*
Aim: Pass an integer offset d. Have threads print (threadIdx.x + d, threadIdx.y + d).

Thought Process:
1. The program must accept an integer offset `d` from the user. The simplest way to pass this
   value is through a command-line argument. If no argument is provided we will default to 0.
2. We need to launch a CUDA kernel that runs in a 2‑D configuration so that each thread
   has `threadIdx.x` and `threadIdx.y`. Each thread will compute
   `x = threadIdx.x + d` and `y = threadIdx.y + d` and print the pair.
3. Device side printing is done with `printf` (supported in modern CUDA). We must
   ensure the host synchronizes after the kernel launch so all prints are flushed.
4. The kernel itself is straightforward: a single loop that prints the result.
5. For safety and debugging we include basic CUDA error checking after kernel launch.
6. The entire program is written in C‑style CUDA (.cu file). No external libraries are used.
7. The grid and block dimensions are set to a small example (e.g., 4x4 blocks, 2x2 grid)
   so that output is manageable. This can be modified by the user if needed.
8. The code uses `nvcc` for compilation, so it should compile with `nvcc -o print_offset print_offset.cu`.

*/

#include <cstdio>
#include <cstdlib>

__global__ void printOffsetKernel(int d) {
    // Each thread computes its offset and prints the result
    int x = threadIdx.x + d;
    int y = threadIdx.y + d;
    printf("(%d, %d)\n", x, y);
}

int main(int argc, char* argv[]) {
    int d = 0; // default offset

    if (argc > 1) {
        // Try to parse the first argument as an integer offset
        char* endptr = nullptr;
        long val = std::strtol(argv[1], &endptr, 10);
        if (*endptr == '\0' && endptr != argv[1]) {
            d = static_cast<int>(val);
        } else {
            fprintf(stderr, "Invalid offset value. Using default 0.\n");
        }
    } else {
        printf("No offset provided. Using default 0.\n");
    }

    // Define block and grid dimensions
    dim3 blockDim(4, 4); // 4x4 threads per block
    dim3 gridDim(2, 2);  // 2x2 blocks

    // Launch the kernel
    printOffsetKernel<<<gridDim, blockDim>>>(d);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for the kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA sync error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
