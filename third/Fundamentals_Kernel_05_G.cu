```c
/*
Aim of the program:
Write a kernel that calculates the global ID and then prints whether it is even or odd.

Thinking:
The requirement is to write a CUDA kernel that determines each thread's global ID
(i.e., a unique number across the entire grid) and then prints whether that ID is
even or odd. In CUDA device code we can use `printf` to write to the console,
but we must compile with device printf enabled (usually `-arch=sm_35` or higher).
The global ID is computed as:
    gid = blockIdx.x * blockDim.x + threadIdx.x
Once we have the ID, we use the modulo operator to check if it's even or odd
and then print a message. The host code will launch the kernel with a simple
grid and block configuration so we can see the output for several threads.
The program will include basic error checking for CUDA API calls to ensure
that any launch failures are reported. The final code is a self‑contained
`.cu` file that can be compiled with `nvcc` and executed directly.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            return EXIT_FAILURE;                                            \
        }                                                                   \
    } while (0)

// Kernel that prints whether the global ID is even or odd
__global__ void print_even_odd()
{
    // Compute global thread ID
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine parity
    const char* parity = (gid % 2 == 0) ? "even" : "odd";

    // Print result
    printf("Thread %d is %s\n", gid, parity);
}

int main(void)
{
    // Define grid and block dimensions
    const int threadsPerBlock = 4;
    const int blocksPerGrid = 2; // Total 8 threads

    // Launch kernel
    print_even_odd<<<blocksPerGrid, threadsPerBlock>>>();
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Wait for GPU to finish before accessing on host
    CUDA_CHECK(cudaDeviceSynchronize());

    return EXIT_SUCCESS;
}
```