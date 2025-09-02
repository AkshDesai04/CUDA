/*
Modify the program to allocate space for 10 integers. Have the kernel write `1337` to the 5th element (`data[4]`). Verify on the host.

Thinking:
- We need a CUDA C (.cu) file.
- Allocate an array of 10 integers on the GPU using cudaMalloc.
- Write a simple kernel that writes the constant 1337 into the 5th element (index 4).
- Launch the kernel with a single thread (1 block, 1 thread) because only one element needs to be written.
- Copy the data back to the host with cudaMemcpy.
- Verify that the host array's 5th element equals 1337, and optionally print success or failure.
- Include basic error checking after CUDA API calls.
- The program should be self-contained, including headers, main function, and cleanup.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void writeKernel(int *d_data)
{
    // Write 1337 to the 5th element (index 4)
    d_data[4] = 1337;
}

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                      \
    do {                                                       \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",  \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

int main(void)
{
    const int N = 10;                      // Number of integers
    int *d_data = NULL;                    // Device pointer
    int h_data[N];                         // Host array

    // Allocate device memory for 10 integers
    CHECK_CUDA(cudaMalloc((void**)&d_data, N * sizeof(int)));

    // Launch kernel: 1 block, 1 thread
    writeKernel<<<1, 1>>>(d_data);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy data back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify that the 5th element is 1337
    if (h_data[4] == 1337) {
        printf("Success: data[4] = %d\n", h_data[4]);
    } else {
        printf("Failure: data[4] = %d (expected 1337)\n", h_data[4]);
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_data));

    // Reset device (optional)
    cudaDeviceReset();

    return 0;
}
