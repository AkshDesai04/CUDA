/*
Aim of the program: Launch with more threads than you need (e.g., for an array of size 100, launch 128 threads). Add a check `if (id < 100)` before printing or writing to memory.

[Thinking:
- The program demonstrates launching more CUDA threads than the number of elements in an array.
- It uses a simple kernel that processes each element of an array if the global thread index is less than the array size.
- The kernel prints the thread ID and the array value it processes, and writes a modified value back to an output array.
- After the kernel finishes, the host copies the output array back to CPU memory and prints its contents.
- Device-side printf is used, so the program is compiled with `-rdc=true` and run on a device that supports device printf.
- All CUDA API calls include basic error checking for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Size of the data array
#define ARRAY_SIZE 100
// Number of threads to launch (more than ARRAY_SIZE)
#define NUM_THREADS 128
// Number of threads per block
#define THREADS_PER_BLOCK 32

// Kernel that processes elements only if the global thread index is within bounds
__global__ void processArray(const int *input, int *output, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < N) {
        // Print from device (requires device printf support)
        printf("Thread %d processing index %d, value %d\n", id, id, input[id]);

        // Example operation: increment each element by 1
        output[id] = input[id] + 1;
    }
}

int main(void)
{
    // Host arrays
    int h_input[ARRAY_SIZE];
    int h_output[ARRAY_SIZE];

    // Initialize input data
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        h_input[i] = i * 2;   // arbitrary values
    }

    // Device pointers
    int *d_input = NULL;
    int *d_output = NULL;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_input, ARRAY_SIZE * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_input failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaMalloc((void**)&d_output, ARRAY_SIZE * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_output failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        return EXIT_FAILURE;
    }

    // Copy input data from host to device
    err = cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to device failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return EXIT_FAILURE;
    }

    // Determine grid dimensions
    int blocks = (NUM_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    processArray<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, ARRAY_SIZE);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return EXIT_FAILURE;
    }

    // Copy output data back to host
    err = cudaMemcpy(h_output, d_output, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to host failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return EXIT_FAILURE;
    }

    // Print the output array
    printf("\nHost received output array:\n");
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    // Reset device
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
