/*
Use the alternative syntax for `cudaMemcpyToSymbol` which takes the symbol as a variable: `cudaMemcpyToSymbol(my_const, ...)`.

In this program I demonstrate the use of the alternative `cudaMemcpyToSymbol` syntax that accepts the symbol as a variable rather than a symbol name string.  I first declare a constant memory array `my_const` on the device, then allocate a host array `h_data` with the same size, fill it with some values, and copy those values into the constant memory using `cudaMemcpyToSymbol(my_const, ...)`.  A simple kernel `use_const` multiplies each element of a global array by the corresponding constant memory value.  The kernel is launched with a one‑to‑one mapping of threads to array elements.  After kernel execution I copy the resulting global array back to the host and print it to verify that the constant memory was correctly loaded and used by the kernel.  The code is fully self‑contained, compiles with `nvcc`, and uses only standard CUDA runtime APIs.  The key line that illustrates the alternative syntax is:

    cudaMemcpyToSymbol(my_const, h_data, size);

All other parts of the program are standard CUDA boilerplate: memory allocation, error checking, kernel launch, and cleanup.  The program outputs the final array values so that one can visually confirm the multiplication operation performed using the constant memory.  This serves both as a functional example and a minimal test harness for the requested syntax.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

// Declare a constant memory array of size N
__constant__ int my_const[N];

// Kernel that multiplies each element of the input array by the corresponding constant value
__global__ void use_const(const int *in, int *out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        out[idx] = in[idx] * my_const[idx];
    }
}

// Helper macro for error checking
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main(void)
{
    int h_in[N];
    int h_out[N];

    // Initialize host input array with some values
    for (int i = 0; i < N; ++i)
    {
        h_in[i] = i + 1; // 1, 2, ..., N
    }

    // Allocate device memory for input and output
    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_out, N * sizeof(int)));

    // Copy host input array to device input array
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // Prepare constant memory data: for demonstration, let's use the same values as input
    int h_const[N];
    for (int i = 0; i < N; ++i)
    {
        h_const[i] = i + 1; // same as h_in
    }

    // Copy constant data to constant memory using the alternative syntax
    CUDA_CHECK(cudaMemcpyToSymbol(my_const, h_const, N * sizeof(int)));

    // Launch kernel: one thread per element
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    use_const<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Input\tConst\tOutput (input * const)\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%5d\t%5d\t%6d\n", h_in[i], h_const[i], h_out[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
