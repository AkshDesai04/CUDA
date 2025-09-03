```cuda
/*
Aim of the program:
Create a `bool` flag `is_correct` and set it to `false` if any element mismatches. Print a final 'Verification successful' or 'Verification FAILED' message.

Thinking:
The task is to demonstrate how to validate GPU-computed results against expected CPU values using a boolean flag.
- Allocate an array of integers on both host and device.
- Initialize the host array with a simple pattern (e.g., 0, 1, 2, …).
- Launch a CUDA kernel that computes the square of each element.
- Copy the results back to the host.
- Compare each GPU result with the expected square value calculated on the CPU.
- If any element does not match, set the flag `is_correct` to `false`.
- After the comparison, print "Verification successful" if `is_correct` remains `true`, otherwise print "Verification FAILED".

This code includes minimal error checking and uses standard C++ for output. It is a complete, self‑contained `.cu` file that can be compiled with `nvcc`.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 1024

// CUDA kernel to compute the square of each element
__global__ void squareKernel(const int *input, int *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        int val = input[idx];
        output[idx] = val * val;
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(err)                                      \
    do {                                                     \
        cudaError_t err__ = (err);                           \
        if (err__ != cudaSuccess) {                          \
            fprintf(stderr, "CUDA Error: %s (err=%d) at %s:%d\n", \
                    cudaGetErrorString(err__), err__, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    } while (0)

int main()
{
    int *h_input = (int*)malloc(N * sizeof(int));
    int *h_output = (int*)malloc(N * sizeof(int));
    if (!h_input || !h_output)
    {
        fprintf(stderr, "Host memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    // Initialize host input array
    for (int i = 0; i < N; ++i)
        h_input[i] = i;

    // Device memory allocation
    int *d_input = nullptr;
    int *d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel: use 256 threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch

    // Copy output back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verification
    bool is_correct = true;
    for (int i = 0; i < N; ++i)
    {
        int expected = h_input[i] * h_input[i];
        if (h_output[i] != expected)
        {
            is_correct = false;
            printf("Mismatch at index %d: host %d, device %d, expected %d\n",
                   i, h_output[i], h_input[i], expected);
            // Break early if desired
            // break;
        }
    }

    if (is_correct)
        printf("Verification successful\n");
    else
        printf("Verification FAILED\n");

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
```