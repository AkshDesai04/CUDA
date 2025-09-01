/*
Aim: In your helper function, if an error is found, print the error string using `cudaGetErrorString`.

Thinking:
- The goal is to create a reusable helper that checks the result of any CUDA API call.
- When an error occurs, the helper should print a descriptive message plus the string returned by `cudaGetErrorString`.
- This helper can be used throughout the program to simplify error handling.
- A small CUDA program is written to demonstrate the helper: it allocates a small device buffer, attempts a large memcpy to trigger an error, launches a trivial kernel, and synchronizes the device.
- Each CUDA call is wrapped by the helper to ensure any failure prints the error string and aborts the program.
- The code is self‑contained and compiles with `nvcc`.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

/* Helper function that checks a CUDA error code.
 * If the error code is not cudaSuccess, it prints the provided message
 * and the human readable string obtained from cudaGetErrorString,
 * then exits the program.
 */
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " : " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/* A simple empty kernel used only for demonstration. */
__global__ void simpleKernel()
{
    // No operation performed
}

int main()
{
    // Allocate a small device buffer intentionally smaller than the host data
    int* d_arr;
    size_t hostSize   = 1024 * sizeof(int); // 1024 integers
    size_t deviceSize = 10 * sizeof(int);   // only 10 integers

    checkCudaError(cudaMalloc((void**)&d_arr, deviceSize), "cudaMalloc failed");

    // Host data to be copied to device
    int* h_arr = new int[1024];
    for (int i = 0; i < 1024; ++i)
        h_arr[i] = i;

    /* This memcpy will fail due to size mismatch, triggering the helper */
    cudaError_t err = cudaMemcpy(d_arr, h_arr, hostSize, cudaMemcpyHostToDevice);
    checkCudaError(err, "cudaMemcpy failed");

    /* Launch the trivial kernel */
    simpleKernel<<<1, 256>>>();
    err = cudaGetLastError();
    checkCudaError(err, "Kernel launch failed");

    /* Synchronize to ensure all work is finished */
    err = cudaDeviceSynchronize();
    checkCudaError(err, "cudaDeviceSynchronize failed");

    // Clean up
    cudaFree(d_arr);
    delete[] h_arr;

    std::cout << "Program completed successfully." << std::endl;
    return 0;
}
