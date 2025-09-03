/*
What happens if you forget to copy a value to the constant memory before the kernel uses it? What is its initial value?

This program demonstrates the behavior of CUDA constant memory when a value is not copied into it before kernel execution. In CUDA, symbols declared in constant memory are zero‑initialized by the device driver. Thus, if a kernel reads from such a symbol without a prior cudaMemcpyToSymbol, the value read will be zero (or the default representation for the data type). The program defines a constant integer, launches a kernel that prints its value without initializing it, then initializes it with a non‑zero value and launches the kernel again to show the difference. Device printf is used for output, so the device must support compute capability 2.0 or higher. The program includes simple error checking macros for clarity.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                                \
    } while (0)

// Constant memory variable
__constant__ int d_constVal;

// Kernel that prints the constant memory value
__global__ void printConst()
{
    // Use device printf to output the value
    printf("constVal = %d\n", d_constVal);
}

int main(void)
{
    // Launch kernel without initializing constant memory
    printConst<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());

    // Initialize constant memory with a non-zero value
    int h_val = 42;
    CHECK(cudaMemcpyToSymbol(d_constVal, &h_val, sizeof(int)));

    // Launch kernel again to see updated value
    printConst<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());

    // Reset device (optional)
    CHECK(cudaDeviceReset());

    return 0;
}
