/*
Write a comment in your code showing the full `nvcc` command used to compile it.

The goal of this program is to demonstrate how to embed the exact NVCC compilation command
inside a CUDA source file.  We will write a very simple kernel that does nothing
and launch it from the host.  The program prints a message on success so that the
output is visible when it is executed.  The key part is the multiline comment at
the end of the file that contains the full `nvcc` command used to compile this
source.  By putting the command in the source we make it self‑documenting,
which is useful for examples and educational purposes.

The design choices are straightforward: a trivial kernel named `dummyKernel`,
host code that launches it, checks for errors, synchronizes, and reports
success.  All CUDA API calls are wrapped with simple error checks using
`cudaGetLastError()` and `cudaDeviceSynchronize()`.  The program compiles
and runs on any GPU that supports at least the compute capability specified
in the compile command (e.g. `-arch=sm_35`).

The compilation command at the end of the file is intentionally simple,
without extra options like debugging or profiling flags, to keep it clear
what is needed to build this example.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// A simple kernel that performs no operations.
__global__ void dummyKernel() {
    // Intentionally left blank.
}

int main(void) {
    cudaError_t err;

    // Launch the dummy kernel with a single thread block of one thread.
    dummyKernel<<<1, 1>>>();

    // Check for kernel launch errors.
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the device to finish.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Dummy kernel executed successfully.\n");
    return 0;
}

/*
nvcc -arch=sm_35 -o dummy dummy.cu
*/