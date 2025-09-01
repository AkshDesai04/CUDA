/*
Can tools like Valgrind detect CUDA device memory leaks? (Not directly, but `cuda-memcheck --leak-check` can).

The aim of this program is to demonstrate that typical host‑side tools such as Valgrind cannot detect memory leaks that occur on the CUDA device.  We intentionally allocate memory on the GPU, use it, and then never free it.  Running this program under `cuda-memcheck --leak-check=full` will report the leaked device allocation, whereas running under Valgrind will give no information about device memory.

The code below is a minimal, self‑contained CUDA program.  It allocates a buffer on the device, launches a trivial kernel that writes to that buffer, and then exits without freeing the device memory.  The `checkCudaErrors` macro is used to catch any CUDA API errors.  The program can be compiled with `nvcc` and run normally, but to observe the leak detection one should execute:

    cuda-memcheck --leak-check=full ./device_leak_demo

Valgrind, on the other hand, will not report the leaked device memory, which illustrates the limitation of host‑side memory checkers for CUDA applications.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define checkCudaErrors(call)                                                       \
    do {                                                                            \
        cudaError_t err = (call);                                                   \
        if (err != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));                  \
            exit(EXIT_FAILURE);                                                    \
        }                                                                           \
    } while (0)

// A trivial kernel that writes to each element
__global__ void fillKernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = idx;
    }
}

// Function that allocates device memory and never frees it (intentional leak)
void allocateLeakingDeviceMemory(int N) {
    int *d_buf = NULL;
    size_t size = N * sizeof(int);

    // Allocate device memory
    checkCudaErrors(cudaMalloc((void **)&d_buf, size));

    // Launch kernel to fill the buffer
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    fillKernel<<<blocks, threadsPerBlock>>>(d_buf, N);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Intentionally do NOT free d_buf to create a leak
    // Normally we would call cudaFree(d_buf);
}

int main(int argc, char **argv) {
    int N = 1024 * 1024; // 1M integers

    printf("Allocating and using device memory (intentional leak)...\n");
    allocateLeakingDeviceMemory(N);
    printf("Finished. Device memory was not freed.\n");

    // Optionally reset device to release all resources at program termination
    // (commented out to preserve the leak for detection tools)
    // checkCudaErrors(cudaDeviceReset());

    return 0;
}
