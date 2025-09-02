/*
What happens if you allocate memory but your program crashes before `cudaFree` is called? Does the memory get reclaimed?

In this program we allocate a block of device memory with `cudaMalloc`, launch a simple kernel to
demonstrate that the allocation succeeded, and then deliberately cause a host segmentation fault by
dereferencing a null pointer. We do not call `cudaFree` before the crash. The aim is to illustrate
that when a CUDA-enabled application terminates abnormally, the CUDA driver automatically releases
all device memory that was allocated by that process. This is similar to how the operating system
reclaims process memory on exit. By observing the GPU memory usage (e.g., with `nvidia-smi` or
`cudaMemGetInfo`) before and after running this program, you will see that the device memory
allocated in this run is reclaimed once the process dies. This demonstrates that you do not
necessarily have to explicitly free device memory on every crash; however, in long-running
applications or those that recover from errors, it is still good practice to free memory when it
is no longer needed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel(int *d_ptr, int val) {
    int idx = threadIdx.x;
    if (idx == 0) {
        d_ptr[0] = val;
    }
}

int main() {
    int *d_data;
    size_t size = 256 * sizeof(int);

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_data, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("Device memory allocated at %p\n", (void*)d_data);

    // Launch a simple kernel to verify allocation works
    dummyKernel<<<1, 256>>>(d_data, 42);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data); // Clean up before exit
        return -1;
    }
    cudaDeviceSynchronize();
    printf("Kernel executed successfully.\n");

    // Intentionally crash the host program before freeing device memory
    // This simulates a crash scenario where cudaFree is not called.
    int *null_ptr = NULL;
    *null_ptr = 0; // This causes a segmentation fault

    // The following code is unreachable due to the crash above
    // cudaFree(d_data);
    // return 0;
    return 0;
}
