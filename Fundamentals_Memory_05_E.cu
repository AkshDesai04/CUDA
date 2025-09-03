/*
Aim: Use `nvidia-smi` to watch the GPU memory usage while your program runs. Step through with a debugger to see the memory increase after `cudaMalloc` and decrease after `cudaFree`.

Thinking:
The goal of this CUDA program is to demonstrate how GPU memory usage changes when we allocate and free device memory. By compiling and running this code while monitoring GPU memory with `nvidia-smi`, you should observe a spike in the memory usage after the `cudaMalloc` call, and a corresponding drop after the `cudaFree` call. This simple example also includes a basic kernel launch to ensure the memory allocation is actually used on the device, which can be useful for debugging and stepping through with a CUDA debugger like `cuda-gdb` or Nsight Systems.

Key points in the code:
1. Allocate a large array of integers (64 MB) on the device using `cudaMalloc`.
2. Launch a trivial kernel that writes the index value into each array element; this forces the allocation to be touched by the GPU.
3. Synchronize to make sure the kernel has finished before freeing the memory.
4. Free the memory with `cudaFree`.
5. Perform basic error checking for each CUDA API call to make debugging easier.

When you run this program, open another terminal and execute `watch -n 0.5 nvidia-smi` (or simply `nvidia-smi` if you prefer to manually refresh). You should see the GPU memory consumption rise during the allocation and kernel execution, then drop back after `cudaFree`. Step through the code in a debugger to observe the exact moments when the memory usage changes.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void simpleKernel(int *d_arr, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_arr[idx] = static_cast<int>(idx);
}

int main(void)
{
    const size_t N = 1 << 24;          // 16,777,216 integers (~64 MB)
    int *d_ptr = nullptr;
    cudaError_t err;

    /* Allocate memory on the device */
    err = cudaMalloc((void**)&d_ptr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    /* Launch kernel to use the allocated memory */
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    simpleKernel<<<blocks, threadsPerBlock>>>(d_ptr, N);

    /* Check for kernel launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_ptr);  // Clean up before exiting
        return EXIT_FAILURE;
    }

    /* Wait for kernel to finish */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_ptr);
        return EXIT_FAILURE;
    }

    /* Free the allocated memory */
    err = cudaFree(d_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
