```cu
/*
Aim: Launch the hanging kernel and try to use nvidia-smi in another terminal. You should see the GPU utilization at 100%.

Thinking:
To demonstrate that a kernel can keep the GPU busy indefinitely, we create a simple CUDA program that launches a kernel which enters an infinite loop. A naive infinite loop like `while(true) {}` may be optimized away by the compiler if it doesn't have side effects, so we use `__syncthreads()` inside the loop. This acts as a memory fence and ensures the loop body has a noticeable effect that prevents optimization while also keeping all threads synchronized at the same point. By launching many blocks and threads, we saturate the GPU’s execution units. After launching, we call `cudaDeviceSynchronize()` so the host program blocks until the kernel finishes; since the kernel never finishes, the host process will hang, keeping the GPU busy. Running `nvidia-smi` in another terminal while this program runs should show 100% GPU utilization. No external libraries or complex logic are needed; just basic CUDA API calls and error checking.

Key points:
1. Define a kernel that loops forever, using `__syncthreads()` to avoid compiler elimination.
2. Launch the kernel with a reasonable number of blocks and threads to saturate the GPU.
3. Call `cudaDeviceSynchronize()` to block the host process, preventing it from exiting.
4. Print a message indicating the kernel has been launched.
5. Include error checking for the kernel launch.
6. Keep the code self‑contained and compile‑ready with `nvcc`.

*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Infinite loop kernel that keeps the GPU busy */
__global__ void hangingKernel()
{
    while (true)
    {
        /* Barrier to prevent compiler optimization and keep threads synchronized */
        __syncthreads();
    }
}

int main()
{
    /* Configure grid and block dimensions */
    dim3 blockSize(256);      // 256 threads per block
    dim3 gridSize(256);       // 256 blocks

    /* Launch the hanging kernel */
    hangingKernel<<<gridSize, blockSize>>>();

    /* Check for launch errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    /* Inform the user that the kernel is running */
    printf("Hanging kernel launched. GPU should be 100%% utilized.\n");
    printf("Press Ctrl+C to terminate the program.\n");

    /* Block until the kernel finishes (it never does) */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error during cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
```