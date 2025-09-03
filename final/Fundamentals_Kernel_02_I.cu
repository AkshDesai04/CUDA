/*
Call `cudaDeviceSynchronize()` after the kernel launch. Does this guarantee the `printf` output will appear before subsequent host code executes? Explain in a comment. (Answer: No, `printf` is buffered).

The following CUDA program demonstrates that calling `cudaDeviceSynchronize()` does not guarantee that device `printf` output has been flushed to the host console before the host continues execution. Device `printf` output is stored in a buffer on the GPU and is typically flushed only when the device is reset or when a large enough amount of output has accumulated. The `cudaDeviceSynchronize()` function only guarantees that the kernel has finished executing, not that the printf buffer has been drained. Therefore, the host code that follows may execute and potentially print its own messages before the device printf output is actually displayed. To force the output to be printed earlier, one would need to explicitly flush the buffer by resetting the device or by using a host function that reads the buffer, but this is not done automatically by `cudaDeviceSynchronize()`.

The code below launches a kernel that prints a message from each thread, then calls `cudaDeviceSynchronize()`. After synchronization, the host prints its own message. Because the device `printf` buffer may not have been flushed yet, the order of the printed messages may not be deterministic.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernelPrint(int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        printf("Kernel thread %d prints before sync\n", idx);
    }
}

int main(void)
{
    int n = 5;
    int threadsPerBlock = 2;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    kernelPrint<<<blocksPerGrid, threadsPerBlock>>>(n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Synchronize device
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Host code execution after synchronization
    printf("Host code executed after cudaDeviceSynchronize\n");

    // Reset device to flush any remaining printf output (optional)
    // cudaDeviceReset();

    return 0;
}
