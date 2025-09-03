/*
Aim: What happens to the value of a __device__ variable when the application exits?

Thinking: 
- A __device__ variable lives in GPU memory (device global memory) for the lifetime of the CUDA context.
- When the host program ends, the CUDA driver cleans up the context, which frees all device memory, including any __device__ variables.
- Therefore the value of a __device__ variable does not persist across separate runs of the application.
- This code demonstrates that by declaring a __device__ variable, updating it in a kernel, reading it back, and then resetting the device at the end. If you run the program again, the variable starts at its initial value again because the previous value was discarded when the context was destroyed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// A device‑global variable
__device__ int dev_counter = 0;

// Kernel that increments the device variable
__global__ void incCounter()
{
    // Simple atomic addition to avoid race conditions
    atomicAdd(&dev_counter, 1);
}

int main(void)
{
    int host_val = 0;

    // Launch the kernel to modify the device variable
    incCounter<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Copy the device variable back to the host
    cudaMemcpyFromSymbol(&host_val, dev_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
    printf("Device counter after kernel launch: %d\n", host_val);

    // Reset the device to clean up the CUDA context
    cudaDeviceReset();

    return 0;
}
