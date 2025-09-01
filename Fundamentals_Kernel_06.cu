/*
Write a kernel that accepts an integer pointer `int* data` as an argument. The host code should allocate a single integer on the GPU (`cudaMalloc`), launch the kernel with one thread, and pass the pointer. The kernel should write the value `1337` to the memory location pointed to by `data`. Finally, the host should copy the value back (`cudaMemcpy`) and print it to verify.

Thinking:
1. The program requires a simple CUDA kernel that writes a constant value to a memory location.
2. We need to allocate memory on the device for a single integer, launch a single-thread kernel, and then copy the result back to host for printing.
3. Proper error handling will be added for each CUDA API call to ensure robustness.
4. The kernel signature: `__global__ void write_value(int *data)`.
5. The host main function will:
   - Allocate device memory with `cudaMalloc`.
   - Launch `write_value<<<1,1>>>(dev_ptr)`.
   - Synchronize and check errors.
   - Copy back the integer with `cudaMemcpy`.
   - Print the value to stdout.
6. Include necessary headers and compile with `nvcc`.
7. Ensure the code is self‑contained and follows the requested structure: a single .cu file with the comment block at the top followed by the code.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that writes the value 1337 into the memory pointed to by `data`
__global__ void write_value(int *data)
{
    *data = 1337;
}

int main(void)
{
    int *dev_ptr = NULL;
    int host_val = 0;
    cudaError_t err;

    // Allocate memory for one integer on the device
    err = cudaMalloc((void**)&dev_ptr, sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel with a single thread
    write_value<<<1, 1>>>(dev_ptr);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_ptr);
        return 1;
    }

    // Wait for the kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_ptr);
        return 1;
    }

    // Copy the result back to host
    err = cudaMemcpy(&host_val, dev_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_ptr);
        return 1;
    }

    // Print the value to verify
    printf("Value copied back from device: %d\n", host_val);

    // Clean up
    cudaFree(dev_ptr);

    return 0;
}
