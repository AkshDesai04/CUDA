/*
Aim of the program: Modify the kernel to increment the value at `data[0]`. The host should initialize the value to 100, copy it to the device, launch the kernel to increment it, copy it back, and verify the result is 101.

Thinking:

1. We need a simple CUDA program that demonstrates memory transfer between host and device, a kernel that updates a single element, and a verification step.
2. Use a single `int` on the host, initialize it to 100.
3. Allocate device memory for one `int` using `cudaMalloc`.
4. Copy the host value to the device using `cudaMemcpy`.
5. Define a kernel that increments `data[0]`. Since we only have one element, launching a single thread (1 block of 1 thread) is sufficient.
6. After kernel execution, copy the result back to the host with `cudaMemcpy`.
7. Check that the host value is now 101. Print success or error.
8. Add minimal error checking after CUDA API calls to aid debugging, but keep it concise.
9. Include necessary headers: `<stdio.h>` and `<cuda_runtime.h>`.
10. Use `cudaDeviceSynchronize()` to ensure kernel completion before copying back.
11. The code is self-contained and can be compiled with `nvcc example.cu -o example`.

Now we present the full code.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void increment(int *data)
{
    // Only one thread will execute this kernel
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Increment the value at data[0] by 1
        data[0] += 1;
    }
}

int main(void)
{
    int h_data = 100;                // Host data initialized to 100
    int *d_data = NULL;              // Device pointer

    // Allocate device memory for one int
    cudaError_t err = cudaMalloc((void**)&d_data, sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy host data to device
    err = cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy (host to device) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Launch kernel with 1 block of 1 thread
    increment<<<1, 1>>>(d_data);

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy (device to host) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Free device memory
    cudaFree(d_data);

    // Verify the result
    if (h_data == 101)
    {
        printf("Success: data[0] = %d\n", h_data);
    }
    else
    {
        printf("Failure: data[0] = %d, expected 101\n", h_data);
    }

    return 0;
}
