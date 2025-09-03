/*
Aim: Allocate on host, allocate on device, but forget the cudaMemcpy call. What is in the device memory?

Thinking:
When a CUDA program allocates memory on the host using malloc or new, the host memory is initialized only if explicitly done so.
When memory is allocated on the device using cudaMalloc, the device memory is *not* automatically initialized.
If the programmer allocates a device array but does not transfer any data from the host to the device via cudaMemcpy,
the contents of that device array remain whatever values were previously residing in that region of device memory.
This could be garbage from earlier allocations, residual data from a previous kernel, or simply zeros if the driver
chooses to zero-initialize memory (though this is not guaranteed and should not be relied upon).

The following CUDA program demonstrates this behavior. It:
1. Allocates a small integer array on the host and fills it with known values.
2. Allocates a corresponding array on the device but never copies the host data to it.
3. Launches a kernel that uses device-side printf to print the first few elements of the device array.
   Because we never copied data, these prints will show whatever was in that memory region before,
   illustrating that the device memory is uninitialized.
4. The program then cleans up resources.

Compile with: nvcc uninitialized_device.cu -o uninitialized_device
Run: ./uninitialized_device

Observe that the printed values are random / garbage rather than the expected host values.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printDeviceMemory(const int *d_arr, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        // Device-side printf requires that the device supports printf.
        printf("d_arr[%d] = %d\n", idx, d_arr[idx]);
    }
}

int main(void)
{
    const int N = 10;
    size_t size = N * sizeof(int);

    // Allocate and initialize host array
    int *h_arr = (int *)malloc(size);
    if (h_arr == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i)
    {
        h_arr[i] = i * 10;  // Arbitrary values for demonstration
    }

    // Allocate device array
    int *d_arr = NULL;
    cudaError_t err = cudaMalloc((void **)&d_arr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return EXIT_FAILURE;
    }

    // NOTE: Intentionally omit cudaMemcpy from host to device.

    // Launch kernel to print device memory contents
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printDeviceMemory<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Wait for kernel to finish and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
    }

    // Clean up
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
