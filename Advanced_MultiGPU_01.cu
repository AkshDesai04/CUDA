/*
Write a program that enumerates all GPUs on the system, and for each one, prints its name and memory capacity. Then, explicitly set the active device to GPU 1 (if it exists) using `cudaSetDevice` and run a simple kernel on it.

Thought Process:
- Use CUDA Runtime API to query the number of GPUs (`cudaGetDeviceCount`).
- For each device index, retrieve properties via `cudaGetDeviceProperties`, then print the name and total global memory in MB.
- Check if at least two GPUs are present; if so, switch to device 1 with `cudaSetDevice`.
- Allocate a small array on the chosen GPU, launch a trivial kernel that adds 1 to each element.
- Copy the results back to host to verify that the kernel executed.
- Include basic error checking after each CUDA call.
- The kernel is defined with `__global__` and uses simple thread indexing.
- Compile with `nvcc -o gpu_enum gpu_enum.cu` and run.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple kernel that adds 1 to each element of the array */
__global__ void addOneKernel(int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_arr[idx] += 1;
}

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", i, cudaGetErrorString(err));
            continue;
        }
        printf("Device %d: %s, Memory: %lu MB\n", i, prop.name,
               prop.totalGlobalMem / (1024 * 1024));
    }

    /* Attempt to set device 1 if it exists */
    int targetDevice = 1;
    if (deviceCount > targetDevice)
    {
        err = cudaSetDevice(targetDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to set device %d: %s\n", targetDevice, cudaGetErrorString(err));
            return -1;
        }
        printf("Switched to device %d.\n", targetDevice);
    }
    else
    {
        printf("Device 1 does not exist. Exiting.\n");
        return 0;
    }

    /* Allocate memory and launch kernel on device 1 */
    const int N = 10;
    int *h_arr = (int *)malloc(N * sizeof(int));
    if (!h_arr)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return -1;
    }

    for (int i = 0; i < N; ++i)
        h_arr[i] = i;

    int *d_arr;
    err = cudaMalloc((void **)&d_arr, N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return -1;
    }

    err = cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return -1;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addOneKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return -1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return -1;
    }

    err = cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data back to host: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return -1;
    }

    printf("Result after kernel on device %d:\n", targetDevice);
    for (int i = 0; i < N; ++i)
        printf("%d ", h_arr[i]);
    printf("\n");

    cudaFree(d_arr);
    free(h_arr);
    return 0;
}
