/*
Write a program that allocates an array of 256 integers on the host and initializes it with values from 0 to 255. Then, allocate memory for 256 integers on the GPU device using `cudaMalloc`. Finally, copy the host array to the device array using `cudaMemcpy` with the `cudaMemcpyHostToDevice` kind.

Thinking:
1. The program is a minimal demonstration of host-to-device memory transfer in CUDA.
2. We need to include the CUDA runtime header for memory functions.
3. The host array is a simple static array of size 256.
4. Initialize it with a loop from 0 to 255.
5. Allocate device memory using cudaMalloc; check the returned error status.
6. Use cudaMemcpy with the correct direction flag to copy the data.
7. After copying, we can optionally print a confirmation or the first few elements from the device memory by copying back, but it's not required per prompt.
8. Finally, free both host and device memory and return.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    const int N = 256;
    int h_array[N];
    int *d_array = NULL;
    cudaError_t err;

    // Initialize host array
    for (int i = 0; i < N; ++i)
    {
        h_array[i] = i;
    }

    // Allocate device memory
    err = cudaMalloc((void **)&d_array, N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy from host to device
    err = cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy HostToDevice failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        return 1;
    }

    printf("Successfully copied %d integers from host to device.\n", N);

    // Clean up
    cudaFree(d_array);

    return 0;
}
