```c
/*
Write a function `void print_device_array(int* d_array, int size)` that allocates temporary host memory, copies from device, prints, and then frees the temporary memory.

My thought process:

1. The function needs to accept a device pointer `d_array` and its length `size`.
2. To access the data on the host, we allocate a temporary host array of the same size.
3. We then use `cudaMemcpy` with the `cudaMemcpyDeviceToHost` flag to copy the data from device to host.
4. After a successful copy, we iterate over the host array and print each element (separated by spaces) followed by a newline.
5. Finally, we free the host memory to avoid leaks.
6. Error handling is added for both host allocation and the CUDA copy.
7. For demonstration purposes, a small `main` function is included that allocates a device array, fills it with data, calls `print_device_array`, and then frees the device memory.

This should be sufficient for the requested functionality while also providing a minimal example of usage.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* 
 * Prints an integer array stored on the device.
 * Allocates temporary host memory, copies data from device to host,
 * prints the array elements, and frees the temporary memory.
 */
void print_device_array(int* d_array, int size)
{
    /* Allocate temporary host memory */
    int* h_array = (int*)malloc(size * sizeof(int));
    if (!h_array)
    {
        fprintf(stderr, "Host memory allocation failed\n");
        return;
    }

    /* Copy from device to host */
    cudaError_t err = cudaMemcpy(h_array, d_array, size * sizeof(int),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(h_array);
        return;
    }

    /* Print the array */
    for (int i = 0; i < size; ++i)
    {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    /* Free temporary host memory */
    free(h_array);
}

/* Simple kernel to fill device array with sequential numbers */
__global__ void fill_kernel(int* d_array, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_array[idx] = idx;
}

int main(void)
{
    const int N = 10;
    int *d_array;

    /* Allocate device memory */
    cudaError_t err = cudaMalloc((void**)&d_array, N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    /* Launch kernel to fill the array */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    fill_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, N);
    cudaDeviceSynchronize();

    /* Print the device array */
    print_device_array(d_array, N);

    /* Free device memory */
    cudaFree(d_array);

    return EXIT_SUCCESS;
}
```