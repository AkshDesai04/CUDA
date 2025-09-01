/*
Write a helper function `void host_to_device_copy(int* h_data, int* d_data, int size)` that encapsulates the `cudaMemcpy` call.
I thought about the requirements: we need a single helper function that takes a pointer to host data, a pointer to device data, and the number of integers to copy. The function should call cudaMemcpy with the appropriate arguments (src, dst, size in bytes, and the direction flag). I also considered adding error checking for the CUDA call, but the prompt didn't explicitly ask for it, so I kept the function minimal while still making it safe to use. I added a simple main function that demonstrates usage: allocating host and device memory, initializing the host array, copying it to device, and finally freeing resources. This gives a complete, compilable example that can be used as a template for more complex programs. 
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Copies data from host to device.
 *
 * @param h_data Pointer to host data array.
 * @param d_data Pointer to device data array.
 * @param size Number of integers to copy.
 */
void host_to_device_copy(int* h_data, int* d_data, int size)
{
    // The number of bytes to copy is size * sizeof(int)
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);
}

int main()
{
    const int N = 10;
    int* h_array = (int*)malloc(N * sizeof(int));
    int* d_array = NULL;

    // Initialize host array
    for (int i = 0; i < N; ++i)
        h_array[i] = i;

    // Allocate device memory
    cudaMalloc((void**)&d_array, N * sizeof(int));

    // Copy data from host to device using the helper
    host_to_device_copy(h_array, d_array, N);

    // (Optional) Verify copy by copying back to host
    int* h_verify = (int*)malloc(N * sizeof(int));
    cudaMemcpy(h_verify, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Copied data:\n");
    for (int i = 0; i < N; ++i)
        printf("%d ", h_verify[i]);
    printf("\n");

    // Clean up
    free(h_array);
    free(h_verify);
    cudaFree(d_array);

    return 0;
}