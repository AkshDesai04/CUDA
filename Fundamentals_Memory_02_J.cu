/*
 * Write a function `void print_device_array(int* d_array, int size)` that allocates temporary host memory, copies from device, prints, and then frees the temporary memory.
 *
 * The aim of this code is to provide a utility function that helps developers debug CUDA programs by printing the contents of a device array to the host console.
 *
 * Thinking process:
 * 1. Allocate temporary host memory sufficient to hold `size` integers. We use standard `malloc` for simplicity, but `cudaMallocHost` could also be used for pinned memory if desired.
 * 2. Copy the data from device to host using `cudaMemcpy` with `cudaMemcpyDeviceToHost`. Always check for errors after the copy.
 * 3. Iterate over the host array and print each element. For readability, we print each integer followed by a space, and after the loop we print a newline.
 * 4. Free the allocated host memory with `free`.
 * 5. Provide simple error handling for CUDA calls: after each CUDA API call we check `cudaGetLastError` and print a message if something went wrong.
 * 6. Include a small `main` function to demonstrate the usage of `print_device_array`. In a real project this would be omitted or replaced by actual kernels, but it makes the file self‑contained and compilable.
 * 7. Keep the code minimal, focusing on the required function and ensuring it compiles with `nvcc`.
 *
 * Note: The function assumes that `d_array` is a valid device pointer and that `size` is non‑negative. No bounds checking is performed beyond that.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper macro for error checking */
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/* Prints an array located on the device to the host console.
 * Allocates temporary host memory, copies the data, prints it,
 * and then frees the temporary memory. */
void print_device_array(int* d_array, int size)
{
    if (size <= 0) {
        printf("Array size is non‑positive (%d). Nothing to print.\n", size);
        return;
    }

    /* Allocate temporary host memory */
    int* h_array = (int*)malloc(size * sizeof(int));
    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory for printing.\n");
        return;
    }

    /* Copy from device to host */
    CHECK_CUDA(cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost));

    /* Print the array */
    for (int i = 0; i < size; ++i) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    /* Free host memory */
    free(h_array);
}

/* Example kernel that initializes an array with a simple pattern */
__global__ void init_kernel(int* d_arr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_arr[idx] = idx * 10 + blockIdx.x;
    }
}

int main(void)
{
    const int N = 16;
    int *d_arr = NULL;
    size_t bytes = N * sizeof(int);

    /* Allocate device memory */
    CHECK_CUDA(cudaMalloc((void**)&d_arr, bytes));

    /* Launch kernel to fill the array */
    int threadsPerBlock = 8;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    init_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Print the device array */
    printf("Device array contents:\n");
    print_device_array(d_arr, N);

    /* Clean up */
    CHECK_CUDA(cudaFree(d_arr));
    return 0;
}
