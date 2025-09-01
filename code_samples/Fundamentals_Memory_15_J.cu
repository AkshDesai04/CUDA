/*
Use `cudaGetSymbolAddress` to get a device pointer to the `__device__` variable, which you can then pass to other API functions like `cudaMemset`.

My thinking:
- Define a global `__device__` array to hold device data.
- Use `cudaGetSymbolAddress` with the symbol name to obtain the device address of this global variable.
- Use `cudaMemset` to initialize the device memory (e.g., set each byte to `0xAB`).
- Copy the memory back to host using `cudaMemcpy` and print it to verify that the initialization succeeded.
- Include error handling for each CUDA API call.
- Keep the code simple and self-contained for demonstration purposes.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Global device array
__device__ int deviceArray[10];

int main(void) {
    // Pointer to hold the device address
    void *d_ptr = nullptr;
    cudaError_t err;

    // Get device address of the global variable
    err = cudaGetSymbolAddress(&d_ptr, "deviceArray");
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetSymbolAddress failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    size_t size = sizeof(int) * 10;

    // Initialize device memory with 0xAB bytes
    err = cudaMemset(d_ptr, 0xAB, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Allocate host array
    int h_array[10] = {0};

    // Copy data back from device to host
    err = cudaMemcpy(h_array, d_ptr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print the results
    printf("Contents of deviceArray after cudaMemset(0xAB):\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_array[%d] = 0x%X\n", i, h_array[i]);
    }

    return 0;
}
