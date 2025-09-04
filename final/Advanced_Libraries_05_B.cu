/*
Use CUB's `DeviceRadixSort::SortPairs` to sort key-value pairs.

Thinking:
1. Include the CUB header `<cub/cub.cuh>` and standard CUDA headers.
2. Use a simple example: an array of integer keys and float values of size N.
3. Allocate device memory for keys, values, and output arrays.
4. Allocate temporary storage required by `DeviceRadixSort::SortPairs`. This requires a two-pass approach: first call with a null pointer to get the size, then allocate the buffer and call again.
5. Invoke `cub::DeviceRadixSort::SortPairs` with the appropriate template parameters (key type, value type, number of items).
6. Copy sorted pairs back to host and print to verify.
7. Free device memory and temporary storage.
8. Wrap everything in a `main` function and compile with `nvcc`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

int main() {
    const int N = 10;
    // Host input data
    int h_keys[N];
    float h_vals[N];
    for (int i = 0; i < N; ++i) {
        h_keys[i] = rand() % 100;    // random keys 0-99
        h_vals[i] = (float)(i);     // simple values
    }

    // Device pointers
    int *d_keys_in = nullptr;
    int *d_keys_out = nullptr;
    float *d_vals_in = nullptr;
    float *d_vals_out = nullptr;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Allocate device memory
    cudaMalloc((void**)&d_keys_in,  N * sizeof(int));
    cudaMalloc((void**)&d_keys_out, N * sizeof(int));
    cudaMalloc((void**)&d_vals_in,  N * sizeof(float));
    cudaMalloc((void**)&d_vals_out, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_keys_in, h_keys, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals_in, h_vals, N * sizeof(float), cudaMemcpyHostToDevice);

    // Determine temporary device storage requirements
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_keys_in, d_keys_out,
                                    d_vals_in, d_vals_out,
                                    N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Sort key-value pairs
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_keys_in, d_keys_out,
                                    d_vals_in, d_vals_out,
                                    N);

    // Copy results back to host
    int h_keys_out[N];
    float h_vals_out[N];
    cudaMemcpy(h_keys_out, d_keys_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vals_out, d_vals_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print sorted pairs
    printf("Sorted key-value pairs:\n");
    for (int i = 0; i < N; ++i) {
        printf("Key: %d, Value: %.2f\n", h_keys_out[i], h_vals_out[i]);
    }

    // Cleanup
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_vals_in);
    cudaFree(d_vals_out);
    cudaFree(d_temp_storage);

    return 0;
}
