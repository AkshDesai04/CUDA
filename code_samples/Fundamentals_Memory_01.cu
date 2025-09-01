/*
Write a program that allocates an array of 256 integers on the host and initializes it with values from 0 to 255. Then, allocate memory for 256 integers on the GPU device using `cudaMalloc`. Finally, copy the host array to the device array using `cudaMemcpy` with the `cudaMemcpyHostToDevice` kind.

The goal of this program is straightforward: set up data on the host, transfer it to the device, and verify that the transfer succeeds. The main challenges are correctly managing memory on both sides and handling CUDA errors. The code will:

1. Allocate an array of 256 `int` on the host using `malloc`. It will check that the allocation succeeded.
2. Initialize the host array with values from 0 to 255 using a simple loop.
3. Allocate space for 256 `int` on the GPU with `cudaMalloc`. The returned pointer will be checked for errors.
4. Copy the host array to the device array with `cudaMemcpy`, specifying the `cudaMemcpyHostToDevice` direction.
5. Report success or any error messages using `cudaGetErrorString`.
6. Clean up both host and device memory before exiting.

Throughout, error checking is performed after each CUDA API call. This is crucial because any failure (e.g., insufficient device memory) should be caught early, and resources should be freed properly to avoid leaks. The program ends with a clean exit code indicating success (0) or failure (non‑zero). This structure is typical for introductory CUDA examples that focus on memory management rather than kernel execution.
*/
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    const int N = 256;
    int *h_arr = (int*)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return 1;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    // Allocate device memory
    int *d_arr = NULL;
    cudaError_t err = cudaMalloc((void**)&d_arr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return 1;
    }

    // Copy from host to device
    err = cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    printf("Data successfully copied to device.\n");

    // Clean up
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
