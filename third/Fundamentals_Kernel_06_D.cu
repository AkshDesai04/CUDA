```cpp
/*
Use a `float*` instead of an `int*` and write the value `3.14f`.

Thinking:
- The goal is to demonstrate using a pointer to float in CUDA.
- Allocate a device array of floats, launch a kernel that writes 3.14f into each element.
- Copy the array back to host and print the values to verify.
- Use simple error checking for CUDA API calls.
- Keep the program minimal and self‑contained.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void write_pi(float *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = 3.14f;
    }
}

int main() {
    const int N = 10;
    float *h_arr = (float *)malloc(N * sizeof(float));
    float *d_arr = nullptr;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void **)&d_arr, N * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel: one thread per element
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    write_pi<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Print the values
    printf("Values written by the kernel:\n");
    for (int i = 0; i < N; ++i) {
        printf("h_arr[%d] = %f\n", i, h_arr[i]);
    }

    // Clean up
    cudaFree(d_arr);
    free(h_arr);
    return 0;
}
```