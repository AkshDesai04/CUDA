/*
Aim of the program: Allocate on host, allocate on device, but forget the `cudaMemcpy` call. What is in the device memory?

Thought process:
1. The user wants a CUDA program that illustrates the effect of allocating memory on the host and on the device, but not copying data from the host to the device.
2. We'll allocate a small array on the host and initialize it with known values.
3. We'll allocate a corresponding array on the device.
4. We intentionally omit `cudaMemcpy` from host to device to simulate forgetting the copy.
5. We'll launch a simple kernel that reads from the device array and writes the values to an output array on the device.
6. Then we copy the output array back to the host and print it to show that the input array on the device was uninitialized, thus the output will contain garbage values.
7. For clarity, we will also print the device memory contents directly using a CUDA kernel that writes them to a second host-accessible array, again without initializing them.
8. This demonstrates that without copying, the device memory holds whatever was there previously (often zeros for freshly allocated memory on some GPUs, but not guaranteed; can be garbage).
9. The program includes error checking for CUDA API calls and will print the results.

The final .cu file will compile with `nvcc` and run on any CUDA-capable device.

*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10  // Size of the arrays

// Simple kernel that copies input to output
__global__ void copyKernel(const int* d_in, int* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx];
    }
}

// Kernel to print device memory (by writing to host-readable array)
__global__ void dumpKernel(const int* d_mem, int* d_dump, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_dump[idx] = d_mem[idx];
    }
}

int main() {
    // Host memory allocation
    int h_hostArray[N];
    int h_result[N];
    int h_dump[N];

    // Initialize host array with known values
    for (int i = 0; i < N; ++i) {
        h_hostArray[i] = i * 10;  // e.g., 0, 10, 20, ...
    }

    // Device memory allocation
    int *d_in, *d_out, *d_dump;
    cudaError_t err;

    err = cudaMalloc((void**)&d_in, N * sizeof(int));
    if (err != cudaSuccess) { printf("cudaMalloc d_in failed: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMalloc((void**)&d_out, N * sizeof(int));
    if (err != cudaSuccess) { printf("cudaMalloc d_out failed: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMalloc((void**)&d_dump, N * sizeof(int));
    if (err != cudaSuccess) { printf("cudaMalloc d_dump failed: %s\n", cudaGetErrorString(err)); return 1; }

    // *** IMPORTANT *** Intentionally omit cudaMemcpy from host to device
    // cudaMemcpy(d_in, h_hostArray, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch copy kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    copyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("Kernel launch failed: %s\n", cudaGetErrorString(err)); return 1; }
    cudaDeviceSynchronize();

    // Copy result back to host
    err = cudaMemcpy(h_result, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("cudaMemcpy device to host failed: %s\n", cudaGetErrorString(err)); return 1; }

    // Dump the uninitialized device memory to host for inspection
    dumpKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_dump, N);
    cudaDeviceSynchronize();
    err = cudaMemcpy(h_dump, d_dump, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("cudaMemcpy dump failed: %s\n", cudaGetErrorString(err)); return 1; }

    // Print results
    printf("Host array (initialized):\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_hostArray[i]);
    }
    printf("\n\nResult from device after copyKernel (without host-to-device memcpy):\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_result[i]);  // likely garbage
    }
    printf("\n\nDump of device memory d_in (without initialization):\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_dump[i]);  // same as above
    }
    printf("\n");

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_dump);

    return 0;
}
