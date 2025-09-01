/*
Modify the program to allocate space for 10 integers. Have the kernel write `1337` to the 5th element (`data[4]`). Verify on the host.

My thinking:
- The aim is straightforward: allocate 10 integers on the device, launch a kernel that writes the constant 1337 to the 5th element (index 4), copy the data back, and check that the host sees the expected value.
- I choose a very simple kernel with one thread since only one element needs to be written.
- Use cudaMalloc to allocate device memory, cudaMemcpy to transfer data back, and basic error checking with cudaGetLastError.
- After copying back, the host will print the array and confirm that data[4] equals 1337.
- The code is self‑contained and can be compiled with `nvcc` to produce an executable that demonstrates the requested behavior.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that writes 1337 to data[4]
__global__ void write_value(int *data) {
    // Only thread 0 needs to write; other threads can do nothing
    if (threadIdx.x == 0) {
        data[4] = 1337;
    }
}

int main(void) {
    const int N = 10;               // Number of integers
    const int bytes = N * sizeof(int);
    int *d_data = NULL;
    int h_data[N];

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_data, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel with 1 block of 1 thread
    write_value<<<1, 1>>>(d_data);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Copy back to host
    err = cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Verify the value on the host
    if (h_data[4] == 1337) {
        printf("Success: data[4] = %d\n", h_data[4]);
    } else {
        printf("Failure: data[4] = %d (expected 1337)\n", h_data[4]);
    }

    // Optional: print entire array
    printf("Array contents:\n");
    for (int i = 0; i < N; ++i) {
        printf("data[%d] = %d\n", i, h_data[i]);
    }

    // Clean up
    cudaFree(d_data);
    return 0;
}