/*
Aim of the program: Write a kernel that accepts an integer pointer `int* data` as an argument. The host code should allocate a single integer on the GPU (`cudaMalloc`), launch the kernel with one thread, and pass the pointer. The kernel should write the value `1337` to the memory location pointed to by `data`. Finally, the host should copy the value back (`cudaMemcpy`) and print it to verify.

Thinking process:
1. The kernel is trivial: it just dereferences the pointer and assigns 1337.
2. Host side:
   a. Allocate device memory for a single int.
   b. Launch the kernel with a single block and a single thread.
   c. Synchronize to ensure kernel completion.
   d. Copy the integer back to host memory.
   e. Print the value to confirm correctness.
3. Include basic error checking after each CUDA API call to aid debugging.
4. Use `__global__` for the kernel definition and `cudaMalloc`, `cudaMemcpy`, etc. for memory operations.
5. Provide a clean exit by freeing device memory with `cudaFree`.
6. Use `printf` to display the retrieved value.

The final code below implements all of the above steps in a single `.cu` file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that writes 1337 to the memory location pointed to by data
__global__ void write1337(int *data) {
    *data = 1337;
}

int main() {
    int *d_val = NULL;
    int h_val = 0;
    cudaError_t err;

    // Allocate one integer on the GPU
    err = cudaMalloc((void**)&d_val, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel with 1 block and 1 thread
    write1337<<<1, 1>>>(d_val);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_val);
        return 1;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_val);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(&h_val, d_val, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_val);
        return 1;
    }

    // Print the value to verify
    printf("Value copied back from GPU: %d\n", h_val);

    // Clean up
    cudaFree(d_val);
    return 0;
}
