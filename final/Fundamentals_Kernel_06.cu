/*
Aim of the program:
Write a kernel that accepts an integer pointer `int* data` as an argument. The host code should allocate a single integer on the GPU (`cudaMalloc`), launch the kernel with one thread, and pass the pointer. The kernel should write the value `1337` to the memory location pointed to by `data`. Finally, the host should copy the value back (`cudaMemcpy`) and print it to verify.

Thinking:
1. The kernel is trivial: it receives an `int*` and writes the constant `1337` into the memory location it points to. Since we only need one write, a single thread is sufficient.
2. On the host side, we need to allocate device memory for one `int`, launch the kernel with a single thread, synchronize, copy the result back to host, print it, and finally free the device memory.
3. Basic CUDA error checking (e.g., checking return codes) can be added, but for this simple demonstration it can be omitted or done minimally.
4. The program uses the standard `cuda_runtime.h` header and `stdio.h` for printing.
5. Compile with `nvcc` to produce an executable.

Now the full .cu code:
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void writeValue(int *data)
{
    // Only one thread will execute this; write the constant value.
    *data = 1337;
}

int main()
{
    int *d_data = NULL;
    cudaError_t err;

    // Allocate space for one integer on the GPU.
    err = cudaMalloc((void**)&d_data, sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch the kernel with a single thread.
    writeValue<<<1, 1>>>(d_data);

    // Ensure kernel finished.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Copy the result back to host.
    int h_val = 0;
    err = cudaMemcpy(&h_val, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Print the value to verify.
    printf("Value on host after kernel execution: %d\n", h_val);

    // Free device memory.
    cudaFree(d_data);

    return 0;
}
