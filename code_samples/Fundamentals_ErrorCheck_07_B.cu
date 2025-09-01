/*
Provide a list of 3 common asynchronous CUDA API calls (`cudaMemcpyAsync`, kernel launch, etc.).

Thinking:
The user’s request is a simple list of asynchronous CUDA API calls, but the developer instructions require us to produce a .cu file that contains the entire code from start to end. Therefore, the aim of the program is exactly the text from the user’s prompt, which must appear verbatim in a multiline comment at the very top of the file. We also need to include all of our reasoning within that comment so that the user can see how we arrived at the implementation.

The code itself should be a minimal, self‑contained CUDA C program that demonstrates the three asynchronous calls mentioned in the aim:
1. `cudaMemcpyAsync` – used to copy data from host to device asynchronously.
2. Kernel launch – the kernel launch is already asynchronous by default when a CUDA stream is supplied.
3. Another `cudaMemcpyAsync` – copying results back from device to host asynchronously.

We will create a simple kernel that adds two vectors. The main function will:
- Allocate pinned host memory (so that `cudaMemcpyAsync` can be used).
- Allocate device memory.
- Create a CUDA stream.
- Asynchronously copy input arrays to the device.
- Launch the kernel in that stream.
- Asynchronously copy the result back to host.
- Synchronize the stream.
- Verify the result and clean up.

The program will compile with `nvcc` and run on any system with a CUDA capable GPU. No external dependencies are required beyond the standard CUDA headers.

Below is the complete `.cu` file with the comment and the implementation.*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// Simple vector addition kernel
__global__ void vecAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

int main(void)
{
    size_t size = N * sizeof(float);
    float *h_A, *h_B, *h_C;          // Host arrays
    float *d_A, *d_B, *d_C;          // Device arrays
    cudaStream_t stream;

    // Allocate pinned host memory for asynchronous copies
    cudaMallocHost((void **)&h_A, size);
    cudaMallocHost((void **)&h_B, size);
    cudaMallocHost((void **)&h_C, size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Create a CUDA stream
    cudaStreamCreate(&stream);

    // 1. Asynchronous copy from host to device
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

    // 2. Kernel launch (asynchronous by virtue of specifying the stream)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);

    // 3. Asynchronous copy from device to host
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);

    // Wait for all operations in the stream to complete
    cudaStreamSynchronize(stream);

    // Simple verification
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) {
            success = false;
            printf("Mismatch at index %d: %f != %f\n", i, h_C[i], h_A[i] + h_B[i]);
            break;
        }
    }
    if (success)
        printf("Vector addition successful. All values match.\n");
    else
        printf("Vector addition failed.\n");

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaStreamDestroy(stream);

    return 0;
}
