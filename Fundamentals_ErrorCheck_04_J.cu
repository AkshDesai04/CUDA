/*
Pass a size of 0 to `cudaMemcpy`. Is this an error?

Thinking: In CUDA programming, `cudaMemcpy` copies a specified number of bytes between host and device memory. 
The size argument is of type `size_t`. According to the CUDA Runtime API documentation, a `size` of zero is 
legitimate: it is equivalent to a no‑op copy and should return `cudaSuccess`. The operation will not touch any 
memory and is essentially a no‑op, but it can still be used for consistency or to avoid branching on the host side.

To confirm this behaviour, the following program allocates a small array on both host and device, performs a
`cudaMemcpy` with a size of 0, and checks the return status. It also performs a normal copy with a non‑zero
size to verify that data transfer works as expected. After each copy, the program checks for errors and prints
the results. The program demonstrates that a zero‑length copy does not produce an error and returns
`cudaSuccess`. This is useful when writing generic code that might sometimes need to copy zero bytes.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void identity_kernel(int *d_out, const int *d_in, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_out[idx] = d_in[idx];
}

int checkCudaError(const char *msg, cudaError_t err)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: %s : %s\n", msg, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int main(void)
{
    const int N = 10;
    const size_t bytes = N * sizeof(int);

    int h_in[N];
    int h_out[N];

    // Initialize host input array
    for (int i = 0; i < N; ++i)
        h_in[i] = i;

    // Allocate device memory
    int *d_in = nullptr;
    int *d_out = nullptr;
    cudaError_t err;

    err = cudaMalloc((void**)&d_in, bytes);
    if (checkCudaError("cudaMalloc d_in", err)) return 1;

    err = cudaMalloc((void**)&d_out, bytes);
    if (checkCudaError("cudaMalloc d_out", err)) return 1;

    // 1. Perform a zero‑length copy from host to device
    err = cudaMemcpy(d_in, h_in, 0, cudaMemcpyHostToDevice);
    if (checkCudaError("cudaMemcpy (size=0) HostToDevice", err))
        return 1;
    printf("Zero‑length cudaMemcpy (HostToDevice) succeeded.\n");

    // 2. Perform a normal copy from host to device
    err = cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    if (checkCudaError("cudaMemcpy (normal) HostToDevice", err))
        return 1;
    printf("Normal cudaMemcpy (HostToDevice) succeeded.\n");

    // 3. Launch kernel to copy data from d_in to d_out
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    identity_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, N);
    err = cudaGetLastError();
    if (checkCudaError("Kernel launch", err))
        return 1;

    // 4. Copy result back to host
    err = cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    if (checkCudaError("cudaMemcpy DeviceToHost", err))
        return 1;

    // Verify the results
    int success = 1;
    for (int i = 0; i < N; ++i)
    {
        if (h_out[i] != h_in[i])
        {
            printf("Mismatch at index %d: expected %d, got %d\n", i, h_in[i], h_out[i]);
            success = 0;
            break;
        }
    }
    if (success)
        printf("Kernel copy verified successfully.\n");

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
