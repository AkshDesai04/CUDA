/*
Aim: Does `cudaPeekAtLastError` right after the launch catch an in-kernel memory error? (No, the kernel hasn't run yet).

Thinking:
- `cudaPeekAtLastError` returns the last error that occurred during a previous asynchronous operation.
- Kernel launches are asynchronous: the launch returns immediately, and the kernel starts executing on the device later.
- Therefore, an error that occurs inside the kernel (such as an out-of-bounds memory access) will not be reported by `cudaPeekAtLastError` immediately after the launch, because the kernel hasn't executed yet.
- The error will be reported only after a synchronization point (`cudaDeviceSynchronize`, `cudaMemcpy`, etc.) when the device has finished executing the kernel and the runtime can report errors.
- The program below demonstrates this: a kernel deliberately writes out of bounds, the error is not caught by `cudaPeekAtLastError` right after launch, but is caught after synchronizing.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void out_of_bounds_kernel(int *data, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Intentionally write outside the bounds when idx >= N
    if (idx < N + 10) { // allow 10 out-of-bounds writes
        data[idx] = idx; // This will trigger a memory error when idx >= N
    }
}

int check_and_print(const char *msg, cudaError_t err)
{
    if (err != cudaSuccess) {
        printf("%s: %s\n", msg, cudaGetErrorString(err));
    } else {
        printf("%s: %s\n", msg, cudaGetErrorString(err));
    }
    return err;
}

int main(void)
{
    const int N = 10;
    int *d_array;
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc((void**)&d_array, N * sizeof(int));
    check_and_print("cudaMalloc", err);
    if (err != cudaSuccess) return 1;

    // Launch kernel that writes out of bounds
    out_of_bounds_kernel<<<1, N>>>(d_array, N);

    // Check error immediately after launch
    err = cudaPeekAtLastError();
    printf("After kernel launch, cudaPeekAtLastError: %s\n", cudaGetErrorString(err));

    // Now synchronize to force kernel execution
    err = cudaDeviceSynchronize();
    printf("After cudaDeviceSynchronize, cudaGetLastError: %s\n", cudaGetErrorString(cudaGetLastError()));

    // Clean up
    cudaFree(d_array);
    return 0;
}
