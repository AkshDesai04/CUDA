```cpp
/*
Aim of the program:
Test your program with `N = 1` and a block size of 256. Does it work correctly?

Thinking:
The goal is to demonstrate that a CUDA kernel can correctly handle a case where the
number of threads launched (block size * number of blocks) exceeds the amount of
work (the size of the input array).  For this test we choose the simplest possible
work: summing an array of ones of length N=1.  We launch a single block of 256
threads.  Each thread checks whether its global index is within bounds; if it
is, it atomically adds its element to a global sum.

Key points considered:
1. **Memory allocation** – we allocate host memory for an array of size N and
   a device pointer for the same data.  The device sum is stored in global
   device memory and initialized to zero before the kernel launch.

2. **Kernel launch configuration** – `blocks = 1`, `threadsPerBlock = 256`.  The
   kernel uses the standard pattern of `int idx = blockIdx.x * blockDim.x + threadIdx.x;`
   and proceeds only if `idx < N`.

3. **Atomic addition** – `atomicAdd` is used to accumulate the sum safely
   across multiple threads.  This is essential even when many threads are idle
   because they do not participate; the atomic ensures no race conditions.

4. **Boundary handling** – When N < blockDim.x, the threads with idx >= N
   simply return without touching memory, so the kernel works correctly.

5. **Verification** – After the kernel finishes, we copy the sum back to the
   host and print it.  For N=1, the expected output is 1.  If the program
   prints 1, it indicates correct behavior.

Potential pitfalls:
- Forgetting to check the result of CUDA API calls can hide errors.  For this
  educational example we keep error checking minimal but include basic checks.
- Using `atomicAdd` on non-atomic-friendly data types could lead to undefined
  behavior; we use `int` which is supported.

Running this program should print `Sum = 1` confirming that the kernel works
correctly even when the block size is larger than the problem size.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sumKernel(const int *data, int *sum, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Use atomicAdd to safely accumulate into global sum
        atomicAdd(sum, data[idx]);
    }
}

int main(void)
{
    const int N = 1;              // Size of the array
    const int threadsPerBlock = 256; // Block size
    const int blocksPerGrid = 1;  // Single block

    // Allocate host memory
    int *h_data = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) h_data[i] = 1; // Fill with ones

    // Allocate device memory
    int *d_data = nullptr;
    int *d_sum = nullptr;
    cudaMalloc((void **)&d_data, N * sizeof(int));
    cudaMalloc((void **)&d_sum, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(int));

    // Launch kernel
    sumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_sum, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back to host
    int h_sum = 0;
    cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Sum = %d\n", h_sum);

    // Clean up
    cudaFree(d_data);
    cudaFree(d_sum);
    free(h_data);

    return 0;
}
```