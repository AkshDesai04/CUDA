/*
Aim: Explain why this 'silent failure' is one of the most dangerous types of bugs in parallel programming.

Thinking:
- Parallel programs rely on multiple threads executing simultaneously.
- A silent failure occurs when a bug causes incorrect behavior or results without raising an error flag or providing a clear diagnostic.
- In CUDA, many errors (e.g., out‑of‑bounds memory access, race conditions, illegal instructions) may set the error state but the host code might not check it, so the program continues as if everything is fine.
- This leads to unpredictable results, data corruption, or subtle incorrectness that can be extremely hard to trace because the code appears to run correctly.
- Silent failures are dangerous because:
  * They can propagate wrong data through a computation pipeline.
  * They may only manifest under certain input sizes, timing, or hardware configurations.
  * Debugging is difficult because the failure point is not reported.
- The example below intentionally introduces a data race in a kernel and omits error checking on the host side, demonstrating how a seemingly innocent program can silently produce an incorrect result.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel that has a data race: each thread increments the same memory location without an atomic operation. */
__global__ void raceKernel(int *value, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        // Each thread attempts to add 1 to *value without synchronization.
        // This causes a race condition leading to an unpredictable final value.
        *value += 1;
    }
}

int main()
{
    const int N = 1024;                 // Number of threads that will try to increment the value
    int *d_val;
    int h_val = 0;

    // Allocate and initialize device memory
    cudaMalloc(&d_val, sizeof(int));
    cudaMemset(d_val, 0, sizeof(int));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    raceKernel<<<blocks, threadsPerBlock>>>(d_val, N);

    // Intentionally do NOT call cudaGetLastError() or cudaDeviceSynchronize()
    // This omission exemplifies silent failure: any launch or execution errors are ignored.

    // Copy result back to host
    cudaMemcpy(&h_val, d_val, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result. The expected value is N, but due to the race condition,
    // the actual result may be lower and will not trigger an error.
    printf("Result after kernel execution: %d (expected %d)\n", h_val, N);

    // Clean up
    cudaFree(d_val);
    return 0;
}
