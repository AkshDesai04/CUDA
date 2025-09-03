/*
Launch an 8x8 block. Have only the threads on the first row (`threadIdx.y == 0`) print their `threadIdx.x`.

Thinking process:
- The program needs a simple CUDA kernel that runs on a single block of size 8×8.
- Inside the kernel we want only the threads where the y‑coordinate of the thread index is zero to emit a message.
- We use `printf` from device code; this requires the device to support printf (most recent GPUs do).
- In the host code we launch the kernel with a single block and the specified block dimensions.
- After launching we synchronize the device so that all prints are flushed before the program exits.
- No extra includes or error checks are necessary for this minimal example, but `cudaDeviceSynchronize` is called to ensure correctness.
*/

#include <stdio.h>

__global__ void print_first_row(void)
{
    if (threadIdx.y == 0) {
        printf("threadIdx.x = %d\n", threadIdx.x);
    }
}

int main(void)
{
    dim3 blockDim(8, 8);          // 8x8 block
    dim3 gridDim(1, 1);           // single block

    print_first_row<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();      // wait for kernel to finish

    return 0;
}
