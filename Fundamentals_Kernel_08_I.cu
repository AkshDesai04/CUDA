/*
Aim: Modify the kernel so that blocks with `blockIdx.x == blockIdx.y` (on the diagonal) print a special message.

Thinking:
1. The requirement is to detect diagonal blocks in a 2‑D grid. In CUDA, `blockIdx.x` and `blockIdx.y` provide the block coordinates.
2. If they are equal, we print a special message; otherwise we print a normal message. 
3. Use `printf` inside the kernel. To avoid huge output, keep the grid small (e.g., 4x4) and the block size 1x1.
4. Host code will launch the kernel with `dim3 grid(4,4)` and `dim3 block(1,1)`.
5. After kernel launch, call `cudaDeviceSynchronize()` to ensure all prints complete before the program exits.
6. Compile with `nvcc` and run. The output will show diagonal messages highlighted.

Implementation details:
- Include <stdio.h> for `printf`.
- Define the kernel `diagKernel`.
- In the kernel, use an `if` statement to check `blockIdx.x == blockIdx.y`.
- Print a message accordingly.
- In `main`, set up grid and block dimensions, launch kernel, synchronize, and exit.

No external libraries or complex logic are needed – the focus is purely on the conditional printing in the kernel.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void diagKernel()
{
    if (blockIdx.x == blockIdx.y)
    {
        printf("Diagonal block (%d, %d) - special message!\n", blockIdx.x, blockIdx.y);
    }
    else
    {
        printf("Block (%d, %d) - normal.\n", blockIdx.x, blockIdx.y);
    }
}

int main()
{
    dim3 gridDim(4, 4);   // 4x4 grid of blocks
    dim3 blockDim(1, 1);  // 1 thread per block

    // Launch the kernel
    diagKernel<<<gridDim, blockDim>>>();

    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
