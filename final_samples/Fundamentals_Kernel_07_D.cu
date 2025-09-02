```c
/*
Aim: Launch a 3D block of (4, 4, 4) threads. Have each thread print its 3D thread index (threadIdx.x, threadIdx.y, threadIdx.z).

Thinking:
1. We need a CUDA kernel that will be executed by each thread in the block.
2. Since we want each thread to output its own indices, we will use the built-in printf function available in CUDA for device code.
3. The kernel will simply call printf with the threadIdx.x, threadIdx.y, and threadIdx.z values.
4. In the host code, we set up a 3D block dimension of (4,4,4) using dim3. The grid dimension can be (1,1,1) since we only need a single block.
5. We launch the kernel with the <<<grid, block>>> syntax.
6. After the kernel launch, we call cudaDeviceSynchronize() to ensure all printf outputs are flushed before the program exits.
7. The program includes necessary headers (#include <stdio.h>) and checks for errors is optional but omitted for brevity, as the focus is on printing thread indices.
8. The entire program is written in a single .cu file.

Note: Printing from many threads can interleave output, but for a small block size this is manageable.
*/

#include <stdio.h>

__global__ void print_thread_indices()
{
    printf("ThreadIdx: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
    dim3 block(4, 4, 4);
    dim3 grid(1, 1, 1);

    print_thread_indices<<<grid, block>>>();

    // Wait for the kernel to finish and ensure all printf output is flushed
    cudaDeviceSynchronize();

    return 0;
}
```