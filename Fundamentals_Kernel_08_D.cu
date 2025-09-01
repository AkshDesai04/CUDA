```cuda
/* 
Aim of the program (as passed in the prompt word to word):
"Launch a 3D grid of (2, 2, 2) blocks. Have each thread print its 3D block index (blockIdx.x, blockIdx.y, blockIdx.z)."

Thinking:

1. **Objective**  
   We need to launch a CUDA kernel with a 3‑dimensional grid where each dimension has size 2, resulting in 8 blocks in total.  
   Every block will contain at least one thread that prints the block’s 3‑D indices using the built‑in `printf` function on the device.

2. **Choosing the Block Size**  
   Since the task is simply to output the block indices, the workload per block is trivial.  
   A single thread per block (`threadsPerBlock = 1`) is sufficient.  
   Using a larger number of threads would require extra synchronization or duplication of the same print statement, which is unnecessary.

3. **Kernel Implementation**  
   ```cpp
   __global__ void printBlockIdx() {
       printf("BlockIdx: (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
   }
   ```  
   The `printf` function is a device function available in CUDA ≥ 2.0, and it writes to the host console after the kernel completes.  
   No arguments from `threadIdx` or `blockDim` are needed.

4. **Launching the Kernel**  
   ```cpp
   dim3 grid(2, 2, 2);
   dim3 block(1, 1, 1);
   printBlockIdx<<<grid, block>>>();
   ```  
   After the launch, we call `cudaDeviceSynchronize()` to ensure the kernel has finished and the output has been flushed to the console.

5. **Error Checking**  
   Basic error checking is included after the kernel launch and after the synchronization to catch any launch or runtime errors.

6. **Compilation**  
   This file is a self‑contained CUDA source (`.cu`). Compile with:
   ```bash
   nvcc -o print_block_idx print_block_idx.cu
   ./print_block_idx
   ```
   The program will print 8 lines, one per block, showing the indices.

7. **Edge Cases**  
   - `printf` output order is not strictly guaranteed because threads execute concurrently.  
   - In larger grids, the output may intermix; however, with a single thread per block, the order is typically by launch order.

With this plan, the code below follows the outlined steps and satisfies the prompt’s requirements.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the 3D block index
__global__ void printBlockIdx()
{
    // Each thread (there is only one per block) prints its block's indices
    printf("BlockIdx: (%d, %d, %d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z);
}

int main()
{
    // Define a 3D grid of size 2x2x2
    dim3 gridDim(2, 2, 2);
    // One thread per block is sufficient for this task
    dim3 blockDim(1, 1, 1);

    // Launch the kernel
    printBlockIdx<<<gridDim, blockDim>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA synchronization failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```