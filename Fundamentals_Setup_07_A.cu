/*
 * Aim: Explain in a comment what a grid of blocks is in the CUDA execution model.
 *
 * Thinking:
 * The CUDA execution model organizes parallel work into a hierarchy:
 *
 * 1. **Threads** – the smallest unit of execution.  Each thread has its own
 *    register file and executes the kernel code independently.
 *
 * 2. **Blocks** – a group of threads that execute together on a single
 *    streaming multiprocessor (SM).  Threads within a block can cooperate
 *    via fast shared memory and can be synchronised with __syncthreads().
 *    The number of threads per block is limited (e.g., 1024 on many GPUs).
 *
 * 3. **Grid** – a collection of blocks that together comprise the entire
 *    kernel launch.  All blocks in a grid share the same kernel function
 *    and can access global device memory.  Blocks are independent of each
 *    other (no direct communication or synchronisation between blocks),
 *    but the grid as a whole can be traversed using blockIdx, which
 *    gives each block a unique index within the grid.
 *
 * A kernel launch is therefore specified by:
 *   - **gridDim**   – the number of blocks in the grid (1D, 2D, or 3D).
 *   - **blockDim**  – the number of threads per block (1D, 2D, or 3D).
 *
 * Example: `kernel<<<dim3(4, 1, 1), dim3(256, 1, 1)>>>();`
 * launches a grid of 4 blocks, each containing 256 threads.
 *
 * The grid of blocks forms the high‑level parallel structure that the GPU
 * scheduler maps onto its hardware resources, allowing massive data‑parallel
 * execution while keeping thread groups (blocks) logically coherent.
 */

#include <stdio.h>

int main(void) {
    printf("CUDA grid of blocks example.\n");
    return 0;
}
