/*
 * Aim of the program: Explain in a comment what a grid of blocks is in the CUDA execution model.
 *
 * Thinking:
 * The CUDA execution model is organized hierarchically: threads belong to blocks, and blocks belong to a grid.
 * A grid is a collection of thread blocks that launch together. Each block is identified by a unique blockIdx.
 * Blocks can be arranged in one, two, or three dimensions (gridDim.x, gridDim.y, gridDim.z).
 * Within each block, threads are identified by threadIdx and can be arranged in up to three dimensions as well.
 * The grid size specifies how many blocks are launched, and the block size specifies how many threads per block.
 * The grid of blocks provides a way to scale computation across the GPU by distributing work among many thread blocks,
 * which the GPU scheduler maps onto streaming multiprocessors (SMs). Blocks are independent in terms of synchronization
 * (except via global memory), so they can execute concurrently or in any order.
 * This comment explains the concept; the rest of the code is a minimal CUDA program that compiles but does not perform
 * any computation. It simply demonstrates the structure of a CUDA .cu file and includes a dummy kernel.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Dummy kernel that does nothing */
__global__ void dummyKernel() {
    // No operation
}

int main() {
    // Launch one block of one thread
    dummyKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    printf("Dummy kernel launched.\n");
    return 0;
}
