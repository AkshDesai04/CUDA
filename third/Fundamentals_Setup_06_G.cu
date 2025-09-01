/*
 * The difference between block dimensions and grid dimensions in CUDA:
 *
 * 1. Grid dimensions (gridDim):
 *    - The grid is the collection of thread blocks that execute a kernel.
 *    - gridDim specifies how many thread blocks are launched in each dimension (x, y, z).
 *    - It controls the overall number of blocks that will be dispatched by the kernel.
 *    - Each block in the grid is identified by its block index (blockIdx) which is derived from the grid dimensions.
 *
 * 2. Block dimensions (blockDim):
 *    - Within each block, threads are organized into a 1D, 2D, or 3D array.
 *    - blockDim specifies how many threads per block exist in each dimension (x, y, z).
 *    - It determines the size of each thread block and therefore the number of threads that can run concurrently within that block.
 *    - Each thread in a block is identified by its thread index (threadIdx) which is derived from the block dimensions.
 *
 * In summary:
 *    - gridDim → how many blocks (the global spread of work).
 *    - blockDim → how many threads per block (the local parallelism within each block).
 *    - Together, they define the total number of threads launched: totalThreads = gridDim.x * gridDim.y * gridDim.z
 *      * blockDim.x * blockDim.y * blockDim.z.
 */