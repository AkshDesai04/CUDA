```cuda
/* 
 * Aim of the program (word for word from the prompt):
 * Implement a single-kernel sum reduction using a grid group sync. 
 * A first phase has all threads reduce data into shared memory. 
 * After a block sync, thread 0 of each block atomically adds its partial sum to a global total. 
 * After a `grid.sync()`, thread 0 of block 0 reads the final total.
 *
 * --------------------------------------------------------------
 * Thinking:
 *
 * 1. The problem asks for a reduction kernel that uses a grid group sync.
 *    This implies we need to use the CUDA cooperative groups API to
 *    synchronize all blocks after the per‑block reduction.
 *
 * 2. The data to reduce is stored in a global array.  Each thread
 *    loads one element into shared memory and then we perform a
 *    standard tree‑based reduction within the block using __syncthreads()
 *    after each step.  After the reduction, thread 0 of the block
 *    has the partial sum for that block.
 *
 * 3. Thread 0 of each block then atomically adds its partial sum to a
 *    global total that lives in device memory.  We use atomicAdd on an
 *    `int` variable because the data type is an integer sum.
 *
 * 4. After all blocks have finished adding to the global total we need
 *    a synchronization point that covers the whole grid.  CUDA provides
 *    `cooperative_groups::grid_group` with its `sync()` method.  This
 *    requires launching the kernel with `cudaLaunchCooperativeKernel`
 *    and that all blocks use the same number of threads and that the
 *    device supports cooperative launches.
 *
 * 5. Once the grid sync is complete, thread 0 of block 0 can read the
 *    final total from global memory and store it into a location
 *    visible to the host.
 *
 * 6. The host code generates a test array of integers, copies it to
 *    device memory, sets up a global total variable initialized to zero,
 *    launches the kernel cooperatively, and copies back the final sum
 *    to the host for verification.
 *
 * 7. Error handling macros are used to keep the code concise.
 *
 * 8. The code is self‑contained and can be compiled with:
 *       nvcc -o sum_reduce sum_reduce.cu --cooperative-launch
 *
 * 9. The kernel only uses one grid and one block dimension; the grid
 *    dimension is chosen so that there is at least one thread per
 *    element.  The shared memory size is the block size.
 *
 * 10. Because cooperative launches require the device to support
 *     them, the code checks `cudaDeviceGetAttribute` for
 *     `cudaDevAttrCooperativeLaunch`.  If unsupported, the program
 *     exits with an informative message.
 *
 * 11. The kernel is written to be generic for any array size; if the
 *     number of elements is not a multiple of the block size we
 *     handle out‑of‑bounds accesses gracefully.
 *
 * 12. Finally, we verify that the computed sum matches the sum
 *     performed on the host as a sanity check.
 *
 * --------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

// Error checking macro
#define CUDA_CHECK(call)                                                  \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// Kernel performing reduction with grid group sync
__global__ void sum_reduce_kernel(const int *data, int *global_total, int N)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    int value = 0;
    if (gid < N) {
        value = data[gid];
    }
    sdata[tid] = value;
    __syncthreads();

    // Tree-based reduction within block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 of each block adds its partial sum to global total
    if (tid == 0) {
        atomicAdd(global_total, sdata[0]);
    }

    // Synchronize all blocks
    grid_group grid = this_grid();
    grid.sync();

    // Thread 0 of block 0 reads the final total
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // For demonstration, we could store the result elsewhere if needed
        // Here we simply leave it in global_total
    }
}

int main()
{
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(int);

    // Allocate and initialize host array
    int *h_data = (int *)malloc(bytes);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_data[i] = 1; // simple test: all ones, expected sum = N
    }

    // Allocate device memory
    int *d_data;
    CUDA_CHECK(cudaMalloc((void **)&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Allocate global total on device
    int *d_total;
    CUDA_CHECK(cudaMalloc((void **)&d_total, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_total, 0, sizeof(int)));

    // Determine block and grid sizes
    const int BLOCK_SIZE = 256;
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Check for cooperative launch support
    int canCoop = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&canCoop,
                                      cudaDevAttrCooperativeLaunch,
                                      0));
    if (!canCoop) {
        fprintf(stderr, "Device does not support cooperative launch.\n");
        return EXIT_FAILURE;
    }

    // Launch kernel cooperatively
    void *kernelArgs[] = { &d_data, &d_total, &N };
    CUDA_CHECK(cudaLaunchCooperativeKernel((void *)sum_reduce_kernel,
                                           dim3(grid_size),
                                           dim3(BLOCK_SIZE),
                                           kernelArgs,
                                           BLOCK_SIZE * sizeof(int),
                                           0));

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    int h_total = 0;
    CUDA_CHECK(cudaMemcpy(&h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify
    long long expected = 0;
    for (int i = 0; i < N; ++i) expected += h_data[i];
    if (h_total == expected) {
        printf("Success! Sum = %d (expected %lld)\n", h_total, expected);
    } else {
        printf("Mismatch! Sum = %d, expected %lld\n", h_total, expected);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_total));
    free(h_data);

    return 0;
}
```