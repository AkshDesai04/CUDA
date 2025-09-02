```cuda
/*
Aim of the program: To use cooperative groups for grid-wide sync, the kernel must be launched with cudaLaunchCooperativeKernel. Adapt your launch code.

Thinking:
- The goal is to demonstrate grid-wide synchronization using cooperative groups. 
- We need a kernel that reduces an array across the entire grid, requiring synchronization of all blocks. 
- Cooperative launch is required to enable grid-wide sync, which is achieved by launching the kernel with cudaLaunchCooperativeKernel. 
- We allocate an array of floats, copy it to the device, launch a cooperative kernel that performs a block-wise reduction, then a grid-wide reduction using grid.sync() from cooperative_groups. 
- The kernel writes the block sums to a device array; block 0 then reduces these into a single total sum. 
- The host checks for errors, queries device capabilities for cooperative launch, and ensures the grid size is within limits. 
- We also perform simple error checking after each CUDA call and free all resources before exiting.
*/

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>

using namespace cooperative_groups;

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

// Kernel that performs a reduction across the entire grid
__global__ void sumKernel(const float *input, float *blockSums, int N) {
    // Shared memory for block reduction
    __shared__ float sh[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load element
    float val = 0.0f;
    if (idx < N) val = input[idx];
    sh[tid] = val;
    __syncthreads();

    // Intra-block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh[tid] += sh[tid + stride];
        }
        __syncthreads();
    }

    // Write block sum
    if (tid == 0) blockSums[blockIdx.x] = sh[0];

    // Grid-wide synchronization
    grid_group grid = this_grid();
    grid.sync();

    // Block 0 performs final reduction of block sums
    if (blockIdx.x == 0) {
        // Load block sums into shared memory
        if (tid < gridDim.x) {
            sh[tid] = blockSums[tid];
        } else {
            sh[tid] = 0.0f;
        }
        __syncthreads();

        // Reduction of block sums
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sh[tid] += sh[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) blockSums[0] = sh[0];
    }
}

int main() {
    // Problem size
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Host allocation and initialization
    float *h_in = (float*)malloc(size);
    if (!h_in) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f; // all ones

    // Device allocation
    float *d_in = nullptr;
    float *d_blockSums = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, size));
    CHECK_CUDA(cudaMalloc(&d_blockSums, size)); // allocate temporary, will resize later

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Query device for cooperative launch capability
    int dev = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&dev));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    if (!prop.cooperativeLaunch) {
        fprintf(stderr, "Device does not support cooperative launch.\n");
        return EXIT_FAILURE;
    }

    // Ensure grid size does not exceed maximum cooperative grid size
    int maxGridSize[3] = { prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] };
    if (gridSize.x > maxGridSize[0]) gridSize.x = maxGridSize[0];

    // Resize block sums buffer to grid size
    CHECK_CUDA(cudaFree(d_blockSums));
    CHECK_CUDA(cudaMalloc(&d_blockSums, gridSize.x * sizeof(float)));

    // Prepare kernel launch arguments
    void* args[] = { &d_in, &d_blockSums, &N };

    // Launch the kernel cooperatively
    cudaError_t err = cudaLaunchCooperativeKernel((void*)sumKernel, gridSize, blockSize, args, 0, nullptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaLaunchCooperativeKernel failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for kernel to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Retrieve result
    float h_sum = 0.0f;
    CHECK_CUDA(cudaMemcpy(&h_sum, d_blockSums, sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result (should be N * 1.0f)
    printf("Computed sum: %f\n", h_sum);
    printf("Expected sum: %f\n", (float)N);

    // Clean up
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_blockSums));
    free(h_in);

    return 0;
}
```