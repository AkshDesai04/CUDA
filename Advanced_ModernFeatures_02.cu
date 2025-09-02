```cuda
/*
Write a program that uses Cooperative Groups to perform a grid-wide synchronization.
Launch a kernel and get the `grid_group`. Use the group's `sync()` method to ensure all
threads in the entire grid have reached a certain point before proceeding. This
allows for single-kernel reductions on large arrays.

=== Thinking ===
The task is to demonstrate grid-wide synchronization using CUDA Cooperative Groups.
Grid-wide sync is only valid for cooperative kernels launched with
`cudaLaunchCooperativeKernel`. The program will perform a reduction (sum) of a large
array entirely within one kernel launch.

Key points:
1. Include the cooperative groups header and use the `cooperative_groups` namespace.
2. Inside the kernel:
   - Each thread loads one element from the global array and performs a block‑level
     reduction using shared memory.
   - After the block reduction, each block writes its partial sum to a global array.
   - Use `grid.sync()` to ensure all blocks have written their partial sums before
     the final aggregation step.
   - The thread (0,0) of block 0 will perform the final reduction of the partial sums
     into the final result.
3. Host side:
   - Allocate device memory for input, partial sums, and final result.
   - Check if the device supports cooperative launches.
   - Configure grid and block dimensions.
   - Prepare the argument array for `cudaLaunchCooperativeKernel`.
   - Launch the kernel cooperatively.
   - Copy back the final result and verify correctness.
4. Error checking is added via a helper macro.
5. The example uses a 1D grid and 1D blocks for simplicity.
6. The shared memory size is computed at launch and passed to the kernel.

This program compiles with `nvcc -arch=sm_70 -lcuda test.cu` (assuming the GPU
supports cooperative launches).
*/

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                      << " code=" << static_cast<int>(err)                 \
                      << " \"" << cudaGetErrorString(err) << "\"\n";        \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel that performs a single-kernel reduction using grid-wide sync
__global__ void reduceKernel(const float* __restrict__ d_in,
                             float* __restrict__ d_out,
                             float* __restrict__ d_partial,
                             int N)
{
    // Shared memory for block reduction
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one element (or 0 if out of bounds)
    float val = 0.0f;
    if (idx < N) val = d_in[idx];

    // Store into shared memory
    sdata[tid] = val;
    __syncthreads();

    // Block reduction (binary tree)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes its partial sum
    if (tid == 0) {
        d_partial[blockIdx.x] = sdata[0];
    }

    // Grid-wide synchronization to make sure all partial sums are ready
    cg::grid_group grid = cg::this_grid();
    grid.sync();

    // Now thread 0 of block 0 performs the final reduction
    if (blockIdx.x == 0 && tid == 0) {
        float sum = 0.0f;
        int numBlocks = gridDim.x;
        for (int i = 0; i < numBlocks; ++i) {
            sum += d_partial[i];
        }
        d_out[0] = sum;
    }
}

int main()
{
    const int N = 1 << 20;                 // Size of input array (1M elements)
    const int threadsPerBlock = 256;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Host memory allocation
    float *h_in = new float[N];
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;  // Simple test: sum should be N

    // Device memory allocation
    float *d_in, *d_out, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, numBlocks * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Check if device supports cooperative launches
    int canCooperate = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&canCooperate,
                                      cudaDevAttrCooperativeLaunch, 0));
    if (!canCooperate) {
        std::cerr << "Device does not support cooperative launches.\n";
        return EXIT_FAILURE;
    }

    // Kernel launch configuration
    dim3 gridDim(numBlocks);
    dim3 blockDim(threadsPerBlock);
    size_t sharedMemBytes = threadsPerBlock * sizeof(float);

    // Prepare arguments for cudaLaunchCooperativeKernel
    void *args[] = { &d_in, &d_out, &d_partial, &N };

    // Launch the kernel cooperatively
    cudaError_t err = cudaLaunchCooperativeKernel((void*)reduceKernel,
                                                  gridDim,
                                                  blockDim,
                                                  args,
                                                  sharedMemBytes,
                                                  0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch cooperative kernel: "
                  << cudaGetErrorString(err) << "\n";
        return EXIT_FAILURE;
    }

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    float h_out = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result
    std::cout << "Sum of array = " << h_out << " (expected " << static_cast<float>(N) << ")\n";

    // Clean up
    delete[] h_in;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_partial));

    return 0;
}
```