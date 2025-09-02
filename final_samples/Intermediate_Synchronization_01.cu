```cpp
/* 
Aim of the program: 
Implement a parallel sum reduction for a large array that does not fit in a single block. The first kernel launch reduces each block's segment of the array into a partial sum (using the shared memory technique). A second, smaller kernel launch then sums up these partial results. 

Thinking and design notes:
1. We allocate a large array of floats on the host and fill it with some data (here a simple sequence). The array size is chosen to be larger than what can be processed by a single CUDA block (e.g. 16 million elements).
2. The first kernel (`reduce_kernel`) performs a block-wise reduction:
   - Each thread loads one element from global memory (with bounds checking) into a shared memory array.
   - A typical pairwise reduction is performed inside the block using `__syncthreads()` after each step.
   - Thread 0 of each block writes the block's partial sum to an output array (`blockSums`).
3. After the first launch, we have one partial sum per block. This array may still be too large for a single block, so we perform a second reduction phase:
   - We launch the same `reduce_kernel` again but this time on the `blockSums` array.
   - We use a loop on the host that repeatedly calls the reduction kernel until only one element remains (the final sum). Each iteration reduces the number of elements by roughly a factor of the block size.
4. The host code copies the final sum back to host memory and verifies it against a CPU-computed reference sum.
5. Error checking macros are used after CUDA API calls to catch any runtime errors.
6. The kernel uses shared memory sized at `blockDim.x * sizeof(float)`. The grid dimensions are calculated based on the size of the input array and the chosen block size.
7. For simplicity and clarity, the kernel does not employ warp‑level intrinsics; it uses a straightforward recursive halving method which is easy to understand and still efficient enough for demonstration purposes.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cassert>

// Error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__     \
                      << " : " << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Reduction kernel: reduces an input array into an output array of block sums
__global__ void reduce_kernel(const float *input, float *output, size_t N)
{
    extern __shared__ float sdata[];        // shared memory buffer
    unsigned int tid   = threadIdx.x;       // thread index within block
    unsigned int idx   = blockIdx.x * blockDim.x + tid;  // global index

    // Load one element per thread (with bounds check)
    float sum = 0.0f;
    if (idx < N) sum = input[idx];
    sdata[tid] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    // Iterate halving the active thread count each step
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block's partial sum to global memory
    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

int main()
{
    // Size of the large array (must be > max number of threads in a single block)
    const size_t N = 1 << 24;   // 16,777,216 elements (~64 MB)
    const size_t bytes = N * sizeof(float);

    // Host allocation
    float *h_input = (float *)malloc(bytes);
    assert(h_input != nullptr);

    // Initialize input array with a simple pattern (e.g., 1.0f)
    for (size_t i = 0; i < N; ++i)
    {
        h_input[i] = 1.0f;  // each element is 1.0, so the sum should be N
    }

    // Device allocation
    float *d_input = nullptr;
    float *d_blockSums = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_input, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Parameters for the first kernel launch
    const unsigned int threadsPerBlock = 256;
    const unsigned int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate array for partial sums (one per block)
    CUDA_CHECK(cudaMalloc((void **)&d_blockSums, blocksPerGrid * sizeof(float)));

    // First reduction: compute block sums
    reduce_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_input, d_blockSums, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Second reduction phase: reduce block sums until only one element remains
    size_t currentSize = blocksPerGrid;
    while (currentSize > 1)
    {
        unsigned int threads = (currentSize < threadsPerBlock) ? currentSize : threadsPerBlock;
        unsigned int blocks  = (currentSize + threads - 1) / threads;

        // Note: we reuse d_blockSums as both input and output
        reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(
            d_blockSums, d_blockSums, currentSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        currentSize = blocks;
    }

    // Copy final sum back to host
    float h_result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_result, d_blockSums, sizeof(float), cudaMemcpyDeviceToHost));

    // Compute reference sum on host
    float host_ref = 0.0f;
    for (size_t i = 0; i < N; ++i)
    {
        host_ref += h_input[i];
    }

    std::cout << "GPU sum = " << h_result << std::endl;
    std::cout << "CPU sum = " << host_ref << std::endl;
    std::cout << "Difference = " << fabs(h_result - host_ref) << std::endl;

    // Clean up
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_blockSums));
    free(h_input);

    return 0;
}
```