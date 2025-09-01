/*
Implement a min-finding reduction.

Thinking:
The task is to find the minimum value in an array using CUDA. A typical way is to perform a reduction on the GPU: each block reads a chunk of the input array into shared memory, then performs a tree-style reduction to compute the minimum of that block, writing the block’s minimum to a temporary array. This temporary array can then be reduced again until a single minimum value remains. Because the array can be large, we iterate until we have only one value left, launching a new kernel each time with the size of the temporary array as the number of elements to reduce.

Key points to consider:
1. Choose a reasonable block size (e.g., 256 threads). Each thread will load one element from global memory into shared memory (if it exists). The reduction uses a classic two-phase approach: first a loop that halves the active threads each step, then optionally use warp-level primitives to finish within a warp, but for simplicity we’ll keep the loop.

2. The reduction kernel uses shared memory of size blockDim.x * sizeof(T). Each thread loads an element into shmem[i] (or a sentinel value if out of bounds). Then we iterate for stride = blockDim.x / 2 down to 1, synchronizing after each step. At each step, if thread id < stride, we compare shmem[tid] with shmem[tid+stride] and keep the smaller one.

3. After the loop, thread 0 writes shmem[0] to an output array. We need to pass the number of input elements to the kernel so it knows if a thread’s index is beyond the array.

4. The iterative reduction loop on the host: start with inputSize = N, input pointer = d_in. While inputSize > 1, compute number of blocks = ceil(inputSize / blockSize). Allocate d_out array of that size. Launch kernel with these parameters. Then swap input and output pointers, update inputSize to number of blocks. Repeat. After the loop, copy the single value back to host.

5. Error handling: wrap CUDA calls with a macro to check errors.

6. Testing: In main, allocate an array of N integers, fill with random values, call the reduction, print the result. Also verify correctness by computing min on host.

7. Clean up: free device and host memory.

The code will be a self-contained .cu file that can be compiled with nvcc. It will include necessary headers, error checking, the kernel, and a main function demonstrating usage.

Edge cases:
- If N is 0, we can return an appropriate value (e.g., INT_MAX) or handle specially.
- The reduction assumes that the data type supports a defined maximum value for initial sentinel. For int, we can use INT_MAX.
- Ensure that shared memory allocation is correct: the kernel’s shared memory size must be specified at launch or declared as extern __shared__.

Now with the plan in place, the actual code will be written accordingly.
*/

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <climits>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line "   \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// Kernel to perform block-level min reduction
template <typename T>
__global__ void reduce_min_kernel(const T* g_idata, T* g_odata, size_t n) {
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load element into shared memory (or max sentinel if out of bounds)
    if (idx < n) {
        sdata[tid] = g_idata[idx];
    } else {
        sdata[tid] = std::numeric_limits<T>::max();
    }

    __syncthreads();

    // Reduce within the block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] < sdata[tid]) {
                sdata[tid] = sdata[tid + stride];
            }
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// Host function to perform full reduction
template <typename T>
T reduce_min_on_device(const T* d_in, size_t n, int blockSize = 256) {
    if (n == 0) {
        return std::numeric_limits<T>::max();
    }

    const size_t sharedMemSize = blockSize * sizeof(T);
    size_t currSize = n;
    const T* d_curr = d_in;
    T* d_next = nullptr;

    while (currSize > 1) {
        size_t numBlocks = (currSize + blockSize - 1) / blockSize;
        CUDA_CHECK(cudaMalloc(&d_next, numBlocks * sizeof(T)));

        reduce_min_kernel<T><<<numBlocks, blockSize, sharedMemSize>>>(d_curr, d_next, currSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Free previous level
        CUDA_CHECK(cudaFree(const_cast<T*>(d_curr)));

        d_curr = d_next;
        currSize = numBlocks;
    }

    // Copy final result back to host
    T hostMin;
    CUDA_CHECK(cudaMemcpy(&hostMin, d_curr, sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(const_cast<T*>(d_curr)));

    return hostMin;
}

int main() {
    const size_t N = 1 << 20; // 1,048,576 elements
    const int blockSize = 256;

    // Allocate host memory and initialize
    int* h_in = new int[N];
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (size_t i = 0; i < N; ++i) {
        h_in[i] = std::rand();
    }

    // Compute min on host for verification
    int h_min = std::numeric_limits<int>::max();
    for (size_t i = 0; i < N; ++i) {
        if (h_in[i] < h_min) {
            h_min = h_in[i];
        }
    }

    // Allocate device memory and copy input
    int* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // Perform reduction on device
    int d_min = reduce_min_on_device<int>(d_in, N, blockSize);

    // Output results
    std::cout << "Minimum value found on host: " << h_min << std::endl;
    std::cout << "Minimum value found on device: " << d_min << std::endl;

    // Verify correctness
    if (h_min == d_min) {
        std::cout << "Result verified: HOST == DEVICE" << std::endl;
    } else {
        std::cout << "Result mismatch!" << std::endl;
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_in));
    delete[] h_in;

    return 0;
}
