/*
Integrate a CUB block-level primitive, like `BlockScan`, into your own custom kernel to perform a scan within a single thread block.

Thoughts & Design Decisions:
1. **Objective**: Demonstrate how to use CUB’s `BlockScan` to compute an exclusive prefix sum (scan) for an array that fits within a single thread block.
2. **Why BlockScan?**  
   - `BlockScan` is a highly optimized block‑level primitive that runs entirely on the GPU’s shared memory, achieving near‑the‑maximum memory bandwidth.  
   - It abstracts away the details of reduction and shuffle operations, letting the programmer focus on the algorithm.
3. **Input/Output Layout**  
   - The host allocates an array of `int`s (`h_in`) and copies it to device memory (`d_in`).  
   - The kernel writes the exclusive scan results into another device array (`d_out`).  
   - After kernel execution, results are copied back to host memory for display.
4. **Kernel Structure**  
   - Each thread loads one element from global memory into a local variable (`val`).  
   - We use a shared‑memory buffer `temp_storage` (provided by CUB) for the scan’s intermediate data.  
   - `cub::BlockScan` performs an **exclusive** sum; the result for thread `i` is the sum of all elements with index `< i`.  
   - Threads with `tid >= n` simply set their value to 0 to keep the scan consistent when the array size is less than the block size.
5. **Boundary Handling**  
   - The kernel checks `if (tid < n)` before writing back to global memory, ensuring we never write out of bounds.  
   - If the array size exceeds `BLOCK_THREADS`, the kernel would need to be extended to launch multiple blocks and combine partial results. For clarity, this example keeps it to a single block.
6. **Compilation Notes**  
   - Requires the CUB library (header‑only).  
   - Compile with: `nvcc -arch=sm_70 -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudadevrt block_scan.cu -o block_scan` (adjust paths and arch as necessary).
7. **Error Checking**  
   - Simple CUDA error checking macro `CHECK_CUDA` is used to abort on any failure.
8. **Testing**  
   - The program prints the input array and the exclusive scan output to verify correctness.

This code provides a minimal, self‑contained example of integrating CUB’s `BlockScan` into a custom kernel to perform a scan within a single thread block.
*/

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define CHECK_CUDA(call)                                                  \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " : " << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

const int BLOCK_THREADS = 256;

// Kernel that performs an exclusive scan within a single block
__global__ void block_scan_kernel(const int* __restrict__ d_in,
                                  int* __restrict__ d_out,
                                  int n) {
    // Each thread loads its element (or 0 if beyond n)
    int tid = threadIdx.x;
    int val = (tid < n) ? d_in[tid] : 0;

    // Shared memory buffer required by CUB block scan
    __shared__ typename cub::BlockScan<int, BLOCK_THREADS>::TempStorage temp_storage;

    // Create a BlockScan instance
    cub::BlockScan<int, BLOCK_THREADS> block_scan(temp_storage);

    int result;
    // Perform exclusive scan
    block_scan.ExclusiveSum(val, result);

    // Optional: barrier to ensure all threads have finished the scan
    __syncthreads();

    // Write results back to global memory
    if (tid < n) {
        d_out[tid] = result;
    }
}

int main() {
    const int N = 20;  // Example size (<= BLOCK_THREADS)
    int h_in[N];
    int h_out[N];

    // Initialize input with random values
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < N; ++i) {
        h_in[i] = std::rand() % 10;  // small numbers for clarity
    }

    // Device memory allocation
    int *d_in = nullptr;
    int *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(int)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel with one block
    dim3 grid(1);
    dim3 block(BLOCK_THREADS);
    block_scan_kernel<<<grid, block>>>(d_in, d_out, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "Input array:" << std::endl;
    for (int i = 0; i < N; ++i) std::cout << h_in[i] << ' ';
    std::cout << std::endl;

    std::cout << "Exclusive scan result:" << std::endl;
    for (int i = 0; i < N; ++i) std::cout << h_out[i] << ' ';
    std::cout << std::endl;

    // Clean up
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
