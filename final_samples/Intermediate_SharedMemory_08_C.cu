/*
Aim of the program: Implement a min-finding reduction.

Thinking and design:

1. **Problem Statement**  
   We need a CUDA program that finds the minimum value in a large integer array using a parallel reduction approach.  
   The program will run on the GPU, load the data, perform the reduction in one or more kernel launches, and copy the result back to the host.

2. **Algorithm**  
   *Tree‑based reduction* in shared memory is the most common approach.  
   Each block processes a chunk of the input array.  
   Each thread loads one element (or sets `INT_MAX` if out of bounds) into shared memory, then repeatedly halves the number of active threads, comparing pairs of elements and keeping the smaller one.  
   After the loop, the first thread of each block writes the block’s minimum to an output array (`g_odata`).  
   We then launch a second kernel (or iterate) on the block‑min array until only one element remains.

3. **Kernel Design**  
   ```cpp
   __global__ void reduceMin(const int *g_idata, int *g_odata, int n);
   ```  
   * `g_idata` – device pointer to the input data.  
   * `g_odata` – device pointer to store per‑block minima.  
   * `n`      – total number of elements.  
   * Shared memory (`sdata`) holds the partial values for the block.  
   * We use `extern __shared__ int sdata[];` to allow the block size to be variable.  
   * Threads that fall outside the array bounds write `INT_MAX` so they do not affect the min.

4. **Host Logic**  
   * Allocate and initialise a host array.  
   * Allocate device memory for the input and two output arrays (to allow ping‑ponging).  
   * Copy the host array to the device.  
   * Compute grid dimensions: `gridSize = (n + blockSize - 1) / blockSize`.  
   * Launch the first reduction kernel to compute block minima.  
   * If the number of blocks is > 1, launch the reduction again on the block‑min array.  
   * After the final kernel, copy the single minimum value back to the host.  
   * Clean up GPU memory and exit.

5. **Edge Cases**  
   * If the input size is not a multiple of the block size, the extra threads write `INT_MAX` so they do not interfere.  
   * We use `INT_MAX` from `<limits.h>` as a safe sentinel value.  
   * The code handles arrays of arbitrary size, including those larger than the number of available CUDA cores.

6. **Testing**  
   The main function creates a small test array (e.g., 1 000 000 integers) with a known minimum value and verifies that the GPU result matches the expected value.  

7. **Performance Considerations**  
   * The algorithm is O(log N) in terms of the number of kernel launches (actually log₂(N/blockSize) for the full reduction).  
   * Shared memory usage is minimal: one integer per thread.  
   * The kernel uses a simple loop for reduction; for larger block sizes one could unroll the loop for better performance, but the code remains clear for educational purposes.

Overall, this program demonstrates a classic parallel reduction pattern to find the minimum value in an array using CUDA C. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda_runtime.h>

/* Kernel that reduces a segment of the input array to its minimum value.
   Each block processes a chunk of the array. */
__global__ void reduceMin(const int *g_idata, int *g_odata, int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory, guarding against out-of-bounds accesses
    sdata[tid] = (idx < n) ? g_idata[idx] : INT_MAX;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            int val = sdata[tid + s];
            if (val < sdata[tid])
                sdata[tid] = val;
        }
        __syncthreads();
    }

    // Write result of this block to global memory
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

/* Utility function to check CUDA errors */
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    // Example: find min in an array of 1,000,000 integers
    const int N = 1000000;
    const int BLOCK_SIZE = 256;

    // Host allocation
    int *h_array = (int *)malloc(N * sizeof(int));
    if (!h_array) { fprintf(stderr, "Failed to allocate host array\n"); return EXIT_FAILURE; }

    // Initialise array with random values; set a known minimum
    int known_min = 42;
    h_array[0] = known_min; // Put known minimum at first element
    for (int i = 1; i < N; ++i)
    {
        h_array[i] = rand() % 1000 + 1; // Random values between 1 and 1000
    }

    // Device allocation
    int *d_in = nullptr;
    int *d_out = nullptr;
    int *d_out_tmp = nullptr;
    int err;

    err = cudaMalloc((void **)&d_in, N * sizeof(int));
    checkCudaError(err, "allocating d_in");

    // We allocate enough space for the largest intermediate result: at most N elements
    err = cudaMalloc((void **)&d_out, N * sizeof(int));
    checkCudaError(err, "allocating d_out");
    err = cudaMalloc((void **)&d_out_tmp, N * sizeof(int));
    checkCudaError(err, "allocating d_out_tmp");

    // Copy input to device
    err = cudaMemcpy(d_in, h_array, N * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError(err, "copying to device");

    // First reduction pass
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int sharedMemSize = BLOCK_SIZE * sizeof(int);

    reduceMin<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_in, d_out, N);
    err = cudaGetLastError();
    checkCudaError(err, "launching first reduceMin kernel");

    // Iteratively reduce until we have a single value
    int src = 0; // index in d_out that holds current data
    int dst = 1; // index in d_out_tmp that will hold next result
    int currN = numBlocks;

    while (currN > 1)
    {
        int blocks = (currN + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // Launch reduction on currN elements stored in d_out
        reduceMin<<<blocks, BLOCK_SIZE, sharedMemSize>>>( (src==0 ? d_out : d_out_tmp), (dst==0 ? d_out : d_out_tmp), currN );
        err = cudaGetLastError();
        checkCudaError(err, "launching subsequent reduceMin kernel");

        // Prepare for next iteration
        currN = blocks;
        src ^= 1;
        dst ^= 1;
    }

    // Copy the result back to host
    int h_min;
    err = cudaMemcpy(&h_min, (src==0 ? d_out : d_out_tmp), sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError(err, "copying result to host");

    printf("Minimum value found: %d\n", h_min);
    printf("Known minimum: %d\n", known_min);

    // Clean up
    free(h_array);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_tmp);

    return 0;
}
