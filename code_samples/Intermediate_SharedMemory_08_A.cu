```cpp
/*
The reduction loop looks like: `for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) { s_data[threadIdx.x] += s_data[threadIdx.x + s]; } __syncthreads(); }` Implement this.

Thinking:
- The task is to implement a parallel reduction using CUDA shared memory.
- The reduction loop provided iteratively halves the number of active threads, summing pairs of elements in shared memory.
- To make the example self‑contained, I’ll create a simple kernel that reduces an array of floats to a single sum.
- The kernel will first load two elements per thread (to make use of memory bandwidth), store the partial sum in shared memory, then perform the reduction loop.
- On the host side, I’ll launch the kernel iteratively, each time halving the number of elements, until only one element remains.
- Edge cases such as array size not being a multiple of the block stride are handled with bounds checks.
- The program will allocate memory on the device, copy input data, perform the reduction, copy the result back, and print it.
- I’ll use a block size of 256 threads; this can be tuned for performance but is sufficient for demonstration.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// Kernel that reduces an input array to block‑wise partial sums
__global__ void reduceSumKernel(const float *input, float *output, size_t n)
{
    extern __shared__ float s_data[];

    unsigned int tid   = threadIdx.x;
    unsigned int idx   = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float sum = 0.0f;

    // Load up to two elements per thread into sum
    if (idx < n)            sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];

    s_data[tid] = sum;
    __syncthreads();

    // Reduction loop
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // Write result of this block to output
    if (tid == 0)
        output[blockIdx.x] = s_data[0];
}

// Host helper to launch reduction until a single value remains
float reduceSum(const float *d_in, size_t n)
{
    float *d_in_curr  = const_cast<float*>(d_in);
    float *d_out_curr;
    size_t  curr_n = n;

    // Allocate temporary buffer for block results
    cudaMalloc(&d_out_curr, ((curr_n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)) * sizeof(float));

    while (curr_n > 1)
    {
        size_t num_blocks = (curr_n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

        reduceSumKernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            d_in_curr, d_out_curr, curr_n);

        cudaDeviceSynchronize();

        // Prepare for next iteration
        curr_n = num_blocks;
        cudaFree(d_in_curr);
        d_in_curr = d_out_curr;

        // Allocate new output buffer if needed
        if (curr_n > 1)
            cudaMalloc(&d_out_curr, ((curr_n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)) * sizeof(float));
    }

    // Copy final result back to host
    float h_result;
    cudaMemcpy(&h_result, d_in_curr, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in_curr);
    if (d_out_curr != nullptr) cudaFree(d_out_curr);
    return h_result;
}

int main()
{
    const size_t N = 1 << 20; // 1M elements
    size_t size_bytes = N * sizeof(float);

    // Allocate and initialize host array
    float *h_in = (float*)malloc(size_bytes);
    for (size_t i = 0; i < N; ++i) h_in[i] = 1.0f; // simple test value

    // Allocate device memory
    float *d_in;
    cudaMalloc(&d_in, size_bytes);
    cudaMemcpy(d_in, h_in, size_bytes, cudaMemcpyHostToDevice);

    // Perform reduction
    float sum = reduceSum(d_in, N);

    printf("Sum = %f\n", sum); // Expect N * 1.0f

    // Clean up
    cudaFree(d_in);
    free(h_in);

    return 0;
}
```