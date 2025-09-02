/*
Create a parent kernel that launches a dynamically sized grid for a child kernel based on data computed within the parent.
I started by deciding that the most straightforward way to demonstrate dynamic parallelism in CUDA is to use a parent kernel that performs some calculation on the input data and then uses the result of that calculation to determine how many blocks the child kernel should launch. In this example, the parent kernel computes the sum of an integer array. The number of blocks for the child kernel is then set to (sum % 10) + 1 so that it changes each time the parent is launched with different data. The child kernel simply squares each element of the array, demonstrating that it has been launched from device code. The host code allocates memory, initializes data, launches the parent kernel with the `-rdc=true` option (required for dynamic parallelism), and copies the result back for verification. All error checking is performed with a helper macro for brevity.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Child kernel: squares each element of the array
__global__ void childKernel(const int *in, int N, int *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx] * in[idx];
    }
}

// Parent kernel: computes the sum of the array and launches childKernel
__global__ void parentKernel(const int *in, int N, int *out) {
    // Each thread sums one element
    __shared__ int partialSum[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    int val = (idx < N) ? in[idx] : 0;
    partialSum[tid] = val;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialSum[tid] += partialSum[tid + stride];
        }
        __syncthreads();
    }

    // Only thread 0 of each block writes its partial sum to global memory
    if (tid == 0) {
        atomicAdd(&out[0], partialSum[0]);  // use first element of out as accumulator
    }
    __syncthreads();

    // After all blocks have summed, thread 0 of block 0 launches child kernel
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Ensure all partial sums are accounted for
        __syncthreads();
        // Read the total sum from out[0]
        int totalSum = out[0];
        // Determine number of blocks for child kernel
        int childBlocks = (totalSum % 10) + 1;   // dynamic value
        int childThreads = 128;                  // fixed number of threads per block

        // Launch child kernel from device
        childKernel<<<childBlocks, childThreads>>>(in, N, out);
    }
}

int main(void) {
    const int N = 1024;
    size_t bytes = N * sizeof(int);
    int h_in[N], h_out[N];

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_in[i] = i + 1; // simple sequence
    }

    // Device memory
    int *d_in = nullptr;
    int *d_out = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_in, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_out, bytes));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Initialize output to zero (first element used as accumulator)
    int zero = 0;
    CHECK_CUDA(cudaMemcpy(d_out, &zero, sizeof(int), cudaMemcpyHostToDevice));

    // Launch parent kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    parentKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, N, d_out);

    // Wait for completion
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Sample outputs:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_in[%d] = %d, h_out[%d] = %d\n", i, h_in[i], i, h_out[i]);
    }

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
