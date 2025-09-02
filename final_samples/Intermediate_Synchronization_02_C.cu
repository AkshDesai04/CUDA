```cuda
/*********************************************************************
* Aim of the program:
* "Modify the algorithm to find not just the maximum value, but also
* the index of that value. This requires carrying a struct `{value,
* index}` through the reduction."
*
* Thinking:
* The classic parallel reduction to find a maximum value can be
* extended to also track the index of that value by carrying a
* small struct `{value, index}` instead of just a scalar.  In a
* CUDA reduction we usually:
* 1) Load data into registers or shared memory.
* 2) Iteratively combine pairs of elements until a single value
*    remains per block.
* 3) Write each block's partial result to a temporary array.
* 4) Finally combine those partial results to get the global result.
*
* For this program we adopt a two‑stage approach:
*   • Stage 1 (kernel): Each block reduces its assigned portion of
*     the input array into a single `MaxVal` struct (value + index)
*     and writes it into a `partial` array of size equal to the
*     number of blocks.
*   • Stage 2 (host): The host copies the `partial` array back to
*     the CPU and performs a simple serial reduction to obtain the
*     global maximum and its index.
*
* The struct is defined as:
*   struct MaxVal {
*       float value;  // maximum value
*       int   index;  // index of that value
*   };
*
* During the intra‑block reduction we use shared memory for
* `MaxVal` and compare pairs, keeping the larger value and its
* index.  In case of ties we keep the smallest index for
* determinism.  The final result is printed to standard output.
*
* The program is self‑contained, compiles with `nvcc`, and uses
* only the CUDA runtime API.
*********************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ------------------------------------------------------------------
// Struct to hold a value and its index
struct MaxVal {
    float value;
    int   index;
};

// ------------------------------------------------------------------
// Kernel: each block reduces a chunk of the array to a single MaxVal
__global__ void max_reduce_kernel(const float *d_in, MaxVal *d_partial, int N)
{
    extern __shared__ MaxVal sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Each thread loads two elements (if available) and keeps the larger
    MaxVal local;
    local.value = -FLT_MAX;
    local.index = -1;

    if (idx < N) {
        local.value = d_in[idx];
        local.index = idx;
    }

    if (idx + blockDim.x < N) {
        float val2 = d_in[idx + blockDim.x];
        int   idx2 = idx + blockDim.x;
        if (val2 > local.value || (val2 == local.value && idx2 < local.index)) {
            local.value = val2;
            local.index = idx2;
        }
    }

    // Load into shared memory
    sdata[tid] = local;
    __syncthreads();

    // Intra‑block reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            MaxVal other = sdata[tid + stride];
            if (other.value > sdata[tid].value ||
                (other.value == sdata[tid].value && other.index < sdata[tid].index)) {
                sdata[tid] = other;
            }
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        d_partial[blockIdx.x] = sdata[0];
    }
}

// ------------------------------------------------------------------
// Helper to check CUDA errors
inline void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ------------------------------------------------------------------
int main()
{
    const int N = 1 << 20;          // 1M elements
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    // Allocate host memory
    float *h_in = (float*)malloc(N * sizeof(float));
    if (!h_in) { perror("malloc"); return EXIT_FAILURE; }

    // Initialize data with some pattern
    for (int i = 0; i < N; ++i) {
        h_in[i] = sinf(i * 0.01f) * 1000.f + i;  // increasing trend
    }

    // Allocate device memory
    float *d_in;
    checkCuda(cudaMalloc((void**)&d_in, N * sizeof(float)), "cudaMalloc d_in");

    // Copy input to device
    checkCuda(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // Allocate array for partial results
    MaxVal *d_partial;
    checkCuda(cudaMalloc((void**)&d_partial, blocks * sizeof(MaxVal)), "cudaMalloc d_partial");

    // Launch kernel
    size_t sharedMemBytes = threadsPerBlock * sizeof(MaxVal);
    max_reduce_kernel<<<blocks, threadsPerBlock, sharedMemBytes>>>(d_in, d_partial, N);
    checkCuda(cudaGetLastError(), "kernel launch");

    // Copy partial results back to host
    MaxVal *h_partial = (MaxVal*)malloc(blocks * sizeof(MaxVal));
    if (!h_partial) { perror("malloc h_partial"); return EXIT_FAILURE; }

    checkCuda(cudaMemcpy(h_partial, d_partial, blocks * sizeof(MaxVal), cudaMemcpyDeviceToHost),
              "cudaMemcpy D2H");

    // Final reduction on host
    MaxVal global;
    global.value = -FLT_MAX;
    global.index = -1;

    for (int i = 0; i < blocks; ++i) {
        if (h_partial[i].value > global.value ||
            (h_partial[i].value == global.value && h_partial[i].index < global.index)) {
            global = h_partial[i];
        }
    }

    // Output result
    printf("Maximum value: %f\n", global.value);
    printf("Index of maximum value: %d\n", global.index);

    // Clean up
    free(h_in);
    free(h_partial);
    checkCuda(cudaFree(d_in), "cudaFree d_in");
    checkCuda(cudaFree(d_partial), "cudaFree d_partial");

    return 0;
}
```