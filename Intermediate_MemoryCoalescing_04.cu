```cpp
/* 
Aim of the program:
Write a kernel that copies an array, but with a strided access pattern: 
output[i] = input[i * STRIDE]. Run this for STRIDE=1 and STRIDE=2. 
Profile both and observe the drop in memory bandwidth for the strided case 
due to uncoalesced access.

Thinking:
- We want to demonstrate the effect of memory coalescing on a simple copy
  kernel.  The kernel will read from an input array using a stride and 
  write the result to an output array.
- For stride=1 the accesses are contiguous and fully coalesced.  For 
  stride=2 the reads are strided, which on a GPU causes each warp to
  read from non‑contiguous memory locations, resulting in reduced 
  memory bandwidth.
- We'll measure the kernel execution time with cudaEvent_t, compute the
  amount of data read/written, and then calculate an effective memory
  bandwidth (bytes transferred / seconds).  This will allow us to see
  the bandwidth drop for stride=2.
- To keep the code simple, we use a single CUDA kernel template that
  accepts the stride as a template parameter.  We launch the kernel
  twice: once with STRIDE=1 and once with STRIDE=2.
- The input array size is chosen large enough (N = 1 << 24, about 16M 
  elements) to provide a steady stream of memory traffic.  For the 
  strided case we allocate input of size N * MAX_STRIDE so that both
  stride patterns can read safely.
- After kernel execution we copy the result back to host (not timed)
  and print the computed bandwidth for each case.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>

// Problem size: 16M elements
#define N (1 << 24)

// Maximum stride we will test
#define MAX_STRIDE 2

// Kernel that copies input to output with a given stride
template <int STRIDE>
__global__ void copy_strided(const float* __restrict__ input, float* __restrict__ output, int outSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outSize)
    {
        // Guard against out-of-bounds if outSize * STRIDE > input size
        output[idx] = input[idx * STRIDE];
    }
}

// Utility to check CUDA errors
inline void checkCuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Allocate host memory
    size_t inputSize = N * MAX_STRIDE * sizeof(float);
    size_t outputSize = N * sizeof(float);
    float* h_input  = (float*)malloc(inputSize);
    float* h_output = (float*)malloc(outputSize);
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input with random data
    std::srand((unsigned)std::time(nullptr));
    for (size_t i = 0; i < N * MAX_STRIDE; ++i)
        h_input[i] = static_cast<float>(std::rand()) / RAND_MAX;

    // Allocate device memory
    float* d_input;
    float* d_output;
    checkCuda(cudaMalloc((void**)&d_input, inputSize), "malloc d_input");
    checkCuda(cudaMalloc((void**)&d_output, outputSize), "malloc d_output");

    // Copy input to device
    checkCuda(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice), "memcpy h->d");

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Events for timing
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "eventCreate start");
    checkCuda(cudaEventCreate(&stop),  "eventCreate stop");

    // Helper lambda to run kernel with given stride
    auto run_kernel = [&](int stride) {
        // Record start
        checkCuda(cudaEventRecord(start, 0), "eventRecord start");
        // Launch kernel
        if (stride == 1)
            copy_strided<1><<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
        else if (stride == 2)
            copy_strided<2><<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
        else
            fprintf(stderr, "Unsupported stride %d\n", stride);
        // Record stop
        checkCuda(cudaEventRecord(stop, 0), "eventRecord stop");
        // Wait for completion
        checkCuda(cudaEventSynchronize(stop), "eventSynchronize");
        // Compute elapsed time
        float ms = 0;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "eventElapsedTime");
        // Compute bytes transferred: read + write
        size_t bytesRead  = N * stride * sizeof(float);
        size_t bytesWrite = N * sizeof(float);
        size_t totalBytes = bytesRead + bytesWrite;
        double bandwidth = (totalBytes / 1e9) / (ms / 1000.0); // GB/s
        std::cout << "Stride " << stride << ": Time = "
                  << std::fixed << std::setprecision(3) << ms << " ms, "
                  << "Bandwidth = " << bandwidth << " GB/s\n";
    };

    // Run for stride 1
    run_kernel(1);
    // Run for stride 2
    run_kernel(2);

    // Copy result back to host (optional)
    checkCuda(cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost), "memcpy d->h");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
```