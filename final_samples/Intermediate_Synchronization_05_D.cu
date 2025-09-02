```cuda
/* 
A performance optimization: have each thread compute a private histogram in registers or shared memory first. Then, at the end of the thread, atomically add the private histogram's counts to the global bins. This reduces atomic contention.
 
Thinking:
- To reduce atomic contention we avoid a global atomic for every data element. 
- Each thread builds a small private histogram in its own registers (or shared memory if registers would be too large). 
- After processing all its assigned data, the thread performs a small number of atomicAdd calls – one per bin that it encountered. 
- In this example we assume a modest number of bins (32) so that the private histogram comfortably fits in registers. 
- The kernel loops over the input array in a grid-stride loop. 
- After the loop each thread aggregates its local counts into the global histogram with atomicAdd. 
- The global histogram is zero‑initialised before launch. 
- The host code allocates memory, fills an input array with random values, copies to the device, runs the kernel, copies back the result, and prints it. 
- Error checking is added for robustness. 
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                     \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",      \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),         \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Number of bins in the histogram (must be <= 32 for register allocation)
#define BIN_COUNT 32

__global__ void histogramKernel(const unsigned int *d_in, unsigned int *d_hist, int N)
{
    // Each thread has a private histogram in registers
    unsigned int localHist[BIN_COUNT] = {0};

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop over input data
    for (int i = tid; i < N; i += stride) {
        unsigned int val = d_in[i];
        if (val < BIN_COUNT) {
            localHist[val]++;
        }
    }

    // After processing, atomically add the private counts to global histogram
    for (int b = 0; b < BIN_COUNT; ++b) {
        unsigned int count = localHist[b];
        if (count) {
            atomicAdd(&d_hist[b], count);
        }
    }
}

int main()
{
    const int N = 1 << 24;  // 16 million elements
    size_t bytes = N * sizeof(unsigned int);

    // Allocate host memory
    unsigned int *h_in = (unsigned int*)malloc(bytes);
    if (!h_in) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input with random values [0, BIN_COUNT-1]
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_in[i] = rand() % BIN_COUNT;
    }

    // Allocate device memory
    unsigned int *d_in = NULL;
    unsigned int *d_hist = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_in, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_hist, BIN_COUNT * sizeof(unsigned int)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Initialise global histogram to zero
    CHECK_CUDA(cudaMemset(d_hist, 0, BIN_COUNT * sizeof(unsigned int)));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    // Cap blocks to a reasonable number to avoid too many blocks
    if (blocks > 1024) blocks = 1024;

    histogramKernel<<<blocks, threadsPerBlock>>>(d_in, d_hist, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy histogram back to host
    unsigned int h_hist[BIN_COUNT] = {0};
    CHECK_CUDA(cudaMemcpy(h_hist, d_hist, BIN_COUNT * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Print histogram
    printf("Histogram results:\n");
    for (int b = 0; b < BIN_COUNT; ++b) {
        printf("Bin %2d: %10u\n", b, h_hist[b]);
    }

    // Clean up
    free(h_in);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_hist));

    return 0;
}
```