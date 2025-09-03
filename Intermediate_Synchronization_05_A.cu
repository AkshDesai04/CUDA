```cuda
/*
The kernel will take an `int* input_data` and an `int* bins`. Each thread processes one or more elements. For each element `val`, it does `atomicAdd(&bins[val], 1)`.

Thinking:
- I need a CUDA kernel that receives an array of integers (input_data) and an array that represents histogram bins (bins).  
- Each thread should loop over its assigned elements, using a typical grid-stride loop so that we can support any number of input elements, not just a multiple of the block size.  
- For each element read from input_data, we perform an atomic addition to the corresponding bin.  
- To avoid out‑of‑bounds writes we check that the value is in the valid range [0, num_bins).  
- The host code will allocate device memory for the input array and the bins array, initialize them, launch the kernel, and copy back the result.  
- I will add error checking after memory operations and after the kernel launch.  
- For demonstration I will fill the input array with random values in the valid range and set the number of bins via a compile‑time constant or a command‑line argument.  
- The code will be self‑contained in a single .cu file, compile with nvcc, and print the resulting histogram.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// Kernel that atomically increments histogram bins
__global__ void histogramKernel(const int *input_data, int *bins, int N, int num_bins)
{
    // Grid‑stride loop to cover all elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride)
    {
        int val = input_data[i];
        // Bounds check to avoid invalid memory access
        if (val >= 0 && val < num_bins)
        {
            atomicAdd(&bins[val], 1);
        }
    }
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

// Simple helper to print an array
void printArray(const int *arr, int size)
{
    for (int i = 0; i < size; ++i)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    // Parameters (can be overridden by command line)
    int N = 1000000;      // Number of input elements
    int num_bins = 256;   // Number of histogram bins

    if (argc >= 2)
    {
        N = atoi(argv[1]);
        if (argc >= 3)
            num_bins = atoi(argv[2]);
    }

    printf("Histogram example: N=%d, num_bins=%d\n", N, num_bins);

    // Allocate host memory
    int *h_input = (int *)malloc(N * sizeof(int));
    int *h_bins  = (int *)malloc(num_bins * sizeof(int));
    if (!h_input || !h_bins)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input with random values in [0, num_bins)
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i)
        h_input[i] = rand() % num_bins;

    // Initialize bins to zero
    for (int i = 0; i < num_bins; ++i)
        h_bins[i] = 0;

    // Allocate device memory
    int *d_input = NULL;
    int *d_bins  = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_bins,  num_bins * sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize device bins to zero
    CUDA_CHECK(cudaMemset(d_bins, 0, num_bins * sizeof(int)));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    histogramKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_bins, N, num_bins);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_bins, d_bins, num_bins * sizeof(int), cudaMemcpyDeviceToHost));

    // Print first few bins for inspection
    printf("First 10 bins:\n");
    printArray(h_bins, 10);

    // Clean up
    free(h_input);
    free(h_bins);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_bins));

    printf("Histogram computation completed.\n");
    return EXIT_SUCCESS;
}
```