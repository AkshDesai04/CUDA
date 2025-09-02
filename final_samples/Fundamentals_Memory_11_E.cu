```cuda
/*
 * Aim of the program:
 * Demonstrate the performance benefit: write two kernels, one where each thread reads a configuration parameter from global memory,
 * and one where it reads from constant memory. Time both.
 *
 * Thinking:
 * - To compare global vs constant memory access, we create a simple kernel that performs a trivial operation using a configuration
 *   parameter. Each thread reads a config value and multiplies it with an element from an input array, accumulating a result.
 * - For the global memory case, the configuration array resides in device global memory. For the constant memory case, we
 *   copy the same configuration array into a __constant__ variable via cudaMemcpyToSymbol.
 * - We run each kernel many times to obtain stable timing measurements. CUDA events are used to time GPU execution.
 * - Since constant memory is cached per block, we expect the constant version to be faster, especially when the config array
 *   is small enough to fit entirely in the constant cache.
 * - We keep the code simple: the input array is a large array of floats; the configuration array is small (e.g., 256 ints).
 * - Each thread processes one element of the input array and uses the configuration parameter indexed by its thread ID modulo
 *   the config size. The result is written back to an output array (though the output is not used further).
 *
 * The program will print the average execution time for each kernel over a number of iterations.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE   (1 << 20)          // 1M elements
#define CONFIG_SIZE  256                // 256 configuration parameters
#define BLOCK_SIZE   256                // Threads per block
#define NUM_ITER     100                // Number of kernel launches for timing

// Constant memory copy of the configuration array
__constant__ int const_config[CONFIG_SIZE];

// Kernel that reads configuration parameter from global memory
__global__ void kernel_global(const float *input, const int *config, float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ARRAY_SIZE) return;

    // Use config parameter indexed by threadIdx.x (modulo CONFIG_SIZE)
    int cfg = config[threadIdx.x % CONFIG_SIZE];
    float val = input[idx];

    // Simple computation: multiply and store
    output[idx] = val * cfg;
}

// Kernel that reads configuration parameter from constant memory
__global__ void kernel_constant(const float *input, float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ARRAY_SIZE) return;

    int cfg = const_config[threadIdx.x % CONFIG_SIZE];
    float val = input[idx];

    output[idx] = val * cfg;
}

int main()
{
    // Allocate host memory
    float *h_input  = (float *)malloc(ARRAY_SIZE * sizeof(float));
    float *h_output = (float *)malloc(ARRAY_SIZE * sizeof(float));
    int   *h_config = (int   *)malloc(CONFIG_SIZE * sizeof(int));

    // Initialize host data
    for (int i = 0; i < ARRAY_SIZE; ++i) h_input[i] = 1.0f;   // simple data
    for (int i = 0; i < CONFIG_SIZE; ++i) h_config[i] = i + 1; // 1..256

    // Allocate device memory
    float *d_input, *d_output;
    int   *d_config;
    cudaMalloc((void **)&d_input,   ARRAY_SIZE * sizeof(float));
    cudaMalloc((void **)&d_output,  ARRAY_SIZE * sizeof(float));
    cudaMalloc((void **)&d_config,  CONFIG_SIZE * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_input,  h_input,  ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_config, h_config, CONFIG_SIZE * sizeof(int),  cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(const_config, h_config, CONFIG_SIZE * sizeof(int));

    // Kernel launch configuration
    dim3 block(BLOCK_SIZE);
    dim3 grid((ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsed_global = 0.0f;
    for (int i = 0; i < NUM_ITER; ++i) {
        cudaEventRecord(start, 0);
        kernel_global<<<grid, block>>>(d_input, d_config, d_output);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        elapsed_global += ms;
    }
    elapsed_global /= NUM_ITER;

    float elapsed_constant = 0.0f;
    for (int i = 0; i < NUM_ITER; ++i) {
        cudaEventRecord(start, 0);
        kernel_constant<<<grid, block>>>(d_input, d_output);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        elapsed_constant += ms;
    }
    elapsed_constant /= NUM_ITER;

    // Copy result back (optional)
    cudaMemcpy(h_output, d_output, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Report timings
    printf("Average time over %d iterations:\n", NUM_ITER);
    printf("  Global memory access:  %.3f ms\n", elapsed_global);
    printf("  Constant memory access: %.3f ms\n", elapsed_constant);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_config);
    free(h_input);
    free(h_output);
    free(h_config);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```