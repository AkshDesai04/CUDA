```cuda
/*
   Verify the `double` precision results on the CPU. Note the potential for tiny floating point discrepancies.

   The goal of this program is to demonstrate how double‑precision computations performed on a CUDA device
   may differ, by a very small amount, from the same computations carried out on the host CPU.
   Because the GPU performs operations in a different order (especially in parallel reductions),
   the IEEE 754 rounding rules can produce tiny differences.  The program will:

   1. Allocate an array of doubles on both host and device.
   2. Initialize the host array with a known pattern (e.g. a quadratic function).
   3. Launch a CUDA kernel that sums the array elements using a parallel reduction.
   4. Retrieve the GPU result and compute the sum on the host CPU using a simple loop.
   5. Compare the two results and print the absolute difference, the relative difference,
      and whether the difference is within a user‑specified tolerance.

   The code also prints the device's compute capability to remind the user that double‑precision
   performance is dependent on the GPU architecture (sm_20+).  The chosen tolerance (1e-9) is
   representative of what is often acceptable for double precision, but it can be tuned
   depending on the application.

   This example highlights that, while results should be mathematically equivalent,
   floating‑point arithmetic on parallel hardware can introduce very small discrepancies
   that are perfectly normal and usually harmless.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Size of the array (must be a power of two for simplicity)
#define N (1 << 20)  // 1,048,576 elements

// CUDA error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",   \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),         \
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel to perform a parallel reduction to sum an array of doubles.
// Each block will produce a partial sum in shared memory, and the final
// sum will be written to the output array (one element per block).
__global__ void sum_reduction(const double *in, double *out, int n) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double sum = 0.0;

    // Load two elements per thread to improve memory throughput
    if (idx < n)
        sum += in[idx];
    if (idx + blockDim.x < n)
        sum += in[idx + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // In‑block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

int main() {
    // Print device properties
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Using device %d: %s (SM %d.%d)\n",
           dev, prop.name, prop.major, prop.minor);
    if (prop.major < 2) {
        printf("Warning: Double precision performance may be poor on this GPU.\n");
    }

    // Allocate host array
    double *h_in = (double*)malloc(N * sizeof(double));
    double *h_out_gpu = (double*)malloc(((N + 1023) / 1024) * sizeof(double));
    double *h_out_cpu = (double*)malloc(((N + 1023) / 1024) * sizeof(double));

    // Initialize host input with a known pattern: f(i) = 1/(i+1)
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0 / (double)(i + 1);
    }

    // Allocate device memory
    double *d_in;
    double *d_partial;
    CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(double)));
    int blocks = (N + 1023) / 1024; // 1024 threads per block
    CUDA_CHECK(cudaMalloc((void**)&d_partial, blocks * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(double), cudaMemcpyHostToDevice));

    // Launch reduction kernel
    int threads = 1024;
    size_t sharedMemSize = threads * sizeof(double);
    sum_reduction<<<blocks, threads, sharedMemSize>>>(d_in, d_partial, N);
    CUDA_CHECK(cudaGetLastError());

    // If more than one block, we need to reduce the partial sums
    double *d_result = d_partial;
    int curBlocks = blocks;
    while (curBlocks > 1) {
        int nextBlocks = (curBlocks + threads - 1) / threads;
        sum_reduction<<<nextBlocks, threads, threads * sizeof(double)>>>(d_result, d_result, curBlocks);
        CUDA_CHECK(cudaGetLastError());
        curBlocks = nextBlocks;
    }

    // Copy final result back to host
    double gpu_sum = 0.0;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_result, sizeof(double), cudaMemcpyDeviceToHost));

    // Compute sum on CPU
    double cpu_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        cpu_sum += h_in[i];
    }

    // Compare results
    double abs_diff = fabs(gpu_sum - cpu_sum);
    double rel_diff = abs_diff / fabs(cpu_sum);
    double tolerance = 1e-9; // acceptable relative difference

    printf("GPU sum: %.15e\n", gpu_sum);
    printf("CPU sum: %.15e\n", cpu_sum);
    printf("Absolute difference: %.15e\n", abs_diff);
    printf("Relative difference: %.15e\n", rel_diff);
    if (rel_diff <= tolerance) {
        printf("Result is within the tolerance of %e.\n", tolerance);
    } else {
        printf("Result differs more than the tolerance of %e.\n", tolerance);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    free(h_in);
    free(h_out_gpu);
    free(h_out_cpu);

    return 0;
}
```