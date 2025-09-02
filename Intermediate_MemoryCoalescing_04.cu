/*
Aim of the program:
Write a kernel that copies an array, but with a strided access pattern: `output[i] = input[i * STRIDE]`. Run this for `STRIDE=1` and `STRIDE=2`. Profile both and observe the drop in memory bandwidth for the strided case due to uncoalesced access.

Thinking:
1. We need a CUDA kernel that reads from an input array using a stride and writes the result to an output array.  
2. The stride will be passed as a kernel argument so we can test multiple values with the same code.  
3. To allow stride = 2 we allocate an input array that is at least twice the logical size of the logical input.  
4. For a fair comparison we will use the same number of logical elements (`N`) for both runs.  
5. We measure the kernel execution time with CUDA events and compute the effective bandwidth as  
   `bandwidth = (bytes_read + bytes_written) / time`.  
   The amount of data read/written is the same for both runs, so any difference in bandwidth comes from
   memory access efficiency.  
6. After the kernel launch we copy the result back to host to make sure the kernel actually executed
   and to validate correctness (simple check).  
7. Finally we free all allocated memory and report the measured bandwidths.

The code below follows this plan.  It is self‑contained, uses only the CUDA runtime API,
and can be compiled with `nvcc` to produce a `.cu` executable.  The main function runs the
kernel twice, once with stride = 1 and once with stride = 2, prints timing information, and
shows that the stride‑2 run has a noticeably lower bandwidth due to uncoalesced accesses.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 24)   // 16,777,216 elements
#define THREADS_PER_BLOCK 256
#define MAX_STRIDE 2

/* Utility macro for error checking */
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/* Kernel that copies input to output with a given stride */
__global__ void copy_with_stride(const float *input, float *output, int stride, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx * stride];
    }
}

/* Simple host function to verify correctness */
bool verify(const float *host_in, const float *host_out, int stride, int size) {
    for (int i = 0; i < size; ++i) {
        if (host_out[i] != host_in[i * stride]) {
            printf("Verification failed at index %d: %f != %f\n",
                   i, host_out[i], host_in[i * stride]);
            return false;
        }
    }
    return true;
}

int main(void) {
    /* Allocate host memory */
    size_t logicalSize = N;                // logical number of elements we want to copy
    size_t inputSize  = N * MAX_STRIDE;    // physical input size to accommodate stride 2
    size_t outputSize = N;                 // output size stays logicalSize

    float *h_input  = (float *)malloc(inputSize * sizeof(float));
    float *h_output = (float *)malloc(outputSize * sizeof(float));

    /* Initialize input with some pattern */
    for (size_t i = 0; i < inputSize; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    /* Allocate device memory */
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void **)&d_input,  inputSize  * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_output, outputSize * sizeof(float)));

    /* Copy input to device */
    CHECK_CUDA(cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice));

    /* Kernel launch parameters */
    int blocks = (logicalSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    /* Measure for stride = 1 */
    int stride = 1;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, 0));
    copy_with_stride<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, stride, logicalSize);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    double seconds = ms / 1000.0;
    double bytes = 2.0 * logicalSize * sizeof(float);  // read + write
    double bandwidth_gb_s = (bytes / seconds) / 1e9;

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Stride = %d\n", stride);
    printf("Time elapsed: %.3f ms\n", ms);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    printf("Verification: %s\n\n", verify(h_input, h_output, stride, logicalSize) ? "PASS" : "FAIL");

    /* Measure for stride = 2 */
    stride = 2;
    CHECK_CUDA(cudaEventRecord(start, 0));
    copy_with_stride<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, stride, logicalSize);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    seconds = ms / 1000.0;
    bandwidth_gb_s = (bytes / seconds) / 1e9;

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Stride = %d\n", stride);
    printf("Time elapsed: %.3f ms\n", ms);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    printf("Verification: %s\n", verify(h_input, h_output, stride, logicalSize) ? "PASS" : "FAIL");

    /* Clean up */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}
