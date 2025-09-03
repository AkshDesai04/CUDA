```cuda
/*
Verify the results on the CPU.

The aim of this program is to demonstrate how to use CUDA to perform a
simple element‑wise vector addition on the GPU and then verify that the
result is correct by computing the same operation on the CPU.  
The program proceeds through the following steps:

1. **Setup**  
   - Allocate two large float arrays (`h_a` and `h_b`) on the host and fill
     them with random values.  
   - Allocate memory for the result on both the host (`h_c`) and on the
     device (`d_a`, `d_b`, `d_c`).  
   - Copy input data from host to device.

2. **GPU Execution**  
   - Define a CUDA kernel `addVectors` that adds two vectors element‑wise.  
   - Launch the kernel with a suitable block and grid configuration
     (256 threads per block).  
   - Copy the computed result back to the host (`h_c`).

3. **CPU Verification**  
   - Compute the same vector addition on the CPU in a simple loop.  
   - Compare each element of the GPU result with the CPU reference and
     report any mismatches.  
   - Print a success message if all elements match.

4. **Cleanup**  
   - Free all allocated host and device memory.

The program includes error checking after each CUDA API call to catch
runtime problems early.  Running this program will produce a message
indicating whether the GPU computation was verified successfully.
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err),    \
                    #call);                                              \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

__global__ void addVectors(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c_gpu = (float *)malloc(size);
    float *h_c_cpu = (float *)malloc(size);

    if (!h_a || !h_b || !h_c_gpu || !h_c_cpu) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    // Seed random number generator and fill host arrays
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void **)&d_a, size));
    CHECK_CUDA(cudaMalloc((void **)&d_b, size));
    CHECK_CUDA(cudaMalloc((void **)&d_c, size));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost));

    // CPU verification
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        h_c_cpu[i] = h_a[i] + h_b[i];
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5f) {
            mismatches++;
            if (mismatches <= 10) { // print first few mismatches
                printf("Mismatch at index %d: CPU=%f GPU=%f\n",
                       i, h_c_cpu[i], h_c_gpu[i]);
            }
        }
    }

    if (mismatches == 0) {
        printf("Verification PASSED: All %d elements match.\n", N);
    } else {
        printf("Verification FAILED: %d mismatches found.\n", mismatches);
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_c_cpu);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}
```