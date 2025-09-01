/*
Aim: Deliberately corrupt one byte of the data on the host before verification to ensure your verification code works.

Thinking:
- Create a simple CUDA program that allocates host and device memory.
- Use a trivial kernel that processes an array, e.g. multiplies each element by 2.
- After copying the processed data back to host, intentionally corrupt a single byte in the host buffer.
- Then run a verification routine that compares the host output against the expected result.
- The corruption should cause the verification to detect a mismatch, confirming that the verification logic is functioning.
- Include basic CUDA error checking and use simple printing to demonstrate success or failure.
- The program is self-contained, uses only standard headers and CUDA runtime API, and can be compiled with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 256

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            return -1;                                              \
        }                                                           \
    } while (0)

__global__ void doubleKernel(const int* in, int* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2;
    }
}

int main()
{
    int *h_in = NULL;
    int *h_out = NULL;
    int *d_in = NULL;
    int *d_out = NULL;

    size_t size = N * sizeof(int);

    // Allocate host memory
    h_in  = (int*)malloc(size);
    h_out = (int*)malloc(size);
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return -1;
    }

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_in[i] = i;
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_in,  size));
    CHECK_CUDA(cudaMalloc((void**)&d_out, size));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    doubleKernel<<<blocks, THREADS_PER_BLOCK>>>(d_in, d_out, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Deliberately corrupt one byte of the host output data
    // Choose index 10 and flip the least significant byte
    int corruptIdx = 10;
    unsigned char* bytePtr = (unsigned char*)(h_out + corruptIdx);
    *bytePtr ^= 0x01;   // flip a single bit

    // Verification
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        int expected = h_in[i] * 2;
        if (h_out[i] != expected) {
            if (mismatches < 10) {  // print first few mismatches
                printf("Mismatch at index %d: host=%d expected=%d\n",
                       i, h_out[i], expected);
            }
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("All data verified successfully.\n");
    } else {
        printf("Verification failed with %d mismatches.\n", mismatches);
    }

    // Clean up
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
