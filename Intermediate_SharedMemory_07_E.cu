```cuda
/*
Aim of the program:
Write a kernel to perform a bit-reversal of an array's elements, a common operation in FFTs.

Thinking and Design:
1. The goal is to permute an array of length N such that the element originally at index i
   is moved to the position whose binary representation is the bit-reversed form of i.
   This is a fundamental step in many FFT implementations.

2. Requirements:
   - N must be a power of two for a well-defined bit-reversal over log2(N) bits.
   - We will provide a host function to set up data, launch the kernel, and verify
     the result.
   - The kernel will not perform in-place swapping because that would require
     inter-thread synchronization. Instead, each thread writes its value to the
     destination index computed via bit reversal.

3. Implementation details:
   - Use CUDA's intrinsic __brev() to reverse the bits of a 32‑bit unsigned integer.
     Since we only need the lower log2(N) bits, the result is shifted right
     by (32 - log2(N)).
   - The kernel signature: 
        __global__ void bit_reverse(const unsigned int *in, unsigned int *out, unsigned int n);
     Each thread reads in[tid] and writes it to out[rev_tid].
   - Host code:
     * Allocate a host array of unsigned ints of size N and initialize it.
     * Allocate device memory for input and output.
     * Copy input to device.
     * Launch the kernel with enough blocks/threads to cover all N elements.
     * Copy output back to host.
     * Verify correctness by checking that out[bit_reverse(i)] == i.
   - Timing: Use cudaEvent_t for measuring kernel launch time.
   - For demonstration, set N = 1 << 20 (1,048,576). Print first 10 elements
     of input and output arrays to confirm the permutation.

4. Edge cases and safety:
   - If N is not a power of two, the program prints an error and exits.
   - All CUDA API calls are checked with a macro to ensure error handling.
   - The code is self‑contained; no external headers or libraries are required
     beyond the CUDA Runtime.

With this design, the program is clear, efficient, and demonstrates a standard
bit‑reversal kernel suitable for FFT pre‑processing.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper macro to check CUDA errors */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* Kernel: bit-reversal permutation.
   Each thread writes its input element to the output array at the bit-reversed index. */
__global__ void bit_reverse(const unsigned int *in, unsigned int *out, unsigned int n, unsigned int log2n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Reverse all 32 bits, then shift right to keep only the lower log2n bits.
    unsigned int rev = __brev(tid) >> (32 - log2n);
    out[rev] = in[tid];
}

/* Host function to compute log2 of a power-of-two integer. */
static unsigned int log2_pow2(unsigned int n) {
    return 31 - __builtin_clz(n);
}

/* Function to verify bit-reversal result. */
static int verify(const unsigned int *in, const unsigned int *out, unsigned int n) {
    unsigned int log2n = log2_pow2(n);
    for (unsigned int i = 0; i < n; ++i) {
        unsigned int rev = __builtin_bswap32(i) >> (32 - log2n);
        if (out[rev] != in[i]) {
            fprintf(stderr, "Verification failed at index %u: out[%u] = %u, expected %u\n",
                    rev, rev, out[rev], in[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    /* Array size: must be a power of two */
    const unsigned int N = 1 << 20;      // 1,048,576 elements
    const unsigned int log2n = log2_pow2(N);

    /* Allocate host memory */
    unsigned int *h_in  = (unsigned int*)malloc(N * sizeof(unsigned int));
    unsigned int *h_out = (unsigned int*)malloc(N * sizeof(unsigned int));
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize input array with a simple pattern (0..N-1) */
    for (unsigned int i = 0; i < N; ++i) {
        h_in[i] = i;
    }

    /* Allocate device memory */
    unsigned int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in,  N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(unsigned int)));

    /* Copy input to device */
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(unsigned int), cudaMemcpyHostToDevice));

    /* Launch kernel */
    const unsigned int threadsPerBlock = 256;
    const unsigned int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    bit_reverse<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N, log2n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    /* Measure elapsed time */
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Bit-reversal kernel execution time: %f ms\n", ms);

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    /* Verify correctness */
    if (verify(h_in, h_out, N)) {
        printf("Verification passed: bit-reversal performed correctly.\n");
    } else {
        printf("Verification failed.\n");
    }

    /* Print first 10 elements for sanity check */
    printf("First 10 elements of input and output:\n");
    for (int i = 0; i < 10; ++i) {
        unsigned int rev = __builtin_bswap32(i) >> (32 - log2n);
        printf("i=%5u  in=%8u  out[%5u]=%8u\n", i, h_in[i], rev, h_out[rev]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
```