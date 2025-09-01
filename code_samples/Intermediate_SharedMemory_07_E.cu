```cpp
/*
Aim of the program:
Write a kernel to perform a bit-reversal of an array's elements, a common operation in FFTs.

Thinking:
To perform a bit-reversal permutation of an array, we first need to determine the number of bits required to represent indices up to the array length N. If N is a power of two, this number is simply log2(N). Each index i (0 ≤ i < N) will be reversed bitwise to produce a new index rev(i). The simplest CUDA implementation allocates an output array on the device; each thread reads its input element at index i, computes rev(i), and writes the element into the output array at position rev(i). After kernel execution, we copy the output array back to the host (or swap pointers). The implementation uses a small inline device function to reverse bits, iterating over the number of bits and building the reversed value. For simplicity and clarity, the host code sets N to 16 (a power of two), initializes the input array with sequential integers, calls the kernel, and prints the result. Error checking macros are included for robustness.

The program demonstrates:
- Computing log2(N) on the host.
- Copying data between host and device.
- Launching a CUDA kernel with one thread per array element.
- Performing the bit-reversal permutation in parallel.
- Retrieving and displaying the reordered array.

Compile with:
    nvcc -arch=sm_61 bit_reversal.cu -o bit_reversal
Run:
    ./bit_reversal
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                        \
    {                                                                           \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }

// Device function to reverse bits of an integer up to a given number of bits
__device__ __forceinline__ unsigned int reverse_bits(unsigned int x, int bits) {
    unsigned int rev = 0;
    for (int i = 0; i < bits; ++i) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }
    return rev;
}

// Kernel to perform bit-reversal permutation
__global__ void bit_reversal_kernel(const int* d_in, int* d_out, int n, int log2n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int rev = reverse_bits((unsigned int)idx, log2n);
        d_out[rev] = d_in[idx];
    }
}

// Host function to compute log2 of N (assumes N is power of two)
int log2_int(int n) {
    int log2n = 0;
    while ((1 << log2n) < n) {
        ++log2n;
    }
    return log2n;
}

int main() {
    const int N = 16;  // Array size (must be power of two)
    const int log2N = log2_int(N);

    // Allocate host memory
    int* h_in = (int*)malloc(N * sizeof(int));
    int* h_out = (int*)malloc(N * sizeof(int));
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input array with sequential values
    printf("Input array:\n");
    for (int i = 0; i < N; ++i) {
        h_in[i] = i;
        printf("%d ", h_in[i]);
    }
    printf("\n");

    // Allocate device memory
    int *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(int)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    bit_reversal_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N, log2N);
    CUDA_CHECK(cudaGetLastError());

    // Copy output data back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Display the bit-reversed array
    printf("Bit-reversed array:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return EXIT_SUCCESS;
}
```