```cpp
/*
Aim of the program:
Write a kernel to perform a bit-reversal of an array's elements, a common operation in FFTs.

Thinking and design:
1. The bit-reversal operation is commonly used in Fast Fourier Transforms (FFTs) to reorder data
   before or after the butterfly computations.  For a power‑of‑two sized array the operation can be
   expressed as mapping each index i to its bit‑reversed counterpart rev(i).  The kernel will
   perform this mapping in parallel.

2. Since we want a simple and general example, the kernel will read from an input array `in` and
   write the bit‑reversed data to an output array `out`.  Using a separate output buffer avoids
   race conditions because multiple threads could try to write to the same location if we tried
   to do an in‑place swap.  For a power‑of‑two sized array this method is O(n) memory and O(n)
   time, which is acceptable for demonstration purposes.

3. The host code will:
   - Allocate and initialize a host array `h_in` with consecutive integers.
   - Allocate device buffers `d_in` and `d_out`.
   - Compute the number of bits required to represent the indices (`bits = ceil(log2(n))`).
   - Launch the kernel with enough threads to cover all elements.
   - Copy the result back to the host and display the reordered array.

4. The bit reversal logic:
   ```
   rev = 0;
   for (int j = 0; j < bits; ++j)
       rev = (rev << 1) | (i & 1), i >>= 1;
   ```
   This loops over the bits of the original index `i`, shifting `rev` left and appending the
   least‑significant bit of `i`.

5. Error checking is performed via a simple macro `CUDA_CHECK` that wraps CUDA API calls.

6. The code is written for CUDA C++ (.cu) and can be compiled with `nvcc`.  It is kept self‑contained
   and does not rely on any external libraries beyond the CUDA runtime and standard C++ headers.

*/

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__        \
                      << " - " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel: perform bit‑reversal of indices
__global__ void bit_reverse_kernel(const float *in, float *out, int n, int bits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Compute bit‑reversed index
    int rev = 0;
    int temp = idx;
    for (int j = 0; j < bits; ++j) {
        rev = (rev << 1) | (temp & 1);
        temp >>= 1;
    }

    out[rev] = in[idx];
}

// Host function to compute number of bits needed to represent indices up to n-1
int compute_bits(int n)
{
    int bits = 0;
    int temp = n - 1;
    while (temp > 0) {
        ++bits;
        temp >>= 1;
    }
    return bits;
}

int main()
{
    const int n = 16; // example size (must be a power of two)
    const int bits = compute_bits(n);

    // Allocate and initialize host array
    float *h_in = new float[n];
    float *h_out = new float[n];
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<float>(i);

    // Allocate device memory
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, n * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    bit_reverse_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n, bits);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Display results
    std::cout << "Input array:" << std::endl;
    for (int i = 0; i < n; ++i) std::cout << h_in[i] << " ";
    std::cout << std::endl << "Bit‑reversed output:" << std::endl;
    for (int i = 0; i < n; ++i) std::cout << h_out[i] << " ";
    std::cout << std::endl;

    // Clean up
    delete[] h_in;
    delete[] h_out;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
```