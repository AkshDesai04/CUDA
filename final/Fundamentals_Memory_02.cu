```cuda
/*
Aim: Extend the previous program. After copying data to the device, create a second host array of 256 integers. Copy the data from the device array back to this new host array using cudaMemcpy with the cudaMemcpyDeviceToHost kind. Verify that every element in the new host array matches the original.

Thinking:
1. Allocate a host array `h_src` of 256 integers and initialize it with known values (e.g., 0..255).
2. Allocate device memory `d_data` of size 256 * sizeof(int).
3. Copy `h_src` to `d_data` using cudaMemcpy with cudaMemcpyHostToDevice.
4. Allocate a second host array `h_dst` of 256 integers to receive the copy back.
5. Copy from `d_data` to `h_dst` using cudaMemcpy with cudaMemcpyDeviceToHost.
6. Verify that each element of `h_dst` matches the corresponding element in `h_src`.
   - If all match, print "Verification passed".
   - Otherwise, print the index of the first mismatch and its values.
7. Clean up: free device memory and exit.
8. Include basic CUDA error checking after each CUDA API call.
9. Keep the code simple and self‑contained, using only standard headers and CUDA runtime API.

This file can be compiled with `nvcc`:
    nvcc -o verify copy_verify.cu
and run with `./verify`.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in function '" << __FUNCTION__            \
                      << "' at line " << __LINE__ << ": "                      \
                      << cudaGetErrorString(err) << std::endl;                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

int main() {
    const int N = 256;
    int h_src[N];
    int h_dst[N];

    // Initialize source array with values 0..255
    for (int i = 0; i < N; ++i) {
        h_src[i] = i;
    }

    // Allocate device memory
    int *d_data = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_data, N * sizeof(int)));

    // Copy from host to device
    CHECK_CUDA(cudaMemcpy(d_data, h_src, N * sizeof(int), cudaMemcpyHostToDevice));

    // Copy back from device to second host array
    CHECK_CUDA(cudaMemcpy(h_dst, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify that each element matches
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_dst[i] != h_src[i]) {
            std::cerr << "Mismatch at index " << i
                      << ": src=" << h_src[i]
                      << " dst=" << h_dst[i] << std::endl;
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Verification passed: all elements match." << std::endl;
    } else {
        std::cerr << "Verification failed." << std::endl;
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_data));

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
```