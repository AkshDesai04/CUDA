```cpp
/*
Aim of the program (word for word as requested):
Allocate `d_a` and `d_b`. Use `cudaMemcpy(d_b, h_a, size, cudaMemcpyHostToDevice)` to initialize `d_a`. What is the error? Change `h_a` to `d_a`. What is the error? Explain why the kind parameter is critical.

Thinking:
1. We allocate two device pointers `d_a` and `d_b`. We also allocate a host array `h_a` and fill it with some data.
2. The first copy uses `cudaMemcpy(d_b, h_a, size, cudaMemcpyHostToDevice)`. This copies the host data into `d_b`. However, the comment says we want to initialize `d_a`. This is a logical mistake: the destination pointer is wrong. The code compiles fine, but `d_a` remains uninitialized, so any later use of `d_a` will give garbage or cause undefined behaviour. This is not a CUDA runtime error, just a bug in our program logic.
3. The second copy changes the source pointer to `d_a`: `cudaMemcpy(d_b, d_a, size, cudaMemcpyHostToDevice)`. Now both source and destination are device pointers, but we still specify the kind `cudaMemcpyHostToDevice`. The CUDA runtime checks the kind against the pointer types. Because we are copying device → device but told it to copy host → device, the runtime returns `cudaErrorInvalidMemcpyDirection`. This error can be retrieved via `cudaGetLastError()` or `cudaMemcpy`’s return value. This demonstrates that the `kind` parameter is critical; it tells the runtime whether the pointers refer to host or device memory and determines the allowed memory transfer directions. Using the wrong kind causes a runtime error because the runtime cannot safely perform the transfer.

The program below demonstrates both mistakes and prints the corresponding error codes.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    const size_t N = 10;
    const size_t size = N * sizeof(int);

    // Host array
    int h_a[N];
    for (int i = 0; i < N; ++i) h_a[i] = i;

    // Device pointers
    int *d_a = nullptr;
    int *d_b = nullptr;

    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc(&d_a, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc for d_a failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    err = cudaMalloc(&d_b, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc for d_b failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a);
        return -1;
    }

    std::cout << "=== First copy (wrong destination) ===" << std::endl;
    // This copies host data into d_b, but we intended to copy into d_a.
    err = cudaMemcpy(d_b, h_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "cudaMemcpy succeeded (but wrong destination!)." << std::endl;
    }

    // Attempt to use d_a (uninitialized) would be a logical bug here.

    std::cout << "=== Second copy (wrong kind) ===" << std::endl;
    // Now we mistakenly try to copy from d_a (device) to d_b (device) but still use HostToDevice kind.
    err = cudaMemcpy(d_b, d_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed as expected: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "cudaMemcpy succeeded (unexpected)." << std::endl;
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
```