```cpp
/*
Aim: Write a kernel that swaps the values between two arrays: `int temp = a[i]; a[i] = b[i]; b[i] = temp;`.

Thinking:
1. The problem asks for a CUDA kernel that swaps elements of two integer arrays element‑by‑element.  
2. The kernel will run with one thread per element.  
3. To be safe, we will add a boundary check `if (i < n)` so that the kernel can be launched with a number of threads that may exceed the array size.  
4. The kernel itself is straightforward: store `a[i]` in a temporary, assign `b[i]` to `a[i]`, then assign the temp to `b[i]`.  
5. For a complete self‑contained example, the host code will:  
   - Allocate host arrays `h_a` and `h_b` with some test data.  
   - Allocate device memory for `d_a` and `d_b`.  
   - Copy the host arrays to the device.  
   - Compute grid/block dimensions.  
   - Launch the kernel.  
   - Copy the results back to the host.  
   - Print the arrays before and after swapping.  
6. Basic error checking is added via a helper macro `CUDA_CHECK`.  
7. All necessary includes are added (`cuda_runtime.h`, `iostream`, `iomanip`).  
8. The program is self‑contained; compiling with `nvcc swap_arrays.cu -o swap_arrays` and running `./swap_arrays` will demonstrate the swap.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error at " << __FILE__             \
                      << ":" << __LINE__ << " - "                 \
                      << cudaGetErrorString(err) << std::endl;    \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel that swaps elements of two integer arrays
__global__ void swapArrays(int *a, int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int temp = a[i];
        a[i] = b[i];
        b[i] = temp;
    }
}

int main() {
    const int N = 16;                      // Size of arrays
    size_t bytes = N * sizeof(int);

    // Allocate host arrays
    int h_a[N], h_b[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;          // 0, 1, 2, ...
        h_b[i] = N - i - 1;  // 15, 14, 13, ...
    }

    // Print original arrays
    std::cout << "Before swap:\n";
    std::cout << "h_a: ";
    for (int i = 0; i < N; ++i) std::cout << std::setw(3) << h_a[i] << " ";
    std::cout << "\nh_b: ";
    for (int i = 0; i < N; ++i) std::cout << std::setw(3) << h_b[i] << " ";
    std::cout << "\n\n";

    // Allocate device memory
    int *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));

    // Copy host arrays to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    swapArrays<<<blocks, threadsPerBlock>>>(d_a, d_b, N);
    CUDA_CHECK(cudaGetLastError());      // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure completion

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    // Print swapped arrays
    std::cout << "After swap:\n";
    std::cout << "h_a: ";
    for (int i = 0; i < N; ++i) std::cout << std::setw(3) << h_a[i] << " ";
    std::cout << "\nh_b: ";
    for (int i = 0; i < N; ++i) std::cout << std::setw(3) << h_b[i] << " ";
    std::cout << "\n";

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return 0;
}
```