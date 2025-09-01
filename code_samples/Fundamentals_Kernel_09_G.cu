/*
Create a __device__ function that is recursive, like a factorial function. Try calling it from a kernel. Be aware of stack limitations.

Thinking:
- CUDA supports recursive device functions starting with compute capability 2.0, but recursion consumes the per-thread stack.
- The default per-thread stack size is limited (e.g., 1024 bytes). For deep recursion (like factorial of a large number) we might hit stack overflow.
- We can adjust the stack size with cudaDeviceSetLimit(cudaLimitStackSize, value) before kernel launch.
- The factorial function is simple: factorial(n) = n * factorial(n-1) with base case factorial(0) = factorial(1) = 1.
- In this example we will compute factorials for numbers 0..10, which requires at most depth 10, well within limits.
- The kernel will launch one thread per number; each thread calls the recursive device function and stores the result in an array.
- Results are copied back to host and printed.
- Use unsigned long long to hold factorial values up to 20! safely.
- Error checking for CUDA API calls is omitted for brevity but should be added in production code.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Recursive factorial function running on the device.
// The recursion depth is limited by the per-thread stack size.
// For small n (e.g., n <= 20) the default stack size is sufficient.
__device__ unsigned long long factorial(unsigned int n) {
    if (n <= 1) {
        return 1ULL;
    } else {
        return static_cast<unsigned long long>(n) * factorial(n - 1);
    }
}

// Kernel that computes factorial for each thread based on its thread index.
__global__ void compute_factorials(unsigned int max_n, unsigned long long* results) {
    unsigned int idx = threadIdx.x;
    if (idx <= max_n) {
        results[idx] = factorial(idx);
    }
}

int main() {
    const unsigned int MAX_N = 10;                // Compute factorials up to 10!
    const size_t ARRAY_SIZE = MAX_N + 1;          // 0 through 10 inclusive

    // Allocate host array for results
    unsigned long long h_results[ARRAY_SIZE];

    // Allocate device array
    unsigned long long* d_results;
    cudaMalloc((void**)&d_results, ARRAY_SIZE * sizeof(unsigned long long));

    // Optional: increase device stack size to avoid overflow for deeper recursion.
    // For this example depth is small, but setting demonstrates awareness.
    size_t stackSize = 2048; // bytes
    cudaDeviceSetLimit(cudaLimitStackSize, stackSize);

    // Launch kernel with one thread per number 0..MAX_N
    compute_factorials<<<1, ARRAY_SIZE>>>(MAX_N, d_results);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_results, d_results, ARRAY_SIZE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Print the factorials
    printf("Factorial results (using recursive device function):\n");
    for (unsigned int i = 0; i <= MAX_N; ++i) {
        printf("%u! = %llu\n", i, h_results[i]);
    }

    // Clean up
    cudaFree(d_results);
    return 0;
}
