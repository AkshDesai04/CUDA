/*
Launch with one block of N threads. Pass an integer array `int* out` to the kernel. Have each thread write its `threadIdx.x` to `out[threadIdx.x]`.

Thinking:
- The kernel will be very simple: each thread will store its own thread index in the output array.
- We want to launch a single block containing N threads, so we can set the grid dimension to 1 and block dimension to N.
- The kernel signature will accept a device pointer `int* out`. Since the kernel runs on the device, the pointer must be a device pointer; we will allocate it on the device in the host code.
- After launching the kernel, we need to copy the result back to the host for verification or printing.
- Error checking is important: each CUDA API call will be followed by a check of the returned status. A helper macro or function can wrap this to keep the code clean.
- For demonstration purposes we’ll define N as a compile-time constant, but it could also be passed as a runtime argument.
- Finally, we will print the contents of the array on the host to confirm that each element contains its index.
*/

#include <iostream>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
                      << ", " << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that writes threadIdx.x into the output array
__global__ void write_thread_ids(int* out)
{
    // Each thread writes its own index to the corresponding position
    out[threadIdx.x] = threadIdx.x;
}

int main()
{
    const int N = 256;  // Number of threads in the single block

    // Allocate host array
    int* h_out = new int[N];

    // Allocate device array
    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(int)));

    // Launch kernel with one block of N threads
    write_thread_ids<<<1, N>>>(d_out);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print the results
    std::cout << "Thread IDs written to out array:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "out[" << i << "] = " << h_out[i] << std::endl;
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_out;

    return 0;
}
