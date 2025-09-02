/*
Aim: Write a program with a 'callback'. Use `cudaLaunchHostFunc` to queue a CPU function to be executed after all preceding work in a stream is complete. The CPU function can, for example, print a message indicating a chunk is done.

Thinking:
The goal is to demonstrate the use of cudaLaunchHostFunc which allows a CPU function to be scheduled in a CUDA stream, ensuring it runs only after all previous GPU work in that stream has finished. 
To illustrate this, we will:
1. Allocate an array on both host and device.
2. Initialize the host array.
3. Copy it to the device.
4. Launch a simple kernel that increments each element in the array. This kernel will be launched in a dedicated CUDA stream.
5. Immediately after the kernel launch, we will queue a host callback using cudaLaunchHostFunc in the same stream. The callback will simply print a message indicating that the GPU chunk of work has completed.
6. Finally, we will synchronize the stream (or the device), copy the results back to host, and verify that the kernel ran by printing a few output values.

We will use the CUDA Runtime API. Error checking will be performed via a helper macro. The callback function signature must match void(*)(void*), so we can pass a context pointer if needed. In this simple example, we’ll just print a static message. The code will be fully self-contained in a single .cu file. The user can compile with `nvcc` and run to see the callback execution.
*/

#include <iostream>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error in " << __FILE__ << ":"      \
                      << __LINE__ << " : " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                              \
        }                                                          \
    } while (0)

// Simple kernel that increments each element of an array
__global__ void incrementKernel(int *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1;
    }
}

// Host callback function to be called after GPU work in the stream completes
void __host__ hostCallback(void *userData)
{
    const char *msg = static_cast<const char*>(userData);
    std::cout << "Callback executed: " << msg << std::endl;
}

int main()
{
    const int N = 1 << 20;            // 1M elements
    const size_t bytes = N * sizeof(int);

    // Allocate host memory
    int *h_data = new int[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }

    // Allocate device memory
    int *d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));

    // Create a CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Copy data from host to device asynchronously
    CHECK_CUDA(cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream));

    // Define kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel in the stream
    incrementKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N);
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "Kernel launch error." << std::endl;
        return EXIT_FAILURE;
    }

    // Prepare a message to pass to the callback
    const char *callbackMsg = "GPU chunk completed";

    // Queue the host callback in the same stream
    CHECK_CUDA(cudaLaunchHostFunc(stream, hostCallback, (void*)callbackMsg));

    // Copy result back to host asynchronously
    CHECK_CUDA(cudaMemcpyAsync(h_data, d_data, bytes, cudaMemcpyDeviceToHost, stream));

    // Synchronize the stream to ensure all work (kernel + callback + copy) is finished
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Verify a few results
    std::cout << "Sample output after kernel and callback:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "h_data[" << i << "] = " << h_data[i] << std::endl;
    }

    // Clean up
    delete[] h_data;
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaStreamDestroy(stream));

    std::cout << "Program completed successfully." << std::endl;
    return 0;
}
