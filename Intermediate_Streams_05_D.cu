```cpp
/*
Aim: Pass a pointer to a struct or class as the `user_data` to give the callback function context about the work that was completed.

Thinking:
- CUDA streams allow registering a host callback that runs once all prior work in the stream has finished.
- The callback signature is `void callback(cudaStream_t stream, cudaError_t status, void *userData)`.
- By passing a pointer to a user-defined struct as `userData`, the callback can access contextual information (e.g., the result of the kernel or a descriptive string).
- The program will:
  1. Define a simple struct `WorkContext` containing an integer result and a description string.
  2. Launch a trivial kernel that writes a value to device memory.
  3. Use `cudaLaunchHostFunc` to enqueue a host callback that receives the `WorkContext` pointer and prints its contents.
  4. Wait for stream completion and clean up.
- Error handling macros (`CUDA_CHECK`) are used for brevity.
*/

#include <iostream>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__  \
                      << " (" << cudaGetErrorString(err) << ")" << std::endl; \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Struct that will be passed as user data to the callback
struct WorkContext {
    int result;             // Result produced by the kernel
    const char* desc;       // Description of the work
};

// Device kernel: trivial addition
__global__ void addKernel(int a, int b, int* out) {
    int idx = threadIdx.x; // Only one thread needed
    if (idx == 0) {
        *out = a + b;
    }
}

// Host callback function
void CUDART_CB myCallback(cudaStream_t stream, cudaError_t status, void* userData) {
    WorkContext* ctx = static_cast<WorkContext*>(userData);
    if (status != cudaSuccess) {
        std::cerr << "Stream error in callback: " << cudaGetErrorString(status) << std::endl;
        return;
    }
    std::cout << "Callback invoked after kernel completion." << std::endl;
    std::cout << "Work description: " << ctx->desc << std::endl;
    std::cout << "Kernel result: " << ctx->result << std::endl;
}

int main() {
    // Allocate device memory for result
    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    // Create a stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Launch kernel in the stream
    addKernel<<<1, 1, 0, stream>>>(3, 5, d_result);

    // Prepare context to pass to callback
    WorkContext ctx;
    ctx.result = 0; // placeholder; will be set after copy back
    ctx.desc = "Adding 3 and 5";

    // Register the callback with the stream, passing the address of ctx
    CUDA_CHECK(cudaLaunchHostFunc(stream, myCallback, &ctx));

    // Copy the result back to host before waiting for callback
    CUDA_CHECK(cudaMemcpyAsync(&ctx.result, d_result, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));

    // Wait for all operations in the stream to complete
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Cleanup
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}
```