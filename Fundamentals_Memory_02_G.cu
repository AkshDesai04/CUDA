```cpp
/*
Perform a round trip: HtoD, then DtoH. Time the entire process.

Thinking:
- The goal is to measure the total time taken to copy data from the host
  to the device (HtoD) and back from the device to the host (DtoH) using
  CUDA's memory copy API.
- Use cudaEvent_t for precise GPU-side timing since it captures the
  time spent on the device and the asynchronous memory copies.
- Allocate a moderate sized array (e.g., 1 million floats) to make the
  copy operation noticeable while keeping the example simple.
- The host array is initialized with some values; the actual content
  doesn't matter for timing.
- Error checking is essential: each CUDA API call should be verified,
  and any error should terminate the program with an informative
  message.
- After timing, print the elapsed milliseconds.
- Finally, clean up all allocated memory and CUDA resources.

The program is self-contained and can be compiled with `nvcc round_trip.cu -o round_trip`
and executed on a machine with a CUDA-capable GPU.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line "   \
                      << __LINE__ << ": " << cudaGetErrorString(err) << '\n'; \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

int main() {
    const size_t N = 1 << 20; // 1,048,576 elements
    const size_t bytes = N * sizeof(float);

    // Allocate and initialize host memory
    float* h_ptr = (float*)malloc(bytes);
    if (!h_ptr) {
        std::cerr << "Host memory allocation failed.\n";
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < N; ++i) {
        h_ptr[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_ptr, bytes));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // HtoD copy
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));

    // DtoH copy
    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost));

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for the events to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "Round trip (HtoD + DtoH) time: " << ms << " ms\n";

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_ptr));
    free(h_ptr);

    return 0;
}
```