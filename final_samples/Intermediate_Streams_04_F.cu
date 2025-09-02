```cpp
/*
What is the difference between `cudaStreamWaitEvent` and `cudaEventSynchronize`? (The former is a non-blocking device-side dependency, the latter is a blocking host-side wait).

My thinking:
- `cudaStreamWaitEvent` establishes a dependency on a CUDA event *inside the device*.  It tells the GPU that a stream must not start executing its subsequent work until the event has been recorded by another stream.  This does not block the CPU; the host can continue execution immediately after calling the API.
- `cudaEventSynchronize` blocks the *host* until the specified event has been completed by the GPU.  It forces the CPU to wait, ensuring that any GPU work that recorded the event has finished before the host proceeds.

To demonstrate this difference, I will:
1. Launch a simple kernel (`kernelA`) on `stream0` that writes data to a device array.
2. Record an event (`eventA`) after `kernelA` completes.
3. Launch another kernel (`kernelB`) on `stream1` that depends on `eventA` by calling `cudaStreamWaitEvent`.  This ensures `kernelB` runs only after `kernelA` finishes, but the host is not blocked.
4. While `kernelB` is queued, the host will perform a dummy loop to show it is still running concurrently.
5. Finally, call `cudaEventSynchronize(eventA)` to block the host until the event (i.e., `kernelA`) is complete, and then copy the results back to the host and display them.

The code includes simple error checking, timing prints, and comments that highlight where the blocking/non‑blocking behavior occurs.  The program can be compiled with `nvcc` and run on a CUDA‑capable GPU.

*/

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__      \
                      << " (" << cudaGetErrorString(err) << ")\n";           \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// Simple kernel that increments each element by 1
__global__ void kernelA(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] += 1;
    }
}

// Kernel that doubles each element
__global__ void kernelB(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] *= 2;
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(int);

    // Host array
    int *h_arr = new int[N];
    for (int i = 0; i < N; ++i) h_arr[i] = i;

    // Device array
    int *d_arr;
    CHECK_CUDA(cudaMalloc(&d_arr, bytes));
    CHECK_CUDA(cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice));

    // Create streams
    cudaStream_t stream0, stream1;
    CHECK_CUDA(cudaStreamCreate(&stream0));
    CHECK_CUDA(cudaStreamCreate(&stream1));

    // Create event
    cudaEvent_t eventA;
    CHECK_CUDA(cudaEventCreate(&eventA));

    // Launch kernelA on stream0
    dim3 blk(256), grd((N + blk.x - 1) / blk.x);
    std::cout << "Launching kernelA on stream0\n";
    kernelA<<<grd, blk, 0, stream0>>>(d_arr, N);
    CHECK_CUDA(cudaGetLastError());

    // Record event after kernelA
    std::cout << "Recording eventA on stream0 after kernelA\n";
    CHECK_CUDA(cudaEventRecord(eventA, stream0));

    // Set stream1 to wait for eventA
    std::cout << "Setting stream1 to wait for eventA (non‑blocking)\n";
    CHECK_CUDA(cudaStreamWaitEvent(stream1, eventA, 0));

    // Launch kernelB on stream1
    std::cout << "Launching kernelB on stream1 (will wait for eventA)\n";
    kernelB<<<grd, blk, 0, stream1>>>(d_arr, N);
    CHECK_CUDA(cudaGetLastError());

    // Host continues doing work (non‑blocking)
    std::cout << "Host continues while kernels run\n";
    for (volatile int i = 0; i < 100000000; ++i) {
        // Dummy workload
    }
    std::cout << "Host finished dummy workload\n";

    // Now we block host until eventA is complete
    std::cout << "Host will block until eventA completes (cudaEventSynchronize)\n";
    auto start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaEventSynchronize(eventA));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "cudaEventSynchronize returned after " << diff.count() << " seconds\n";

    // Copy back results
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost));

    // Verify a few values
    std::cout << "Sample results:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "h_arr[" << i << "] = " << h_arr[i] << "\n";
    }

    // Clean up
    CHECK_CUDA(cudaEventDestroy(eventA));
    CHECK_CUDA(cudaStreamDestroy(stream0));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaFree(d_arr));
    delete[] h_arr;

    return 0;
}
```