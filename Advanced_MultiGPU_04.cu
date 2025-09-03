```cpp
/*
 * Aim: Implement a 'halo exchange' pattern between two GPUs. Each GPU works on a sub-domain of a 1D array. After a computation step, each GPU needs the boundary data from its neighbor. Use P2P `cudaMemcpy` to exchange these 'halo' regions directly.
 *
 * Thinking:
 *
 * 1. The program will target a system with at least two CUDA-capable GPUs.
 * 2. Each GPU owns a local 1‑D sub‑domain plus two halo cells (one on each side).
 *    For simplicity we allocate an array of size N+2 on each GPU where indices
 *    1..N contain the actual domain data, index 0 is the left halo and index
 *    N+1 is the right halo.
 * 3. After an initial setup where each GPU fills its own domain values,
 *    a dummy compute kernel will be launched that modifies each element.
 * 4. The halo exchange is performed using P2P `cudaMemcpyPeer` (or
 *    `cudaMemcpyPeerAsync`) in a bidirectional fashion:
 *    - GPU0 sends its right‑most domain element (index N) to GPU1's left halo
 *      (index 0).
 *    - GPU1 sends its left‑most domain element (index 1) to GPU0's right halo
 *      (index N+1).
 *    These copies are done device‑to‑device and are accelerated by P2P
 *    bandwidth.
 * 5. After the exchange we launch another compute kernel that uses the halo
 *    values (for example, computing the average of a cell and its two
 *    neighbours) to demonstrate that the exchange worked.
 * 6. Finally, the contents of each GPU's array are copied back to the host
 *    and printed for verification.
 *
 * Implementation details:
 * - Use `cudaGetDeviceCount` to ensure there are at least two GPUs.
 * - Enable peer access on both devices with `cudaDeviceEnablePeerAccess`.
 * - Use simple error checking macro to catch CUDA API errors.
 * - Keep data types simple (float).
 * - Use streams to potentially overlap compute and communication.
 *
 * Edge cases handled:
 * - The program exits gracefully if less than two GPUs are present.
 * - Peer access failures are reported and the program aborts.
 *
 * The code below is a complete, self‑contained .cu file that can be compiled
 * with `nvcc` and run on a dual‑GPU system.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Dummy compute kernel that simply increments each element by 1.0
__global__ void compute_step(float *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1 && idx <= N) { // Operate only on the domain part
        data[idx] += 1.0f;
    }
}

// Kernel that uses halo cells: average of left, center, right
__global__ void use_halo(float *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1 && idx <= N) {
        float left  = data[idx - 1];
        float right = data[idx + 1];
        data[idx] = (left + data[idx] + right) / 3.0f;
    }
}

int main(void)
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        fprintf(stderr, "This program requires at least 2 GPUs.\n");
        return EXIT_FAILURE;
    }

    const int N = 10;          // Size of each local domain
    const int totalSize = N + 2; // Including halo cells

    // Pointers for device memory on each GPU
    float *d_data[2];
    cudaStream_t stream[2];

    // Allocate memory and set up on each GPU
    for (int dev = 0; dev < 2; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaMalloc(&d_data[dev], totalSize * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_data[dev], 0, totalSize * sizeof(float)));

        // Enable peer access
        for (int peer = 0; peer < 2; ++peer) {
            if (peer == dev) continue;
            int canAccessPeer = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, dev, peer));
            if (canAccessPeer) {
                CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
            } else {
                fprintf(stderr, "Device %d cannot access device %d\n", dev, peer);
                return EXIT_FAILURE;
            }
        }

        // Create a stream for asynchronous operations
        CUDA_CHECK(cudaStreamCreate(&stream[dev]));

        // Initialize domain values: each GPU sets its domain to a unique number
        // Copy host data to device
        float *h_init = (float*)malloc(totalSize * sizeof(float));
        for (int i = 0; i < totalSize; ++i) {
            h_init[i] = (float)(dev * 1000 + i); // unique pattern
        }
        CUDA_CHECK(cudaMemcpyAsync(d_data[dev], h_init, totalSize * sizeof(float),
                                   cudaMemcpyHostToDevice, stream[dev]));
        CUDA_CHECK(cudaStreamSynchronize(stream[dev]));
        free(h_init);
    }

    // Perform a dummy compute step on each GPU
    for (int dev = 0; dev < 2; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        compute_step<<<blocks, threadsPerBlock, 0, stream[dev]>>>(d_data[dev], N);
    }

    // Halo exchange using P2P memcpy
    // GPU0 -> GPU1: send rightmost domain element (index N) to GPU1's left halo (index 0)
    // GPU1 -> GPU0: send leftmost domain element (index 1) to GPU0's right halo (index N+1)
    {
        // GPU0 to GPU1
        CUDA_CHECK(cudaMemcpyPeerAsync(
            d_data[1], 1,          // destination pointer, dest device
            d_data[0] + N, 0,      // source pointer, src device
            sizeof(float),         // size
            stream[0]              // use stream[0] for source device
        ));

        // GPU1 to GPU0
        CUDA_CHECK(cudaMemcpyPeerAsync(
            d_data[0] + N + 1, 0,  // dest pointer, dest device
            d_data[1] + 1, 1,      // src pointer, src device
            sizeof(float),         // size
            stream[1]              // use stream[1] for source device
        ));
    }

    // Wait for all streams to finish
    for (int dev = 0; dev < 2; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaStreamSynchronize(stream[dev]));
    }

    // Use halo cells in a second compute step
    for (int dev = 0; dev < 2; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        use_halo<<<blocks, threadsPerBlock, 0, stream[dev]>>>(d_data[dev], N);
    }

    // Copy results back to host and print
    for (int dev = 0; dev < 2; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaStreamSynchronize(stream[dev]));

        float *h_result = (float*)malloc(totalSize * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_result, d_data[dev], totalSize * sizeof(float),
                              cudaMemcpyDeviceToHost));

        printf("Device %d final array (including halos):\n", dev);
        for (int i = 0; i < totalSize; ++i) {
            printf("  [%2d] = %f\n", i, h_result[i]);
        }
        printf("\n");
        free(h_result);
    }

    // Cleanup
    for (int dev = 0; dev < 2; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaStreamDestroy(stream[dev]));
        CUDA_CHECK(cudaFree(d_data[dev]));
    }

    return 0;
}
```