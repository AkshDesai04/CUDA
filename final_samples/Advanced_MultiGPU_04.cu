```cpp
/*
   Aim of the program:
   Implement a 'halo exchange' pattern between two GPUs. Each GPU works on a sub-domain of a 1D array. After a computation step, each GPU needs the boundary data from its neighbor. Use P2P `cudaMemcpyPeer` to exchange these 'halo' regions directly.

   Thought process and design choices:
   1. Detect that there are at least two GPUs available; otherwise the program exits.
   2. Allocate a 1D array of size N (here we use 1 << 20 = 1048576 elements) on each GPU.  
      Each GPU gets a contiguous half of the array: GPU0 owns indices [0, N/2-1], GPU1 owns [N/2, N-1].
   3. Perform a trivial computation on each GPU (increment each element by 1).  
      In a real application this would be replaced by the domain-specific kernel.
   4. After computation, each GPU must send its rightmost element to the left neighbor and its leftmost element to the right neighbor.  
      Since we only have two GPUs, each GPU exchanges its single boundary element with the other GPU.
   5. Use `cudaMemcpyPeer` to perform peer-to-peer copies of these single-element halo regions.  
      Peer-to-peer access must be enabled via `cudaDeviceEnablePeerAccess`.  
      We perform a check to ensure the GPUs support P2P and are compatible.
   6. After the exchange, each GPU holds the halo element from its neighbor at the appropriate location:
      - GPU0 now has GPU1's leftmost element stored in a dedicated halo buffer.
      - GPU1 now has GPU0's rightmost element stored in its halo buffer.
   7. For demonstration purposes, each GPU then computes the sum of its local elements plus the received halo element and prints the result.  
      This illustrates that the exchange took place correctly.
   8. Error handling: all CUDA API calls are wrapped in a macro `CHECK_CUDA` that aborts on failure and prints an informative message.

   Key CUDA functions used:
   - `cudaGetDeviceCount` to verify at least two GPUs.
   - `cudaSetDevice` to select the current GPU.
   - `cudaMalloc` / `cudaFree` for memory allocation.
   - `cudaMemcpy` for initializing arrays on the device.
   - `cudaMemcpyPeer` for halo exchange.
   - `cudaDeviceEnablePeerAccess` to allow P2P.
   - `cudaMemcpyToSymbol` / `cudaMemcpyFromSymbol` are not needed here.
   - Simple kernels (`incrementKernel` and `sumKernel`) demonstrate computation.

   Notes:
   - The program assumes compute capability >= 2.0 for P2P support.
   - Only the host code is written; the kernels are trivial for illustration.
   - The halo buffers are of type `float` and hold a single element each.
   - This program can be compiled with:
        nvcc -arch=sm_60 -o halo_exchange halo_exchange.cu
   - It will run on systems with at least two GPUs that support P2P.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Macro for checking CUDA errors
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Simple kernel that increments each element by 1.0f
__global__ void incrementKernel(float *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        data[idx] += 1.0f;
}

// Kernel to sum local data plus halo value
__global__ void sumKernel(const float *data, int N, float halo, float *result)
{
    float sum = halo; // include halo
    for (int i = 0; i < N; ++i)
        sum += data[i];
    *result = sum;
}

int main(void)
{
    const int deviceCount = 2;
    int N = 1 << 20; // total elements (must be even)
    int localN = N / deviceCount; // elements per GPU

    // Ensure at least two GPUs
    int availableDevices;
    CHECK_CUDA(cudaGetDeviceCount(&availableDevices));
    if (availableDevices < deviceCount) {
        fprintf(stderr, "Error: Need at least %d GPUs.\n", deviceCount);
        return EXIT_FAILURE;
    }

    // Host buffers for results
    float hostResult[deviceCount];

    // Allocate memory on each GPU
    float *d_data[deviceCount];
    float *d_halo[deviceCount]; // each GPU stores the halo from its neighbor
    for (int d = 0; d < deviceCount; ++d) {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaMalloc(&d_data[d], localN * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_halo[d], sizeof(float))); // single element halo
        // Initialize local data to zero
        CHECK_CUDA(cudaMemset(d_data[d], 0, localN * sizeof(float)));
    }

    // Enable peer access between GPU 0 and GPU 1
    for (int src = 0; src < deviceCount; ++src) {
        for (int dst = 0; dst < deviceCount; ++dst) {
            if (src != dst) {
                CHECK_CUDA(cudaSetDevice(src));
                int canAccessPeer = 0;
                CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer, src, dst));
                if (canAccessPeer) {
                    cudaError_t peerErr = cudaDeviceEnablePeerAccess(dst, 0);
                    if (peerErr == cudaErrorPeerAccessAlreadyEnabled)
                        peerErr = cudaSuccess;
                    CHECK_CUDA(peerErr);
                } else {
                    fprintf(stderr, "Device %d cannot access device %d.\n", src, dst);
                    return EXIT_FAILURE;
                }
            }
        }
    }

    // Launch computation on each GPU
    dim3 blockSize(256);
    dim3 gridSize((localN + blockSize.x - 1) / blockSize.x);
    for (int d = 0; d < deviceCount; ++d) {
        CHECK_CUDA(cudaSetDevice(d));
        incrementKernel<<<gridSize, blockSize>>>(d_data[d], localN);
        CHECK_CUDA(cudaGetLastError());
    }

    // Halo exchange: each GPU copies its rightmost element to the left neighbor's halo buffer
    // GPU 0 copies its last element to GPU 1's halo
    // GPU 1 copies its first element to GPU 0's halo
    // We use cudaMemcpyPeer to perform P2P copy of a single float

    // GPU 0 -> GPU 1
    CHECK_CUDA(cudaMemcpyPeerAsync(
        d_halo[1], 1,          // destination device and ptr
        d_data[0] + localN - 1, // source ptr
        0,                      // source device
        sizeof(float),
        0                       // stream 0
    ));

    // GPU 1 -> GPU 0
    CHECK_CUDA(cudaMemcpyPeerAsync(
        d_halo[0], 0,
        d_data[1],          // source ptr (first element)
        1,
        sizeof(float),
        0
    ));

    // Ensure all copies are finished
    for (int d = 0; d < deviceCount; ++d) {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Now each GPU can use the received halo value in further computations.
    // For demonstration, compute the sum of local data + halo and copy back to host.

    float *d_sum[deviceCount];
    for (int d = 0; d < deviceCount; ++d) {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaMalloc(&d_sum[d], sizeof(float)));
        sumKernel<<<1,1>>>(d_data[d], localN, *d_halo[d], d_sum[d]);
        CHECK_CUDA(cudaGetLastError());
    }

    // Copy results back to host
    for (int d = 0; d < deviceCount; ++d) {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaMemcpy(&hostResult[d], d_sum[d], sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Print results
    for (int d = 0; d < deviceCount; ++d) {
        printf("GPU %d: sum = %f\n", d, hostResult[d]);
    }

    // Clean up
    for (int d = 0; d < deviceCount; ++d) {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaFree(d_data[d]));
        CHECK_CUDA(cudaFree(d_halo[d]));
        CHECK_CUDA(cudaFree(d_sum[d]));
    }

    return EXIT_SUCCESS;
}
```