```cpp
/*
Aim of the program:
Use streams and events to overlap the halo exchange with computation on the interior of each sub-domain.

Thinking:
The goal is to demonstrate how to overlap data movement (halo exchange) with computation on a GPU using CUDA streams and events. 
We model a simple 2‑D stencil computation on a single GPU. The domain is conceptually split into sub‑domains; for simplicity we only show one sub‑domain. 
The sub‑domain has interior cells and boundary cells that need halo data from neighboring sub‑domains. In a real multi‑GPU or MPI scenario, the halo data would be transferred between GPUs or processes, but here we use pinned host memory and cudaMemcpyAsync to simulate the halo transfer to device memory.

We use three CUDA streams:
1. `streamCopy`   – performs the asynchronous copy of halo data from host to device.
2. `streamCompute`– launches the kernel that updates only interior cells.
3. `streamHalo`   – waits for the halo copy to finish and then launches the kernel that updates boundary cells using the freshly copied halo.

The key overlapping part is that the interior computation can proceed while the halo data is still being transferred. Only when the halo data has arrived does the halo kernel execute, ensuring correctness without stalling the interior computation.

The code demonstrates:
- Allocation of device memory and pinned host memory for halos.
- Creation and use of multiple CUDA streams.
- Recording of events after the halo copy.
- Use of `cudaStreamWaitEvent` to enforce the dependency between the halo copy and the halo kernel.
- Synchronization of all streams before moving to the next time step.

This skeleton can be expanded for multiple sub‑domains, multi‑GPU setups, or real MPI halo exchanges. 
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Grid parameters
constexpr int NX = 512;           // total grid width
constexpr int NY = 512;           // total grid height
constexpr int HALO = 2;           // halo thickness

// Block size for kernel launch
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;

// Simple stencil kernel for interior cells (no halo usage)
__global__ void interiorKernel(float* d_in, float* d_out, int nx, int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x + HALO; // skip halo
    int iy = blockIdx.y * blockDim.y + threadIdx.y + HALO;

    if (ix < nx - HALO && iy < ny - HALO)
    {
        // 5‑point stencil (for demonstration, just copy value)
        d_out[iy * nx + ix] = d_in[iy * nx + ix];
    }
}

// Stencil kernel that uses halo data (boundary cells)
__global__ void haloKernel(float* d_in, float* d_out, int nx, int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // Process only boundary cells
    if (ix < nx && iy < ny)
    {
        if (ix < HALO || ix >= nx - HALO || iy < HALO || iy >= ny - HALO)
        {
            // For simplicity, just copy value
            d_out[iy * nx + ix] = d_in[iy * nx + ix];
        }
    }
}

// Helper macro for error checking
#define CUDA_CHECK(call)                                                  \
    {                                                                     \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess)                                           \
        {                                                                 \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__  \
                      << " : " << cudaGetErrorString(err) << std::endl;   \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

int main()
{
    // Host data
    std::vector<float> h_data(NX * NY, 1.0f); // initialize with 1.0

    // Device data
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, NX * NY * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, NX * NY * sizeof(float)));

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_data.data(), NX * NY * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Pinned host memory for halo transfer (simulate halo from neighbor)
    float *h_halo_send = nullptr;
    float *h_halo_recv = nullptr;
    size_t halo_bytes = 2 * HALO * NX * sizeof(float) + 2 * HALO * (NY - 2 * HALO) * sizeof(float); // top, bottom, left, right
    CUDA_CHECK(cudaMallocHost((void**)&h_halo_send, halo_bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_halo_recv, halo_bytes));

    // Initialize halo data (for demo)
    std::fill(h_halo_send, h_halo_send + halo_bytes / sizeof(float), 2.0f);

    // Create streams
    cudaStream_t streamCopy, streamCompute, streamHalo;
    CUDA_CHECK(cudaStreamCreate(&streamCopy));
    CUDA_CHECK(cudaStreamCreate(&streamCompute));
    CUDA_CHECK(cudaStreamCreate(&streamHalo));

    // Create event to signal halo copy completion
    cudaEvent_t haloCopyEvent;
    CUDA_CHECK(cudaEventCreate(&haloCopyEvent));

    // Dimensions for kernel launches
    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim((NX + BLOCK_X - 1) / BLOCK_X,
                 (NY + BLOCK_Y - 1) / BLOCK_Y);

    // Simulate one time step
    // 1. Asynchronously copy halo data from host to device
    // For simplicity, we copy entire halo region into a temporary device buffer
    // Here we assume halo data is placed contiguously in h_halo_send
    // We'll copy it into d_in's halo region
    // Top halo
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_halo_send,
                               HALO * NX * sizeof(float),
                               cudaMemcpyHostToDevice,
                               streamCopy));
    // Bottom halo
    CUDA_CHECK(cudaMemcpyAsync(d_in + (NY - HALO) * NX, h_halo_send + HALO * NX,
                               HALO * NX * sizeof(float),
                               cudaMemcpyHostToDevice,
                               streamCopy));
    // Left halo
    CUDA_CHECK(cudaMemcpyAsync(d_in + HALO, h_halo_send + 2 * HALO * NX,
                               (NY - 2 * HALO) * HALO * sizeof(float),
                               cudaMemcpyHostToDevice,
                               streamCopy));
    // Right halo
    CUDA_CHECK(cudaMemcpyAsync(d_in + NX - HALO, h_halo_send + 2 * HALO * NX + (NY - 2 * HALO) * HALO * sizeof(float),
                               (NY - 2 * HALO) * HALO * sizeof(float),
                               cudaMemcpyHostToDevice,
                               streamCopy));

    // Record event after halo copy
    CUDA_CHECK(cudaEventRecord(haloCopyEvent, streamCopy));

    // 2. Launch interior kernel concurrently on compute stream
    interiorKernel<<<gridDim, blockDim, 0, streamCompute>>>(d_in, d_out, NX, NY);

    // 3. Launch halo kernel on halo stream, but wait for halo copy to finish
    CUDA_CHECK(cudaStreamWaitEvent(streamHalo, haloCopyEvent, 0));
    haloKernel<<<gridDim, blockDim, 0, streamHalo>>>(d_in, d_out, NX, NY);

    // Synchronize all streams
    CUDA_CHECK(cudaStreamSynchronize(streamCompute));
    CUDA_CHECK(cudaStreamSynchronize(streamHalo));
    CUDA_CHECK(cudaStreamSynchronize(streamCopy));

    // Copy result back to host for verification
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_out, NX * NY * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(haloCopyEvent));
    CUDA_CHECK(cudaStreamDestroy(streamCopy));
    CUDA_CHECK(cudaStreamDestroy(streamCompute));
    CUDA_CHECK(cudaStreamDestroy(streamHalo));
    CUDA_CHECK(cudaFreeHost(h_halo_send));
    CUDA_CHECK(cudaFreeHost(h_halo_recv));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    std::cout << "Computation completed successfully." << std::endl;
    return 0;
}
```