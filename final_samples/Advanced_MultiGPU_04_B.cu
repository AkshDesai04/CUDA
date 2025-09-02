```cpp
/*
Aim of the program:
Use streams and events to overlap the halo exchange with computation on the interior
of each sub-domain.

Thinking:
We partition a 2D domain into several sub-domains. Each sub-domain holds a
buffer that includes halo cells on all four sides. In a real distributed
application halo cells would be exchanged with neighbouring processes.  For
illustration we mimic the exchange by performing asynchronous device-to-device
memcpy operations between the halo regions of neighbouring sub-domains.
While the halo exchange is in flight, we launch a computation kernel that
operates only on the interior cells of each sub-domain.  Because the kernel
does not touch the halo cells, the kernel and the memcpy can execute
concurrently on the GPU, achieving overlap.

The program:
* Defines a small 2D domain split into 4 sub-domains.
* Each sub-domain contains a halo width of 1 cell.
* Two CUDA streams are created per sub-domain:
    - streamHalo   : performs the async halo copy.
    - streamCompute: launches the interior computation kernel.
* CUDA events are used to mark completion of the compute kernel.  The
  main host code waits on all compute events before finalising.
* The example uses a simple compute kernel that adds two source arrays
  and writes to a destination array.  The kernel is launched over the
  interior region only.

This code demonstrates how to structure overlapping communication and
computation with streams and events in CUDA C.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define NX 64            // total domain width
#define NY 64            // total domain height
#define SUB_X 2          // number of sub-domains in x
#define SUB_Y 2          // number of sub-domains in y
#define HALO 1          // halo width

// Check CUDA errors
#define CHECK_CUDA(call)                                                   \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    }

// Kernel that operates only on interior cells
__global__ void interior_compute(float* a, const float* b, int width, int height, int stride)
{
    // global indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // compute only interior cells, skip halos
    if (x >= HALO && x < width - HALO && y >= HALO && y < height - HALO) {
        int idx = y * stride + x;
        a[idx] = b[idx] + 1.0f;  // simple operation
    }
}

// Helper to compute 2D to 1D index with stride
__device__ __host__ inline int idx(int x, int y, int stride) {
    return y * stride + x;
}

// Host function to copy halo cells between two sub-domains
// We copy top and bottom rows, left and right columns
void copy_halo(float* dst, const float* src,
               int dst_stride, int src_stride,
               int width, int height,
               cudaStream_t stream)
{
    // Top halo: copy first row of src to first halo row of dst
    CHECK_CUDA(cudaMemcpyAsync(dst, src, width * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));
    // Bottom halo: copy last row of src to last halo row of dst
    CHECK_CUDA(cudaMemcpyAsync(dst + (height - 1) * dst_stride,
                               src + (height - 2) * src_stride,
                               width * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));
    // Left and right halos (excluding corners already copied)
    // Left column
    for (int y = 1; y < height - 1; ++y) {
        CHECK_CUDA(cudaMemcpyAsync(dst + y * dst_stride,
                                   src + y * src_stride,
                                   sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    // Right column
    for (int y = 1; y < height - 1; ++y) {
        CHECK_CUDA(cudaMemcpyAsync(dst + y * dst_stride + width - 1,
                                   src + y * src_stride + width - 2,
                                   sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

int main()
{
    // Dimensions of each sub-domain including halos
    const int sub_w = NX / SUB_X + 2 * HALO;
    const int sub_h = NY / SUB_Y + 2 * HALO;
    const size_t sub_size = sub_w * sub_h * sizeof(float);

    // Allocate arrays for each sub-domain
    float* d_a[SUB_X * SUB_Y];
    float* d_b[SUB_X * SUB_Y];
    for (int i = 0; i < SUB_X * SUB_Y; ++i) {
        CHECK_CUDA(cudaMalloc(&d_a[i], sub_size));
        CHECK_CUDA(cudaMalloc(&d_b[i], sub_size));
        // Initialize d_b with some data
        CHECK_CUDA(cudaMemset(d_b[i], 0, sub_size));
    }

    // Create streams and events
    cudaStream_t streamHalo[SUB_X * SUB_Y];
    cudaStream_t streamCompute[SUB_X * SUB_Y];
    cudaEvent_t  eventCompute[SUB_X * SUB_Y];
    for (int i = 0; i < SUB_X * SUB_Y; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streamHalo[i]));
        CHECK_CUDA(cudaStreamCreate(&streamCompute[i]));
        CHECK_CUDA(cudaEventCreate(&eventCompute[i]));
    }

    // Main loop: for illustration perform a single step
    for (int i = 0; i < SUB_X * SUB_Y; ++i) {
        // Identify neighbours (simple 4-neighbour grid)
        int ix = i % SUB_X;
        int iy = i / SUB_X;

        // Halo exchange with left neighbour
        if (ix > 0) {
            int left = i - 1;
            copy_halo(d_a[i], d_b[left], sub_w, sub_w, sub_w, sub_h, streamHalo[i]);
        }
        // Halo exchange with right neighbour
        if (ix < SUB_X - 1) {
            int right = i + 1;
            copy_halo(d_a[i], d_b[right], sub_w, sub_w, sub_w, sub_h, streamHalo[i]);
        }
        // Halo exchange with top neighbour
        if (iy > 0) {
            int top = i - SUB_X;
            copy_halo(d_a[i], d_b[top], sub_w, sub_w, sub_w, sub_h, streamHalo[i]);
        }
        // Halo exchange with bottom neighbour
        if (iy < SUB_Y - 1) {
            int bottom = i + SUB_X;
            copy_halo(d_a[i], d_b[bottom], sub_w, sub_w, sub_w, sub_h, streamHalo[i]);
        }

        // Launch interior computation kernel on streamCompute
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((sub_w + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (sub_h + threadsPerBlock.y - 1) / threadsPerBlock.y);
        interior_compute<<<numBlocks, threadsPerBlock, 0, streamCompute[i]>>>(d_a[i], d_b[i], sub_w, sub_h, sub_w);
        CHECK_CUDA(cudaGetLastError());

        // Record event after compute
        CHECK_CUDA(cudaEventRecord(eventCompute[i], streamCompute[i]));
    }

    // Wait for all compute events to finish
    for (int i = 0; i < SUB_X * SUB_Y; ++i) {
        CHECK_CUDA(cudaEventSynchronize(eventCompute[i]));
    }

    // Optional: copy results back to host and print a few values
    float* h_result = (float*)malloc(sub_w * sub_h * sizeof(float));
    for (int i = 0; i < SUB_X * SUB_Y; ++i) {
        CHECK_CUDA(cudaMemcpy(h_result, d_a[i], sub_w * sub_h * sizeof(float),
                              cudaMemcpyDeviceToHost));
        printf("Sub-domain %d, center value: %f\n", i, h_result[(sub_h/2)*sub_w + sub_w/2]);
    }

    // Clean up
    free(h_result);
    for (int i = 0; i < SUB_X * SUB_Y; ++i) {
        CHECK_CUDA(cudaFree(d_a[i]));
        CHECK_CUDA(cudaFree(d_b[i]));
        CHECK_CUDA(cudaStreamDestroy(streamHalo[i]));
        CHECK_CUDA(cudaStreamDestroy(streamCompute[i]));
        CHECK_CUDA(cudaEventDestroy(eventCompute[i]));
    }

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
```