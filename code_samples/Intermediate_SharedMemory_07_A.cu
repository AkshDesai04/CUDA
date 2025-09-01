```cuda
/* 
Aim of the program:
Generalize this to reverse a large array by having each block reverse its own segment.

Thinking:
The goal is to reverse a very large array that cannot be processed by a single block due to CUDA's
limited shared memory and thread capacity. We divide the array into contiguous segments, each
handled by one block. The reversal is performed in two phases:

1. **Local reversal** – Each block reverses the elements within its own segment.
   For a segment `[segStart, segEnd)` the thread with local offset `o` swaps
   `arr[segStart + o]` with `arr[segEnd - 1 - o]` as long as `o < segmentLength/2`.
   This step ensures that all segments are individually reversed.

2. **Cross‑segment swapping** – After the local reversal, the array is not yet globally
   reversed. We then perform a second kernel launch where each block `i` swaps its
   segment with the symmetric segment from the other end of the array.
   The block with index `i` (where `i < numBlocks/2`) swaps element `segStart1 + o`
   with element `segStart2 + (segmentLength - 1 - o)` for all offsets `o` in its
   segment. If the array size is not evenly divisible by the segment size, the
   last segment may be shorter; we handle this by computing actual segment lengths
   (`len1` and `len2`) at runtime.

By chaining these two kernels, the final array is globally reversed while
leveraging block‑level parallelism. The code includes comprehensive bounds
checking and uses grid‑stride loops to handle arbitrarily large segments.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define BLOCK_SIZE 256          // Threads per block
#define SEGMENT_SIZE 1024       // Elements per segment (can be tuned)
#define CHECK_CUDA(call)                                      \
    {                                                          \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    }

/* Kernel to reverse each segment locally */
__global__ void reverseSegments(int *d_arr, size_t N, size_t segSize, size_t numBlocks)
{
    size_t blockIdxGlobal = blockIdx.x;
    size_t segStart = blockIdxGlobal * segSize;
    size_t segEnd = min(segStart + segSize, N);
    size_t len = segEnd - segStart;

    /* Grid-stride loop over local offsets within the segment */
    for (size_t localIdx = threadIdx.x; localIdx < len / 2; localIdx += blockDim.x)
    {
        size_t g1 = segStart + localIdx;
        size_t g2 = segEnd - 1 - localIdx;
        int tmp = d_arr[g1];
        d_arr[g1] = d_arr[g2];
        d_arr[g2] = tmp;
    }
}

/* Kernel to swap pairs of segments from opposite ends */
__global__ void swapSegmentPairs(int *d_arr, size_t N, size_t segSize, size_t numBlocks)
{
    size_t i = blockIdx.x;          // Index of the left segment
    size_t segStart1 = i * segSize;
    size_t segEnd1 = min(segStart1 + segSize, N);
    size_t len1 = segEnd1 - segStart1;

    size_t segStart2 = max((size_t)0, N - (i + 1) * segSize);
    size_t segEnd2 = segStart2 + len1;   // Should match len1 or be shorter at edges
    size_t len2 = segEnd2 - segStart2;

    /* If the two segments overlap (odd number of segments, middle block), skip */
    if (segStart1 >= segStart2) return;

    /* Each thread swaps one element pair within the segments */
    for (size_t localIdx = threadIdx.x; localIdx < len1; localIdx += blockDim.x)
    {
        size_t g1 = segStart1 + localIdx;
        size_t g2 = segStart2 + (len1 - 1 - localIdx);
        int tmp = d_arr[g1];
        d_arr[g1] = d_arr[g2];
        d_arr[g2] = tmp;
    }
}

int main()
{
    const size_t N = 1 << 24;          // Example: 16 million elements (~64MB)
    size_t bytes = N * sizeof(int);

    /* Allocate host memory and initialize */
    int *h_arr = (int *)malloc(bytes);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    srand((unsigned)time(NULL));
    for (size_t i = 0; i < N; ++i) h_arr[i] = (int)i;

    /* Allocate device memory */
    int *d_arr;
    CHECK_CUDA(cudaMalloc((void **)&d_arr, bytes));

    /* Copy data to device */
    CHECK_CUDA(cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice));

    /* Determine number of blocks */
    size_t numBlocks = (N + SEGMENT_SIZE - 1) / SEGMENT_SIZE;

    /* Launch local reversal kernel */
    dim3 grid1(numBlocks);
    dim3 block1(BLOCK_SIZE);
    reverseSegments<<<grid1, block1>>>(d_arr, N, SEGMENT_SIZE, numBlocks);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Launch cross‑segment swapping kernel */
    size_t numSwapBlocks = numBlocks / 2;
    dim3 grid2(numSwapBlocks);
    dim3 block2(BLOCK_SIZE);
    swapSegmentPairs<<<grid2, block2>>>(d_arr, N, SEGMENT_SIZE, numBlocks);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost));

    /* Verify correctness for first and last 10 elements */
    printf("First 10 elements after reversal:\n");
    for (size_t i = 0; i < 10; ++i)
        printf("%d ", h_arr[i]);
    printf("\n");

    printf("Last 10 elements after reversal:\n");
    for (size_t i = N - 10; i < N; ++i)
        printf("%d ", h_arr[i]);
    printf("\n");

    /* Clean up */
    free(h_arr);
    CHECK_CUDA(cudaFree(d_arr));

    return 0;
}
```