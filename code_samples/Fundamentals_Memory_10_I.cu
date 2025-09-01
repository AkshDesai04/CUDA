```cuda
/* 
Demonstrate copying a sub-matrix from host to device using `cudaMemcpy2D`.

Thinking process:
1. The aim is to copy a contiguous rectangular block (sub-matrix) from a larger matrix stored on the host to a device buffer that may have a different pitch.
2. We create a small 4x4 host matrix filled with known values. 
3. We choose a 2x2 sub-matrix starting at offset (row=1, col=1) to copy.
4. On the device, we allocate memory with `cudaMallocPitch` to get a pitch that satisfies alignment constraints. The device buffer will be exactly the size of the sub-matrix.
5. We perform `cudaMemcpy2D` using:
   - source pointer: `h_A + offsetRow*cols + offsetCol`
   - source pitch: `cols * sizeof(int)` (full row stride on host)
   - destination pointer: device pointer `d_B`
   - destination pitch: device pitch returned by `cudaMallocPitch`
   - width in bytes: `subCols * sizeof(int)`
   - height: `subRows`
   - direction: `cudaMemcpyHostToDevice`
6. After copying, we copy the data back to a separate host buffer to verify that the operation succeeded.
7. We print the original sub-matrix and the retrieved data to show that they match.
8. We also include a small error-checking macro for clarity.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                              \
        }                                                                     \
    } while (0)

int main(void) {
    /* Host matrix dimensions */
    const int rows = 4;
    const int cols = 4;

    /* Sub-matrix dimensions and offset */
    const int subRows = 2;
    const int subCols = 2;
    const int offsetRow = 1;
    const int offsetCol = 1;

    /* Host matrix initialization */
    int h_A[rows][cols] = {
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    /* Device buffer for the sub-matrix */
    int *d_B = NULL;
    size_t pitchB = 0;
    CHECK_CUDA(cudaMallocPitch(&d_B, &pitchB,
                               subCols * sizeof(int), subRows));

    /* Copy sub-matrix from host to device */
    size_t hostPitch = cols * sizeof(int);
    size_t widthBytes = subCols * sizeof(int);

    CHECK_CUDA(cudaMemcpy2D(d_B, pitchB,
                            h_A + offsetRow * cols + offsetCol,
                            hostPitch,
                            widthBytes, subRows,
                            cudaMemcpyHostToDevice));

    /* Allocate a host buffer to copy back for verification */
    int h_C[rows][cols] = {0};

    CHECK_CUDA(cudaMemcpy2D(h_C + offsetRow * cols + offsetCol,
                            hostPitch,
                            d_B, pitchB,
                            widthBytes, subRows,
                            cudaMemcpyDeviceToHost));

    /* Print original sub-matrix */
    printf("Original sub-matrix (host):\n");
    for (int r = 0; r < subRows; ++r) {
        for (int c = 0; c < subCols; ++c) {
            printf("%4d", h_A[offsetRow + r][offsetCol + c]);
        }
        printf("\n");
    }

    /* Print copied sub-matrix */
    printf("\nCopied sub-matrix (retrieved from device):\n");
    for (int r = 0; r < subRows; ++r) {
        for (int c = 0; c < subCols; ++c) {
            printf("%4d", h_C[offsetRow + r][offsetCol + c]);
        }
        printf("\n");
    }

    /* Clean up */
    CHECK_CUDA(cudaFree(d_B));

    return 0;
}
```