/*
Aim of the program: Demonstrate copying a sub-matrix from host to device using cudaMemcpy2D.

Thinking:
- Create a host 2D matrix (flattened into 1‑D array) with known values.
- Define a sub‑matrix region (start row/col, height/width).
- Allocate device pitched memory just for that sub‑matrix.
- Use cudaMemcpy2D to copy the sub‑matrix from the host matrix to the device memory.
- Copy it back to a separate host array and print it to verify the operation.
- Include error checking for all CUDA API calls.
- Keep the example simple: use float elements and a small 8x8 matrix so that the output is easy to read.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                 \
    do {                                                 \
        cudaError_t err = call;                          \
        if (err != cudaSuccess) {                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",\
                    __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                         \
        }                                                \
    } while (0)

int main(void)
{
    /* Host matrix dimensions */
    const int hostRows = 8;
    const int hostCols = 8;
    const size_t hostPitch = hostCols * sizeof(float);  /* bytes per row */

    /* Allocate and initialize host matrix */
    float *h_data = (float*)malloc(hostRows * hostCols * sizeof(float));
    if (!h_data) { fprintf(stderr, "Host allocation failed\n"); return EXIT_FAILURE; }
    for (int i = 0; i < hostRows; ++i)
        for (int j = 0; j < hostCols; ++j)
            h_data[i*hostCols + j] = i * hostCols + j;   /* simple pattern */

    /* Sub-matrix parameters */
    const int srcRow = 2;     /* start row in host matrix */
    const int srcCol = 3;     /* start column in host matrix */
    const int subRows = 4;    /* number of rows to copy */
    const int subCols = 4;    /* number of columns to copy */

    /* Allocate device pitched memory for the sub-matrix */
    float *d_sub;
    size_t devPitch;
    CHECK_CUDA(cudaMallocPitch(&d_sub, &devPitch, subCols * sizeof(float), subRows));

    /* Copy sub-matrix from host to device */
    CHECK_CUDA(cudaMemcpy2D(
        d_sub,                     /* destination device pointer */
        devPitch,                  /* destination pitch in bytes */
        h_data + srcRow*hostCols + srcCol, /* source host pointer offset */
        hostPitch,                 /* source pitch in bytes */
        subCols * sizeof(float),   /* width in bytes to copy */
        subRows,                   /* height (number of rows) */
        cudaMemcpyHostToDevice));  /* direction */

    /* Allocate host memory for the received sub-matrix */
    float *h_sub = (float*)malloc(subRows * subCols * sizeof(float));
    if (!h_sub) { fprintf(stderr, "Host allocation failed\n"); return EXIT_FAILURE; }

    /* Copy sub-matrix back from device to host for verification */
    CHECK_CUDA(cudaMemcpy2D(
        h_sub,                     /* destination host pointer */
        subCols * sizeof(float),   /* destination pitch in bytes (row stride) */
        d_sub,                     /* source device pointer */
        devPitch,                  /* source pitch in bytes */
        subCols * sizeof(float),   /* width in bytes to copy */
        subRows,                   /* height (number of rows) */
        cudaMemcpyDeviceToHost));  /* direction */

    /* Print the copied sub-matrix */
    printf("Sub-matrix copied from host to device and back:\n");
    for (int i = 0; i < subRows; ++i) {
        for (int j = 0; j < subCols; ++j) {
            printf("%4.0f ", h_sub[i*subCols + j]);
        }
        printf("\n");
    }

    /* Clean up */
    free(h_data);
    free(h_sub);
    CHECK_CUDA(cudaFree(d_sub));

    return 0;
}
