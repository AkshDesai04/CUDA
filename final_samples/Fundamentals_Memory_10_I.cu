/*
 * Aim of the program:
 * Demonstrate copying a sub-matrix from host to device using `cudaMemcpy2D`.
 *
 * Thinking process:
 * 1. Allocate a simple 2D matrix on the host. For clarity, use an 8x8 integer array
 *    filled with sequential values so we can easily verify which elements are
 *    part of the sub-matrix.
 * 2. Define the sub-matrix to copy: choose a 4x4 block starting at row 2, column 2.
 * 3. Allocate device memory with cudaMallocPitch to get a pitch (row stride) that
 *    may be greater than the logical width times element size. This mimics real
 *    GPU memory alignment and demonstrates the use of pitch in cudaMemcpy2D.
 * 4. Use cudaMemcpy2D to copy from the host sub-block to the device buffer.
 *    * Source pointer = &h_matrix[row_offset * num_cols + col_offset].
 *    * Source pitch = num_cols * sizeof(int) (contiguous host memory).
 *    * Destination pitch = the pitch returned by cudaMallocPitch.
 *    * Width = sub_width * sizeof(int), Height = sub_height.
 * 5. Copy the data back to a second host array to verify that the transfer worked.
 *    This step also uses cudaMemcpy2D to copy from device back to host.
 * 6. Print the original matrix, the sub-matrix contents, and the retrieved
 *    sub-matrix to confirm that the data matches.
 * 7. Include a small error-checking macro to simplify CUDA API calls.
 *
 * The code is self-contained and can be compiled with:
 *     nvcc -o submatrix_copy submatrix_copy.cu
 * and run with no arguments. It will print the matrices to stdout.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void)
{
    const int num_rows = 8;
    const int num_cols = 8;
    const int sub_row_offset = 2;
    const int sub_col_offset = 2;
    const int sub_rows = 4;
    const int sub_cols = 4;

    /* Allocate and initialize host matrix */
    int h_matrix[num_rows][num_cols];
    for (int r = 0; r < num_rows; ++r) {
        for (int c = 0; c < num_cols; ++c) {
            h_matrix[r][c] = r * num_cols + c;
        }
    }

    /* Print original matrix */
    printf("Original host matrix:\n");
    for (int r = 0; r < num_rows; ++r) {
        for (int c = 0; c < num_cols; ++c) {
            printf("%4d ", h_matrix[r][c]);
        }
        printf("\n");
    }
    printf("\n");

    /* Allocate device memory for sub-matrix with pitch */
    int *d_sub_matrix = NULL;
    size_t d_pitch = 0;
    CUDA_CHECK(cudaMallocPitch((void**)&d_sub_matrix, &d_pitch,
                               sub_cols * sizeof(int), sub_rows));

    /* Copy sub-matrix from host to device using cudaMemcpy2D */
    const void *src_ptr = &h_matrix[sub_row_offset][sub_col_offset];
    size_t src_pitch = num_cols * sizeof(int);
    CUDA_CHECK(cudaMemcpy2D(d_sub_matrix, d_pitch,
                            src_ptr, src_pitch,
                            sub_cols * sizeof(int), sub_rows,
                            cudaMemcpyHostToDevice));

    /* Allocate host buffer to copy back data for verification */
    int h_sub_matrix[sub_rows][sub_cols];
    void *dst_ptr = &h_sub_matrix[0][0];
    size_t dst_pitch = sub_cols * sizeof(int);

    /* Copy sub-matrix back from device to host */
    CUDA_CHECK(cudaMemcpy2D(dst_ptr, dst_pitch,
                            d_sub_matrix, d_pitch,
                            sub_cols * sizeof(int), sub_rows,
                            cudaMemcpyDeviceToHost));

    /* Print sub-matrix extracted from original matrix */
    printf("Sub-matrix (host side, before copy):\n");
    for (int r = 0; r < sub_rows; ++r) {
        for (int c = 0; c < sub_cols; ++c) {
            printf("%4d ", h_matrix[sub_row_offset + r][sub_col_offset + c]);
        }
        printf("\n");
    }
    printf("\n");

    /* Print sub-matrix copied back from device */
    printf("Sub-matrix (device -> host):\n");
    for (int r = 0; r < sub_rows; ++r) {
        for (int c = 0; c < sub_cols; ++c) {
            printf("%4d ", h_sub_matrix[r][c]);
        }
        printf("\n");
    }
    printf("\n");

    /* Clean up */
    CUDA_CHECK(cudaFree(d_sub_matrix));

    return 0;
}
