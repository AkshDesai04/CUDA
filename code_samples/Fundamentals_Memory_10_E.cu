```cpp
/*
Use `cudaMemset2D` to initialize a pitched 2D allocation to zero.

Thinking: The program will allocate a 2D array on the device using cudaMallocPitch to obtain a pitched memory pointer and the pitch value. 
Then it will call cudaMemset2D to set each element to zero. The width in bytes is width * sizeof(int) and the height is number of rows. 
After setting, the program will copy the data back to host using cudaMemcpy2D and verify that all elements are zero. 
Error checking is performed via a macro to abort on any CUDA runtime error. The program prints the pitch and a confirmation that the array was initialized correctly.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

int main(void) {
    const int width  = 10;   // number of columns
    const int height = 5;    // number of rows
    const size_t elemSize = sizeof(int);
    size_t pitch;
    int *d_ptr = NULL;

    /* Allocate pitched memory on the device */
    CHECK_CUDA(cudaMallocPitch((void **)&d_ptr, &pitch,
                               width * elemSize, height));

    printf("Allocated pitched memory: pitch = %zu bytes\n", pitch);

    /* Initialize all elements to zero using cudaMemset2D */
    CHECK_CUDA(cudaMemset2D(d_ptr, pitch, 0,
                            width * elemSize, height));

    /* Allocate host array to copy data back */
    int h_ptr[height][width];

    /* Copy data from device to host */
    CHECK_CUDA(cudaMemcpy2D(h_ptr, width * elemSize,
                            d_ptr, pitch,
                            width * elemSize, height,
                            cudaMemcpyDeviceToHost));

    /* Verify that all elements are zero */
    int allZero = 1;
    for (int i = 0; i < height && allZero; ++i) {
        for (int j = 0; j < width && allZero; ++j) {
            if (h_ptr[i][j] != 0) {
                printf("Non-zero value found at (%d, %d): %d\n", i, j, h_ptr[i][j]);
                allZero = 0;
            }
        }
    }
    if (allZero) {
        printf("All elements are correctly initialized to zero.\n");
    } else {
        printf("Initialization failed.\n");
    }

    /* Clean up */
    CHECK_CUDA(cudaFree(d_ptr));

    return 0;
}
```