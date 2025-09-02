/*
To compile code with dynamic parallelism, you need to add `-rdc=true` (relocatable device code) to your `nvcc` flags. Practice this.

The goal of this program is to demonstrate the use of CUDA dynamic parallelism.  In CUDA, dynamic parallelism allows a kernel running on the GPU to launch additional kernels directly from the device.  This feature requires that the code be compiled with the `-rdc=true` flag, which enables relocatable device code.

The program below follows these steps:

1. A simple `childKernel` is defined.  It receives a pointer to device memory and an offset and length, and it increments each element in that segment by one.
2. A `parentKernel` is defined.  For each block, it computes an offset into the global array and then launches `childKernel` from the device to work on that segment.  The launch configuration for the child kernel is defined at runtime and is a key example of dynamic parallelism.
3. In `main`, a host array is allocated and initialized to zeros.  The array is copied to the device.
4. The `parentKernel` is launched.  Because it launches child kernels on the GPU, the whole program must be compiled with `-rdc=true`.
5. After the kernels finish, the result is copied back to the host and the first few elements are printed to confirm that the child kernels performed the expected increments.
6. All allocated memory is freed, and the program checks for errors.

To compile and run this code, use a command similar to:

```
nvcc -rdc=true -o dynamic_parallism dynamic_parallism.cu
./dynamic_parallism
```

The output should show that the elements of the array have been incremented from 0 to 1, confirming that the dynamic parallelism worked correctly.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for error checking */
#define CUDA_CHECK(err)                                                \
    do {                                                              \
        cudaError_t err_ = (err);                                     \
        if (err_ != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err_));   \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

/* Child kernel: increments each element in a segment of the array */
__global__ void childKernel(int *data, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[offset + idx] += 1;
    }
}

/* Parent kernel: launches childKernel for its assigned segment */
__global__ void parentKernel(int *data, int totalSize) {
    int blockStart = blockIdx.x * blockDim.x;
    int elementsPerBlock = blockDim.x;

    /* Adjust the last block if it goes beyond the array size */
    if (blockStart + elementsPerBlock > totalSize) {
        elementsPerBlock = totalSize - blockStart;
    }

    /* Dynamic launch of childKernel */
    childKernel<<<1, elementsPerBlock>>>(data, blockStart, elementsPerBlock);
    /* No need to synchronize here; the launch is synchronous with respect
       to the parent kernel's execution. */
}

int main(void) {
    const int N = 1024;                 // Size of the array
    const size_t bytes = N * sizeof(int);

    /* Allocate host memory */
    int *h_data = (int *)malloc(bytes);
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host array to zero */
    for (int i = 0; i < N; ++i) {
        h_data[i] = 0;
    }

    /* Allocate device memory */
    int *d_data;
    CUDA_CHECK(cudaMalloc((void **)&d_data, bytes));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    /* Define launch parameters for parent kernel */
    const int parentBlockSize = 256;
    const int parentGridSize = (N + parentBlockSize - 1) / parentBlockSize;

    /* Launch parent kernel */
    parentKernel<<<parentGridSize, parentBlockSize>>>(d_data, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Wait for all kernels to finish */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    /* Print first 10 results to verify */
    printf("First 10 results after dynamic parallelism:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    /* Clean up */
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return 0;
}
