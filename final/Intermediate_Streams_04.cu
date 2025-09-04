```cuda
/* 
Aim of the program:
Create a dependency between two streams. Launch a kernel on stream1. Record an event on stream1 after the kernel. Then, make stream2 wait for that event using `cudaStreamWaitEvent` before launching its own kernel. This ensures the second kernel only starts after the first one is finished.

Thinking process:
1. We need two CUDA streams: stream1 and stream2.
2. Launch a simple kernel on stream1 that writes a known value to a device array.
3. After the kernel launch, record an event on stream1.
4. In stream2, call `cudaStreamWaitEvent(stream2, event, 0)` so that any subsequent work on stream2 will wait until the event is completed (i.e., the first kernel has finished).
5. Launch a second kernel on stream2 that writes a different value to another device array.
6. Synchronize the device to ensure both kernels finish before exiting.
7. Add error checking for all CUDA API calls to make debugging easier.
8. Provide basic output to verify the dependency by printing the results from the host arrays.

The kernels are trivial: they simply set each element of an array to a constant. The host prints the first few elements of each array after both kernels have executed to confirm that the second kernel indeed ran after the first. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(err)                                                    \
    do {                                                                   \
        cudaError_t err_ = (err);                                          \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                    cudaGetErrorString(err_), __FILE__, __LINE__);         \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// Kernel that fills an array with a given value
__global__ void fill_kernel(int *arr, int value, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        arr[idx] = value;
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_arr1 = (int *)malloc(size);
    int *h_arr2 = (int *)malloc(size);
    if (!h_arr1 || !h_arr2) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_arr1, *d_arr2;
    CUDA_CHECK(cudaMalloc((void **)&d_arr1, size));
    CUDA_CHECK(cudaMalloc((void **)&d_arr2, size));

    // Create streams
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Create event
    cudaEvent_t evt;
    CUDA_CHECK(cudaEventCreate(&evt));

    // Launch first kernel on stream1
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    fill_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_arr1, 1, N);
    CUDA_CHECK(cudaGetLastError());

    // Record event on stream1 after kernel1
    CUDA_CHECK(cudaEventRecord(evt, stream1));

    // Make stream2 wait for the event
    CUDA_CHECK(cudaStreamWaitEvent(stream2, evt, 0));

    // Launch second kernel on stream2 (depends on stream1)
    fill_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_arr2, 2, N);
    CUDA_CHECK(cudaGetLastError());

    // Synchronize to ensure all work is done
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_arr1, d_arr1, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_arr2, d_arr2, size, cudaMemcpyDeviceToHost));

    // Verify results by printing first 5 elements of each array
    printf("First 5 elements of array 1 (should be 1):\n");
    for (int i = 0; i < 5; ++i) {
        printf("%d ", h_arr1[i]);
    }
    printf("\n");

    printf("First 5 elements of array 2 (should be 2):\n");
    for (int i = 0; i < 5; ++i) {
        printf("%d ", h_arr2[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaEventDestroy(evt));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaFree(d_arr1));
    CUDA_CHECK(cudaFree(d_arr2));
    free(h_arr1);
    free(h_arr2);

    return EXIT_SUCCESS;
}
```