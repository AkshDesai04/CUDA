```cuda
/* 
Aim of the program:
You will need two sets of device buffers, one for each stream.

Thought process:
- The requirement is to demonstrate the usage of two CUDA streams and 
  allocate separate device buffers for each stream.
- To keep the example simple yet complete, I will create two arrays on 
  the host, copy each to its own device buffer via its respective stream, 
  launch a small kernel that adds a constant to each element, and then 
  copy the results back to the host.
- Each stream will operate independently on its own data so that the 
  device buffers are truly separate.
- Basic error checking will be performed after each CUDA API call to 
  ensure correctness.
- The program will print the final arrays to verify the computation.

*/

// Include necessary headers
#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that adds a value to each element
__global__ void add_kernel(int *d_arr, int val, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] += val;
    }
}

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

int main()
{
    const int N = 10;                    // Size of each array
    const int val = 5;                   // Value to add in kernel

    // Host arrays
    int h_arr1[N];
    int h_arr2[N];
    for (int i = 0; i < N; ++i) {
        h_arr1[i] = i;
        h_arr2[i] = 2 * i;
    }

    // Device arrays
    int *d_arr1 = NULL;
    int *d_arr2 = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_arr1, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_arr2, N * sizeof(int)));

    // Create two streams
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Copy data to device asynchronously using the streams
    CUDA_CHECK(cudaMemcpyAsync(d_arr1, h_arr1, N * sizeof(int),
                               cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_arr2, h_arr2, N * sizeof(int),
                               cudaMemcpyHostToDevice, stream2));

    // Launch kernels on each stream
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_arr1, val, N);
    add_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_arr2, val, N);

    // Copy results back to host asynchronously
    CUDA_CHECK(cudaMemcpyAsync(h_arr1, d_arr1, N * sizeof(int),
                               cudaMemcpyDeviceToHost, stream1));
    CUDA_CHECK(cudaMemcpyAsync(h_arr2, d_arr2, N * sizeof(int),
                               cudaMemcpyDeviceToHost, stream2));

    // Wait for all operations to complete
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // Print results
    printf("Result array 1:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr1[i]);
    }
    printf("\n");

    printf("Result array 2:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr2[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_arr1));
    CUDA_CHECK(cudaFree(d_arr2));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));

    return 0;
}
```