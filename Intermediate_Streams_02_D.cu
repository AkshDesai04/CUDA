/*
Use `cudaMemcpyAsync` and pass the stream argument to all copies and kernel launches.
My thinking:
The task is to produce a CUDA C program (.cu file) that demonstrates the use of cudaMemcpyAsync
and ensures that the stream argument is passed to every copy operation and every kernel launch.
The program should:
1. Allocate host and device memory for a simple data set.
2. Create a CUDA stream.
3. Use cudaMemcpyAsync to copy data from host to device on that stream.
4. Launch a kernel on the same stream.
5. Use cudaMemcpyAsync again to copy the results back to the host on the same stream.
6. Synchronize the stream to ensure all operations are complete before checking the results.
7. Include proper error checking after each CUDA call.
8. Clean up resources (free memory, destroy stream, reset device).

A simple kernel that adds a constant value to each array element will suffice.
The program will print a success message if the computation produces the expected result.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

/* Error checking macro */
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

/* Simple kernel: add a constant value to each element */
__global__ void addKernel(int *d_out, const int *d_in, int value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx] + value;
    }
}

int main(void) {
    int *h_in  = NULL;
    int *h_out = NULL;
    int *d_in  = NULL;
    int *d_out = NULL;
    cudaStream_t stream;

    /* Allocate pinned host memory for faster async copy */
    CUDA_CHECK(cudaMallocHost((void**)&h_in,  N * sizeof(int)));
    CUDA_CHECK(cudaMallocHost((void**)&h_out, N * sizeof(int)));

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc((void**)&d_in,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(int)));

    /* Initialize host input array */
    for (int i = 0; i < N; ++i) {
        h_in[i] = i;
    }

    /* Create a CUDA stream */
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* Asynchronously copy input from host to device on the stream */
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, N * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    /* Launch kernel on the same stream */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_out, d_in, 5, N);

    /* Asynchronously copy result back to host on the stream */
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, N * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));

    /* Wait for all operations in the stream to complete */
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* Verify results */
    int error = 0;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != h_in[i] + 5) {
            fprintf(stderr, "Verification failed at index %d: %d + 5 != %d\n",
                    i, h_in[i], h_out[i]);
            error = 1;
            break;
        }
    }

    if (!error) {
        printf("Success! All values added correctly.\n");
    }

    /* Clean up */
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
