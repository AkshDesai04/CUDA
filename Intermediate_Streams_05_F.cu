```c
/*
Aim: What are the restrictions on what the callback function can do? (It should not call most CUDA API functions, especially synchronous ones).

Thinking:
To illustrate the restrictions on a host callback function (registered via cudaLaunchHostFunc), I will write a minimal CUDA program that:
1. Creates a stream.
2. Launches a simple kernel on that stream.
3. Registers a callback that will be executed after the kernel completes.
4. Inside the callback, I will attempt to perform both an asynchronous operation (cudaMemcpyAsync) which is allowed, and a synchronous operation (cudaDeviceSynchronize) which is disallowed.
5. The program will print out whether the synchronous call succeeded or failed and explain that it is prohibited by the CUDA runtime.
6. The code will compile and run on a CUDA‑capable GPU; it demonstrates that synchronous calls in callbacks are not safe and will generate an error.
*/

#include <cuda_runtime.h>
#include <stdio.h>

/* Simple kernel that adds two vectors */
__global__ void addKernel(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

/* Callback function that runs on the host after kernel completion */
void CUDART_CB myCallback(void *userData)
{
    printf("[Callback] Called after kernel completion.\n");

    /* ---- Allowed asynchronous operation ---- */
    float *devPtr = (float *)userData;
    float hostBuffer[5];
    cudaError_t err = cudaMemcpyAsync(hostBuffer, devPtr, 5 * sizeof(float),
                                      cudaMemcpyDeviceToHost, 0);
    if (err != cudaSuccess)
    {
        printf("[Callback] Async cudaMemcpyAsync failed: %s\n",
               cudaGetErrorString(err));
    }
    else
    {
        printf("[Callback] Async cudaMemcpyAsync succeeded.\n");
    }

    /* ---- Disallowed synchronous operation ---- */
    printf("[Callback] Attempting synchronous cudaDeviceSynchronize (should fail).\n");
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("[Callback] Synchronous cudaDeviceSynchronize failed as expected: %s\n",
               cudaGetErrorString(err));
    }
    else
    {
        printf("[Callback] Synchronous cudaDeviceSynchronize succeeded (unexpected).\n");
    }
}

/* Simple error checking macro */
#define CHECK_CUDA(call)                                        \
    do {                                                        \
        cudaError_t err = call;                                 \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

int main(void)
{
    const int N = 5;
    size_t size = N * sizeof(float);
    float h_a[N] = {1, 2, 3, 4, 5};
    float h_b[N] = {5, 4, 3, 2, 1};
    float h_c[N] = {0};

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void **)&d_a, size));
    CHECK_CUDA(cudaMalloc((void **)&d_b, size));
    CHECK_CUDA(cudaMalloc((void **)&d_c, size));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    /* Create a stream */
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    /* Launch kernel on the stream */
    int threadsPerBlock = 128;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocks, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());

    /* Register callback on the stream */
    CHECK_CUDA(cudaLaunchHostFunc(stream, myCallback, d_c));

    /* Wait for all operations to finish */
    CHECK_CUDA(cudaStreamSynchronize(stream));

    /* Copy result back and print */
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    printf("Result:\n");
    for (int i = 0; i < N; ++i)
        printf("  %f + %f = %f\n", h_a[i], h_b[i], h_c[i]);

    /* Clean up */
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}
```