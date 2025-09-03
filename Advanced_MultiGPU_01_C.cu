```cuda
/* 
Aim of the program: Create two host threads. Have one thread set device 0 and run a kernel, while the second thread sets device 1 and runs a kernel concurrently.

Thinking:
- Use POSIX threads (pthreads) for host-side concurrency.
- Each thread will set its own CUDA device using cudaSetDevice().
- Launch a simple kernel that writes a unique value to a device array.
- Allocate separate device memory for each thread to avoid interference.
- After launching the kernel, call cudaDeviceSynchronize() to ensure completion.
- The main thread will wait for both worker threads to finish before exiting.
- The kernel will run on its assigned device; CUDA runtime will handle context creation.
- We need to include error checking for CUDA calls and pthread functions.
- To observe concurrency, we can insert a sleep or a loop inside the kernel, but for simplicity the kernels will just perform a quick operation.
- Ensure that we use the same stream (default 0) or create streams if desired. For now, default stream is fine.
- The program is self-contained and compiles with nvcc.
*/

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // for sleep

#define CHECK_CUDA(call)                                 \
    do {                                                 \
        cudaError_t err = call;                          \
        if (err != cudaSuccess) {                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__,                  \
                    cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                          \
        }                                                \
    } while (0)

#define CHECK_PTHREAD(call)                               \
    do {                                                 \
        int err = call;                                  \
        if (err != 0) {                                  \
            fprintf(stderr, "Pthread error at %s:%d: %s\n", \
                    __FILE__, __LINE__,                  \
                    strerror(err));                      \
            exit(EXIT_FAILURE);                          \
        }                                                \
    } while (0)

__global__ void simpleKernel(int *d_arr, int val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // For demonstration, each thread writes its value
    d_arr[idx] = val;
}

typedef struct {
    int device_id;
    int val;
} thread_arg_t;

void* threadFunc(void *arg)
{
    thread_arg_t *targ = (thread_arg_t*)arg;
    int device = targ->device_id;
    int val = targ->val;

    // Set the device for this thread
    CHECK_CUDA(cudaSetDevice(device));

    // Allocate device memory
    int *d_arr;
    const int N = 256;
    size_t size = N * sizeof(int);
    CHECK_CUDA(cudaMalloc(&d_arr, size));

    // Launch kernel
    dim3 threadsPerBlock(64);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
    simpleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, val);
    CHECK_CUDA(cudaGetLastError());

    // Wait for kernel to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // For debugging, copy back and print first element
    int h_val;
    CHECK_CUDA(cudaMemcpy(&h_val, d_arr, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Thread for device %d finished, first element = %d\n", device, h_val);

    // Clean up
    CHECK_CUDA(cudaFree(d_arr));

    return NULL;
}

int main()
{
    pthread_t threads[2];
    thread_arg_t args[2];

    // Thread 0 will use device 0
    args[0].device_id = 0;
    args[0].val = 42;

    // Thread 1 will use device 1
    args[1].device_id = 1;
    args[1].val = 84;

    // Create threads
    CHECK_PTHREAD(pthread_create(&threads[0], NULL, threadFunc, &args[0]));
    CHECK_PTHREAD(pthread_create(&threads[1], NULL, threadFunc, &args[1]));

    // Wait for threads to finish
    CHECK_PTHREAD(pthread_join(threads[0], NULL));
    CHECK_PTHREAD(pthread_join(threads[1], NULL));

    printf("Both threads completed.\n");
    return 0;
}
```