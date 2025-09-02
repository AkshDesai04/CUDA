```cuda
/*
Aim of the program: 
Create two host threads. Have one thread set device 0 and run a kernel, while the second thread sets device 1 and runs a kernel concurrently.

Thinking:
- Use POSIX threads (pthread) to create two host threads.
- Each thread will set its CUDA device using cudaSetDevice() with the appropriate device ID (0 or 1).
- Each thread will allocate device memory, launch a simple kernel that writes its device ID into the array, and then copy the result back to host memory.
- Since CUDA kernel launches are asynchronous, launching kernels on different devices from different threads will run concurrently (subject to hardware scheduling).
- We need to handle error checking for CUDA API calls.
- After kernel execution, each thread will print the first few elements of the result array to verify correct execution.
- The main thread will wait for both child threads to finish using pthread_join.
- No external dependencies other than standard CUDA runtime and pthreads.
*/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE 1024

// Simple kernel that writes the device ID into each element
__global__ void writeDeviceID(int *arr, int deviceID) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ARRAY_SIZE) {
        arr[idx] = deviceID;
    }
}

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Thread argument structure
typedef struct {
    int deviceID;
} thread_args;

// Thread function
void *thread_func(void *arg) {
    thread_args *targs = (thread_args *)arg;
    int devID = targs->deviceID;

    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(devID));

    // Allocate device memory
    int *d_arr;
    CUDA_CHECK(cudaMalloc((void **)&d_arr, ARRAY_SIZE * sizeof(int)));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (ARRAY_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    writeDeviceID<<<blocksPerGrid, threadsPerBlock>>>(d_arr, devID);
    // No need to synchronize; kernel launch is asynchronous

    // Copy result back to host
    int *h_arr = (int *)malloc(ARRAY_SIZE * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    // Print first 5 elements to verify
    printf("Thread for device %d: first 5 elements: ", devID);
    for (int i = 0; i < 5; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Cleanup
    free(h_arr);
    CUDA_CHECK(cudaFree(d_arr));

    return NULL;
}

int main(void) {
    // Check number of devices
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        fprintf(stderr, "Need at least 2 CUDA devices. Found %d.\n", deviceCount);
        return EXIT_FAILURE;
    }

    pthread_t thread0, thread1;
    thread_args args0 = { .deviceID = 0 };
    thread_args args1 = { .deviceID = 1 };

    // Create threads
    if (pthread_create(&thread0, NULL, thread_func, &args0) != 0) {
        perror("pthread_create for device 0 failed");
        return EXIT_FAILURE;
    }
    if (pthread_create(&thread1, NULL, thread_func, &args1) != 0) {
        perror("pthread_create for device 1 failed");
        return EXIT_FAILURE;
    }

    // Wait for threads to finish
    pthread_join(thread0, NULL);
    pthread_join(thread1, NULL);

    printf("Both threads completed.\n");
    return EXIT_SUCCESS;
}
```