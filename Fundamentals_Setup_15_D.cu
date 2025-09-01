```cuda
/*
Aim of the program:
Write a program with multiple threads, where each thread sets a different GPU device. Have one thread call `cudaDeviceReset()` and discuss the potential implications for the other threads.

Thinking behind the implementation:
1. We need a multithreaded C program that uses CUDA APIs. The simplest way to create threads in C on Linux is to use the POSIX pthreads library.
2. Each thread will:
   a. Set a unique GPU device using `cudaSetDevice(device_id)`.
   b. Launch a simple CUDA kernel that prints the device ID and the thread ID to verify that the thread is running on the correct device.
3. One of the threads (chosen deterministically, e.g., thread 2) will call `cudaDeviceReset()` after its first kernel launch. According to CUDA documentation, `cudaDeviceReset()` destroys all allocations and resets the device state for the current device. It affects only the device whose context is current for that thread.
4. After the reset, the same thread will attempt to launch a second kernel. This should succeed because the device is still available, but all previous allocations on that device (if any) are cleared. Other threads that are running on different devices should not be impacted, but if multiple threads share the same device, a reset from one thread would invalidate the context for all threads on that device.
5. We add comprehensive error checking after each CUDA call to see if any errors occur. If a thread tries to launch a kernel on a device that has been reset by another thread, it would receive an error such as `cudaErrorInvalidDevice`. The program will print this out to illustrate the implication.
6. Finally, we synchronize all threads before exiting the program.

The code below follows this plan. It uses a kernel that prints a message, and it demonstrates the effect of calling `cudaDeviceReset()` from one thread on that thread’s own device and on other threads that might share a device (if the system has only one GPU). The program prints diagnostic messages to show what happens.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <pthread.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    __func__, __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));                      \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// Simple kernel that prints which thread and device it's running on
__global__ void testKernel(int thread_id, int dev_id)
{
    printf("[Device %d] Thread %d executing kernel.\n", dev_id, thread_id);
}

// Thread argument structure
typedef struct {
    int thread_id;
    int device_id;
    int reset_flag; // if 1, this thread will call cudaDeviceReset()
} thread_arg_t;

// Thread function
void* thread_func(void* arg)
{
    thread_arg_t* t = (thread_arg_t*)arg;
    printf("[Thread %d] Setting device to %d.\n", t->thread_id, t->device_id);
    CUDA_CHECK(cudaSetDevice(t->device_id));

    // First kernel launch
    printf("[Thread %d] Launching first kernel on device %d.\n", t->thread_id, t->device_id);
    testKernel<<<1,1>>>(t->thread_id, t->device_id);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // If this thread is designated to reset, do it
    if (t->reset_flag) {
        printf("[Thread %d] Calling cudaDeviceReset() on device %d.\n", t->thread_id, t->device_id);
        CUDA_CHECK(cudaDeviceReset());
        // After reset, the device context is destroyed; we must set device again
        CUDA_CHECK(cudaSetDevice(t->device_id));
        printf("[Thread %d] Re-set device %d after reset.\n", t->thread_id, t->device_id);
    }

    // Second kernel launch to see if reset affected this thread
    printf("[Thread %d] Launching second kernel on device %d.\n", t->thread_id, t->device_id);
    testKernel<<<1,1>>>(t->thread_id, t->device_id);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[Thread %d] Error after second kernel launch: %s\n",
                t->thread_id, cudaGetErrorString(err));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    printf("[Thread %d] Finished.\n", t->thread_id);
    return NULL;
}

int main()
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }
    printf("Detected %d CUDA device(s).\n", device_count);

    // For demonstration, create as many threads as there are devices, up to 4
    int num_threads = device_count < 4 ? device_count : 4;
    pthread_t threads[num_threads];
    thread_arg_t args[num_threads];

    // Assign each thread a unique device. If more threads than devices, some will share.
    for (int i = 0; i < num_threads; ++i) {
        args[i].thread_id = i;
        args[i].device_id = i % device_count; // cycle if more threads
        args[i].reset_flag = (i == 1); // Let thread 1 perform cudaDeviceReset()
        int rc = pthread_create(&threads[i], NULL, thread_func, &args[i]);
        if (rc) {
            fprintf(stderr, "Error creating thread %d: %s\n", i, strerror(rc));
            return EXIT_FAILURE;
        }
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    printf("All threads completed.\n");
    return EXIT_SUCCESS;
}
```