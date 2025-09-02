/*
Add a `reset()` method that allows the timer to be reused without creating a new object.

Thinking:
- We'll create a GPU timer class using cudaEvent_t for start and stop events.
- The reset method will destroy current events and create fresh ones so that the
  timer can be reused. This avoids the overhead of allocating a new Timer object
  each time we want to time a new kernel launch.
- We'll implement a simple element‑wise addition kernel and a main function
  that demonstrates starting, stopping, getting elapsed time, resetting, and
  timing a second launch with the same timer instance.
- Basic CUDA error checking is provided via a macro to keep the code clean.
- The entire program is self‑contained and should compile as a single .cu file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Basic CUDA error checking macro */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* GPU timer class using CUDA events */
class GPUTimer {
private:
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

public:
    /* Constructor: create events */
    GPUTimer() {
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));
    }

    /* Destructor: destroy events */
    ~GPUTimer() {
        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaEventDestroy(stopEvent));
    }

    /* Start the timer */
    void start() {
        CUDA_CHECK(cudaEventRecord(startEvent, 0));
    }

    /* Stop the timer */
    void stop() {
        CUDA_CHECK(cudaEventRecord(stopEvent, 0));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));
    }

    /* Return elapsed time in milliseconds */
    float elapsed() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
        return ms;
    }

    /* Reset the timer so it can be reused */
    void reset() {
        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaEventDestroy(stopEvent));
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));
    }
};

/* Simple kernel that adds two arrays */
__global__ void addKernel(int *c, const int *a, const int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

/* Main function demonstrating timer usage */
int main() {
    const int N = 1 << 20;               // 1M elements
    const size_t bytes = N * sizeof(int);

    /* Host memory allocation */
    int *h_a = (int *)malloc(bytes);
    int *h_b = (int *)malloc(bytes);
    int *h_c = (int *)malloc(bytes);

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = N - i;
    }

    /* Device memory allocation */
    int *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* Create a timer instance */
    GPUTimer timer;

    /* First kernel launch */
    timer.start();
    addKernel<<<(N + 255) / 256, 256>>>(d_c, d_a, d_b, N);
    CUDA_CHECK(cudaGetLastError());
    timer.stop();
    printf("First kernel elapsed time: %f ms\n", timer.elapsed());

    /* Reset the timer for reuse */
    timer.reset();

    /* Second kernel launch (could be the same or different work) */
    timer.start();
    addKernel<<<(N + 255) / 256, 256>>>(d_c, d_a, d_b, N);
    CUDA_CHECK(cudaGetLastError());
    timer.stop();
    printf("Second kernel elapsed time: %f ms\n", timer.elapsed());

    /* Copy result back to host (optional) */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    for (int i = 0; i < 10; ++i) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
