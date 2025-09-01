```cuda
/*
Make the `stop()` method non-blocking by not calling `cudaEventSynchronize`. The user must call a separate `synchronize()` method before `elapsed_ms()` will be accurate.

Thinking:
- In CUDA, events can be recorded on streams and later queried for elapsed time. 
- The original design had `stop()` perform `cudaEventSynchronize`, making the call blocking and preventing asynchronous measurement.
- To allow non-blocking stops, we modify `stop()` to only record the stop event without synchronizing. 
- A separate `synchronize()` function will be exposed to the user; when called, it will perform `cudaEventSynchronize` on the stop event, ensuring all work up to the stop event is complete.
- `elapsed_ms()` will compute the elapsed time using `cudaEventElapsedTime`. If the stop event has not been synchronized, the result may be inaccurate or undefined. We enforce that by checking a flag and returning a sentinel value (-1.0f) to indicate the measurement is not ready.
- The implementation uses a plain C interface: a `Timer` struct and functions to create, start, stop, synchronize, measure, and destroy the timer. This keeps the code simple and CUDA-idiomatic for a C project.
- A simple example kernel and a `main()` function demonstrate how to use the timer: start, run a dummy kernel, stop (non-blocking), attempt to read elapsed time (should be -1.0), then synchronize and read the accurate elapsed time.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/*--------------------------------------------------
 * Timer struct and functions
 *--------------------------------------------------*/
typedef struct {
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    int is_synchronized;   // 0 = not synchronized, 1 = synchronized
} Timer;

/* Create a timer instance */
Timer* create_timer(void) {
    Timer *t = (Timer*)malloc(sizeof(Timer));
    if (!t) {
        fprintf(stderr, "Failed to allocate Timer\n");
        return NULL;
    }
    cudaEventCreate(&(t->startEvent));
    cudaEventCreate(&(t->stopEvent));
    t->is_synchronized = 0;
    return t;
}

/* Destroy a timer instance */
void destroy_timer(Timer *t) {
    if (!t) return;
    cudaEventDestroy(t->startEvent);
    cudaEventDestroy(t->stopEvent);
    free(t);
}

/* Record start event */
void timer_start(Timer *t) {
    if (!t) return;
    cudaEventRecord(t->startEvent, 0);
}

/* Record stop event (non-blocking) */
void timer_stop(Timer *t) {
    if (!t) return;
    cudaEventRecord(t->stopEvent, 0);
    // Do NOT synchronize here; keep it non-blocking
    t->is_synchronized = 0;
}

/* Synchronize the stop event (must be called before measuring) */
void timer_synchronize(Timer *t) {
    if (!t) return;
    cudaEventSynchronize(t->stopEvent);
    t->is_synchronized = 1;
}

/* Get elapsed time in milliseconds */
float timer_elapsed_ms(Timer *t) {
    if (!t) return -1.0f;
    if (!t->is_synchronized) {
        /* Measurement not ready; return sentinel */
        return -1.0f;
    }
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t->startEvent, t->stopEvent);
    return ms;
}

/*--------------------------------------------------
 * Example kernel (does nothing significant)
 *--------------------------------------------------*/
__global__ void dummy_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * idx;
    }
}

/*--------------------------------------------------
 * Main function demonstrating usage
 *--------------------------------------------------*/
int main(void) {
    const int N = 1 << 20;  // 1M elements
    int *d_data = NULL;
    cudaMalloc((void**)&d_data, N * sizeof(int));

    Timer *t = create_timer();
    if (!t) return EXIT_FAILURE;

    timer_start(t);

    /* Launch dummy kernel */
    dummy_kernel<<<(N + 255) / 256, 256>>>(d_data, N);
    cudaDeviceSynchronize();  // Ensure kernel launch is complete for demo

    timer_stop(t);  // Non-blocking stop

    /* Attempt to read elapsed time before synchronization */
    float ms_before = timer_elapsed_ms(t);
    if (ms_before < 0.0f)
        printf("Elapsed time before synchronize: not ready (sentinel %f ms)\n", ms_before);
    else
        printf("Elapsed time before synchronize: %f ms\n", ms_before);

    /* Now synchronize and read elapsed time */
    timer_synchronize(t);
    float ms_after = timer_elapsed_ms(t);
    printf("Elapsed time after synchronize: %f ms\n", ms_after);

    /* Clean up */
    destroy_timer(t);
    cudaFree(d_data);
    return EXIT_SUCCESS;
}
```