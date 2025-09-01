/*
Aim of the program:
Create a GpuTimer class in C++. The constructor should call cudaEventCreate for start and stop events. A start() method calls cudaEventRecord. A stop() method calls cudaEventRecord and cudaEventSynchronize. An elapsed_ms() method calls cudaEventElapsedTime. Add error checking to all CUDA calls.

Thinking:
1. The class will manage two cudaEvent_t members: startEvent_ and stopEvent_.
2. In the constructor we need to create both events using cudaEventCreate. We'll wrap each CUDA call with error checking.
3. The destructor should destroy the events with cudaEventDestroy.
4. start() records the start event on the default stream.
5. stop() records the stop event and synchronizes the event to ensure completion.
6. elapsed_ms() computes the time difference via cudaEventElapsedTime and returns the elapsed milliseconds.
7. Error checking: Define a macro CUDA_CHECK that checks the return value of a CUDA runtime call. If an error occurs, print the error string and exit (or throw).
8. Include necessary headers: <cuda_runtime.h> and <cstdio> for printing.
9. All methods that interact with CUDA will be __host__ functions (since they run on the CPU side).
10. No other supporting code is required beyond the class definition.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",            \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                                \
    } while (0)

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&startEvent_));
        CUDA_CHECK(cudaEventCreate(&stopEvent_));
    }

    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(startEvent_));
        CUDA_CHECK(cudaEventDestroy(stopEvent_));
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(startEvent_, 0));
    }

    void stop() {
        CUDA_CHECK(cudaEventRecord(stopEvent_, 0));
        CUDA_CHECK(cudaEventSynchronize(stopEvent_));
    }

    float elapsed_ms() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent_, stopEvent_));
        return ms;
    }

private:
    cudaEvent_t startEvent_;
    cudaEvent_t stopEvent_;
};