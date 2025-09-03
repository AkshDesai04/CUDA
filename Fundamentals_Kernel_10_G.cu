```cuda
/*
Aim: Compare passing a struct with 3 floats vs. passing 3 separate float arguments to a kernel. Is there a performance difference? (Likely negligible, but good to think about).

Thinking:
The question concerns the overhead of passing a small struct (containing three floats) to a CUDA kernel versus passing three separate float arguments. The kernel arguments are transferred from the host to the device at launch time; for a tiny number of floats this transfer is negligible compared to the GPU execution time. To empirically verify this, we can:

1. Define a simple struct `Vec3` with three floats.
2. Create two kernels:
   - `kernelStruct` that takes two `Vec3` arguments by value.
   - `kernelFloat` that takes six `float` arguments (three per vector) by value.
3. Each kernel performs the same dot‑product operation inside a tight loop to ensure the kernel body is non‑trivial and not optimized away.
4. The result of each kernel is written to a single‑element global array so the compiler cannot discard the computation.
5. Measure the elapsed time for each kernel launch using CUDA events. To obtain a stable measurement we perform many iterations (e.g., 10,000 launches) and average the times.
6. Print the average launch times for the struct and float kernels.

Because the struct is passed by value, the compiler will copy its three floats into registers. Passing three separate floats also copies them into registers. The difference in the generated machine code should be minimal, resulting in essentially the same launch overhead. The measured times should confirm this expectation, showing negligible difference.

*/

#include <cstdio>
#include <cuda_runtime.h>

#define NUM_ITERATIONS 10000
#define LOOP_COUNT 1000

// Struct containing three floats
struct Vec3 {
    float x, y, z;
};

// Kernel that receives two Vec3 structs by value
__global__ void kernelStruct(Vec3 a, Vec3 b, float* res)
{
    float dot = 0.0f;
    for (int i = 0; i < LOOP_COUNT; ++i) {
        dot = a.x * b.x + a.y * b.y + a.z * b.z;
    }
    *res = dot;
}

// Kernel that receives six float arguments by value
__global__ void kernelFloat(float ax, float ay, float az,
                            float bx, float by, float bz,
                            float* res)
{
    float dot = 0.0f;
    for (int i = 0; i < LOOP_COUNT; ++i) {
        dot = ax * bx + ay * by + az * bz;
    }
    *res = dot;
}

int main()
{
    // Host data
    Vec3 hostVecA = {1.0f, 2.0f, 3.0f};
    Vec3 hostVecB = {4.0f, 5.0f, 6.0f};

    // Device result array
    float *devRes;
    cudaMalloc(&devRes, sizeof(float));

    // Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float timeStruct = 0.0f;
    float timeFloat  = 0.0f;

    // Run struct kernel multiple times and time
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        cudaEventRecord(start, 0);
        kernelStruct<<<1, 1>>>(hostVecA, hostVecB, devRes);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        timeStruct += milliseconds;
    }

    // Run float kernel multiple times and time
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        cudaEventRecord(start, 0);
        kernelFloat<<<1, 1>>>(hostVecA.x, hostVecA.y, hostVecA.z,
                             hostVecB.x, hostVecB.y, hostVecB.z,
                             devRes);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        timeFloat += milliseconds;
    }

    // Compute average times
    timeStruct /= NUM_ITERATIONS;
    timeFloat  /= NUM_ITERATIONS;

    printf("Average kernel launch time (Struct): %.6f ms\n", timeStruct);
    printf("Average kernel launch time (Floats): %.6f ms\n", timeFloat);

    // Clean up
    cudaFree(devRes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```