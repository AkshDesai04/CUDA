/*
Define a struct with many members. Time the difference between copying the whole struct vs. using `cudaMemcpy` on individual members (the former is much better).

Thinking:
To demonstrate the performance difference between copying an entire struct at once and copying each member individually, we will:
1. Define a `BigStruct` containing a large array of floats. Each element of this array will be treated as a separate member for the purpose of individual copying.
2. Allocate memory on both host and device for this struct.
3. Measure the time to copy the entire struct from host to device using a single `cudaMemcpy`.
4. Measure the time to copy each float element individually from host to device by performing many small `cudaMemcpy` calls (one per element).
5. Use CUDA events (`cudaEvent_t`) to accurately time the operations on the GPU.
6. Print out the elapsed times to show that copying the whole struct is significantly faster than performing many individual copies.
7. (Optional) Verify correctness by copying the struct back to the host and comparing values, but this is not strictly required for the timing demonstration.

This program focuses on the timing comparison; it does not involve any kernel execution or device-side computation.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <cuda_runtime.h>

#define N 10000   // number of float members

// Simple error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Define the struct with many members (represented here as a large array)
struct BigStruct
{
    float m[N];
};

int main()
{
    // Host allocation and initialization
    BigStruct hStruct;
    for (int i = 0; i < N; ++i)
        hStruct.m[i] = (float)i;

    // Device allocation
    BigStruct *dStruct;
    gpuErrchk(cudaMalloc((void**)&dStruct, sizeof(BigStruct)));

    // CUDA events for timing
    cudaEvent_t startWhole, stopWhole;
    cudaEvent_t startInd, stopInd;
    gpuErrchk(cudaEventCreate(&startWhole));
    gpuErrchk(cudaEventCreate(&stopWhole));
    gpuErrchk(cudaEventCreate(&startInd));
    gpuErrchk(cudaEventCreate(&stopInd));

    // ----- Copy whole struct -----
    gpuErrchk(cudaEventRecord(startWhole, 0));
    gpuErrchk(cudaMemcpy(dStruct, &hStruct, sizeof(BigStruct), cudaMemcpyHostToDevice));
    gpuErrchk(cudaEventRecord(stopWhole, 0));
    gpuErrchk(cudaEventSynchronize(stopWhole));

    float msWhole = 0.0f;
    gpuErrchk(cudaEventElapsedTime(&msWhole, startWhole, stopWhole));

    // ----- Copy individual members -----
    gpuErrchk(cudaEventRecord(startInd, 0));
    for (int i = 0; i < N; ++i)
    {
        // Compute byte offset for the i-th float in the struct
        size_t offset = offsetof(BigStruct, m) + i * sizeof(float);
        gpuErrchk(cudaMemcpy((char*)dStruct + offset,
                             &hStruct.m[i],
                             sizeof(float),
                             cudaMemcpyHostToDevice));
    }
    gpuErrchk(cudaEventRecord(stopInd, 0));
    gpuErrchk(cudaEventSynchronize(stopInd));

    float msInd = 0.0f;
    gpuErrchk(cudaEventElapsedTime(&msInd, startInd, stopInd));

    // Output the timing results
    printf("Time to copy whole struct (ms): %f\n", msWhole);
    printf("Time to copy each member individually (ms): %f\n", msInd);

    // Clean up
    gpuErrchk(cudaEventDestroy(startWhole));
    gpuErrchk(cudaEventDestroy(stopWhole));
    gpuErrchk(cudaEventDestroy(startInd));
    gpuErrchk(cudaEventDestroy(stopInd));
    gpuErrchk(cudaFree(dStruct));

    return 0;
}
