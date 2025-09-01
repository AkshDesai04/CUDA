/*
Is `cudaDeviceReset()` necessary for a program that cleans up all its resources properly with `cudaFree`? Explain in a comment.

The question is asking whether, after a CUDA program has explicitly freed every allocation it made using `cudaFree`, it still needs to call `cudaDeviceReset()` before it exits.  The answer is that, in most cases, it is not strictly required.  `cudaFree` releases all device memory that was allocated by the host, and the CUDA driver will automatically reclaim other per‑process resources (like streams, events, and context state) when the process terminates.  Therefore, a program that cleans up all its memory and other handles will normally exit cleanly without an explicit reset.

However, there are a few reasons one might still want to call `cudaDeviceReset()`:

1. **Explicit cleanup for debugging** – In development or profiling scenarios, a reset guarantees that the driver state is returned to a clean, initial state.  Some tools (e.g., NSight) may rely on a reset to report that no dangling resources remain.

2. **Reusing the same process for multiple kernel launches** – If a long‑running host process repeatedly creates and destroys contexts or re‑initializes CUDA, a reset can ensure that no old context or stream state lingers between runs.

3. **Platform‑specific quirks** – Certain older GPUs or drivers had bugs that could be avoided by resetting the device at the end of a program.

In a simple short‑lived application that only allocates, uses, and frees memory, you can omit `cudaDeviceReset()` without any negative effect.  For safety and to make the intent explicit, many codebases still call it just before exiting.

The following example demonstrates a minimal CUDA program that allocates and frees memory.  It does *not* call `cudaDeviceReset()`, illustrating that the cleanup is sufficient.  A commented-out line shows where a reset could be added if desired.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void vecAdd(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main()
{
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(N - i);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify a few results
    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            ok = false;
            break;
        }
    }
    printf("Result %s\n", ok ? "OK" : "FAIL");

    // Clean up device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Clean up host memory
    free(h_a);
    free(h_b);
    free(h_c);

    // OPTIONAL: reset the device to clean up all context state
    // This is not strictly necessary if all resources were freed.
    // cudaDeviceReset();

    return 0;
}
