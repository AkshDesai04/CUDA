```cuda
/*
 * Aim: A 2-way bank conflict occurs if two threads in a warp access different
 * locations in the same bank. A 32-way conflict is the worst case. Explain this.
 *
 * Thinking process:
 * 1. In CUDA, shared memory is divided into 32 banks. Each bank can supply
 *    one 32‑bit word per cycle. When two or more threads of a warp request
 *    different words from the same bank, the requests are serialized and
 *    we get a bank conflict. A 2‑way conflict means two threads hit the
 *    same bank; a 32‑way conflict means all 32 threads hit the same bank,
 *    the worst possible case.
 * 2. The bank that a particular address falls into is given by
 *    bank = (address / 4) % 32 for 32‑bit words (assuming default layout).
 *    So indices that differ by multiples of 32 map to the same bank.
 * 3. To illustrate, we create a shared memory array of 64 integers.
 *    Thread i (0‑31) accesses shared[i] and shared[i+32].
 *    shared[i] -> bank = i % 32
 *    shared[i+32] -> bank = (i+32) % 32 = i % 32
 *    Thus every pair of threads accesses the same bank twice,
 *    producing a 2‑way conflict for each pair.
 * 4. We also demonstrate the worst case by having all 32 threads access
 *    the same bank: each thread reads shared[0] (bank 0). This is a
 *    32‑way conflict.
 * 5. The kernel will compute sums from these accesses and write them to
 *    global memory so we can run and observe that the program still
 *    works; the conflict only affects performance, not correctness.
 * 6. In a real profiling scenario, tools like nvprof or Nsight would
 *    show the increased instruction latency due to these conflicts.
 *
 * The program below is a minimal self‑contained CUDA program that
 * demonstrates the 2‑way conflict scenario and prints the results.
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void bankConflictKernel(int *d_out) {
    // Declare shared memory of 64 ints (enough for 2 accesses per thread)
    __shared__ int sdata[64];

    // Thread index within the warp
    int tid = threadIdx.x;

    // Initialize shared memory: just fill with thread index for clarity
    // Only one thread per warp needed for initialization; use thread 0
    if (tid == 0) {
        for (int i = 0; i < 64; ++i) {
            sdata[i] = i;
        }
    }
    __syncthreads();  // Ensure shared memory is ready

    // 2-way conflict pattern:
    // Each thread reads two different elements that map to the same bank
    int val1 = sdata[tid];          // bank = tid % 32
    int val2 = sdata[tid + 32];     // bank = (tid + 32) % 32 = tid % 32
    // Sum the two values
    int sum = val1 + val2;

    // Write result to global memory (one element per warp)
    if (tid == 0) {
        d_out[blockIdx.x] = sum; // sum of first thread's two values
    }
}

int main(void) {
    const int warpSize = 32;
    const int numWarps = 4; // small number for demonstration
    const int totalThreads = warpSize * numWarps;

    // Allocate output array on host
    int h_out[numWarps];
    int *d_out;

    cudaMalloc((void **)&d_out, numWarps * sizeof(int));

    // Launch kernel: one block, enough threads for multiple warps
    bankConflictKernel<<<1, totalThreads>>>(d_out);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_out, d_out, numWarps * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Results from 2-way conflict pattern (per warp sum):\n");
    for (int i = 0; i < numWarps; ++i) {
        printf("Warp %d: sum = %d\n", i, h_out[i]);
    }

    cudaFree(d_out);
    return 0;
}
```