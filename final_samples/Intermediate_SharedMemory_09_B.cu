```cuda
/*********************************************************************
 * Aim of the program (exactly as requested):
 * "A 2-way bank conflict occurs if two threads in a warp access different locations in the same bank. 
 * A 32-way conflict is the worst case. Explain this."
 *
 * Thinking and explanation:
 * 
 * In CUDA, shared memory is divided into a number of equally sized memory banks (for 
 * compute capability 2.x and 3.x there are 16 banks, for 5.x and higher there are 32 banks). 
 * When a warp of 32 threads performs a load or store, the hardware will route each 
 * thread's request to the appropriate bank based on the address modulo the number of banks. 
 * 
 * If all 32 threads access the same bank, all requests are serialized and the warp 
 * takes 32 cycles. If each thread accesses a different bank (perfectly aligned), all 
 * 32 requests are serviced in parallel and the warp completes in a single cycle. 
 * The number of threads that access the same bank determines the degree of bank conflict. 
 * 
 * Example: Suppose shared array s[32] is declared as float s[32]; The address of s[i] 
 * modulo 16 (for a 16‑bank device) determines the bank. If each thread t accesses 
 * s[t], then each thread accesses a different bank (t mod 16) and the access is conflict‑free. 
 * If instead each thread accesses s[(t + 1) % 32], threads 0–15 map to bank 1, 16–31 map to bank 0. 
 * Now two threads (0 and 16) share bank 1, two threads (1 and 17) share bank 1, etc. 
 * This is a 2‑way conflict: each bank sees two simultaneous requests, so each pair is serialized 
 * into two cycles. The warp still finishes in 2 cycles rather than 32. 
 * 
 * Worst case: all 32 threads access the same bank (e.g., each thread accesses s[0] or s[16]). 
 * Then the warp completes in 32 cycles – a 32‑way conflict. 
 * 
 * The code below demonstrates these scenarios using shared memory. The kernel performs a 
 * dummy reduction of the shared array but the pattern of writes/reads causes either 2‑way or 
 * 32‑way bank conflicts. The host code launches two kernels: one that accesses data in a 
 * conflict‑free manner, one that intentionally creates a 2‑way conflict, and one that 
 * creates a 32‑way conflict. The elapsed time measured by CUDA events is printed to show 
 * the impact of bank conflicts.
 *********************************************************************/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void conflictFreeKernel(float *out, int N)
{
    // Each block processes N elements (N = blockDim.x)
    extern __shared__ float s[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) s[tid] = out[idx];
    __syncthreads();

    // Simple reduction (conflict‑free access pattern)
    for (int stride = 1; stride < blockDim.x; stride <<= 1)
    {
        if (tid % (2 * stride) == 0)
        {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = s[0];
    }
}

__global__ void twoWayConflictKernel(float *out, int N)
{
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Induce a 2‑way conflict: threads (i, i+16) access the same bank
    // For compute capability >= 3.0 there are 16 banks
    // Access pattern: s[(tid + 1) % blockDim.x]
    int conflictIdx = (tid + 1) % blockDim.x;
    if (idx < N) s[conflictIdx] = out[idx];
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride <<= 1)
    {
        if (tid % (2 * stride) == 0)
        {
            s[conflictIdx] += s[(conflictIdx + stride) % blockDim.x];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = s[0];
    }
}

__global__ void thirtyTwoWayConflictKernel(float *out, int N)
{
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Induce a 32‑way conflict: all threads write/read to the same shared element
    if (idx < N) s[0] = out[idx];
    __syncthreads();

    // Simple accumulation
    for (int stride = 1; stride < blockDim.x; stride <<= 1)
    {
        if (tid % (2 * stride) == 0)
        {
            s[0] += s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = s[0];
    }
}

void launchKernel(void (*kernel)(float*,int), const char *name, int N)
{
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    // Initialize data
    float *h_in = new float[N];
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f; // simple data
    cudaMemcpy(d_out, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_in;

    dim3 blockDim(32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    size_t sharedSize = blockDim.x * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel<<<gridDim, blockDim, sharedSize>>>(d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%s: elapsed time = %f ms\n", name, ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);
}

int main()
{
    int N = 1 << 20; // 1M elements

    printf("Launching conflict‑free kernel...\n");
    launchKernel(conflictFreeKernel, "Conflict-Free", N);

    printf("Launching 2‑way conflict kernel...\n");
    launchKernel(twoWayConflictKernel, "2‑Way Conflict", N);

    printf("Launching 32‑way conflict kernel...\n");
    launchKernel(thirtyTwoWayConflictKernel, "32‑Way Conflict", N);

    return 0;
}
```