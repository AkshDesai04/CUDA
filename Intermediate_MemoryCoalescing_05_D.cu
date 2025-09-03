```cuda
/*********************************************************************
 * Aim of the program (exactly as requested):
 *
 * Analyze why the SoA kernel has excellent memory performance (fully coalesced).
 *
 * -------------------------------------------------------------------
 * Thinking & Explanation:
 *
 * 1. Memory coalescing in CUDA occurs when consecutive threads access
 *    consecutive memory addresses. For a 32‑bit word, the hardware
 *    gathers data from 128 bytes (four 32‑bit words) per memory transaction.
 *
 * 2. In an Array of Structs (AoS) representation, each element contains
 *    multiple fields (x, y, z, id). When each thread reads the x, y, z
 *    components of its own particle, the memory accesses for x, y, z are
 *    strided: thread 0 reads x[0], y[0], z[0], thread 1 reads x[1], y[1],
 *    z[1], etc. For the x field, accesses are contiguous, but for y
 *    and z the accesses are offset by the size of a struct. This causes
 *    separate memory transactions for each field, and because the
 *    offsets are not aligned to 128‑byte boundaries the accesses can
 *    become non‑coalesced or split into multiple transactions.
 *
 * 3. In a Structure of Arrays (SoA) representation, each component is
 *    stored in its own contiguous array. For example, all x coordinates
 *    are in array x[], all y coordinates in y[], etc. When each thread
 *    accesses x[i], y[i], z[i], the accesses to the x array are
 *    perfectly contiguous and aligned. The same holds for y and z.
 *    Therefore each array access can be satisfied by a single
 *    coalesced transaction, maximizing memory throughput.
 *
 * 4. The program below demonstrates this difference by:
 *    - Allocating data in AoS and SoA form.
 *    - Launching two kernels that perform identical arithmetic
 *      (adding a constant to each component).
 *    - Timing each kernel with CUDA events.
 *    - Printing the execution time, allowing observation of the
 *      superior performance of the SoA kernel.
 *
 * 5. Note: The kernels are deliberately simple to avoid computation
 *    dominating memory latency, so the measured time mainly reflects
 *    memory bandwidth. In real applications, compute may dominate,
 *    but memory coalescing remains crucial for bandwidth‑bound
 *    workloads.
 *********************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N  (1 << 24)          // Number of particles (~16M)
#define BLOCK_SIZE 256        // Threads per block

/* ------------------------------------------------------------------ */
/* Struct definition for Array of Structs (AoS)                        */
/* ------------------------------------------------------------------ */
typedef struct {
    float x, y, z;
    int   id;
} Particle;

// Kernel operating on Array of Structs
__global__ void aos_kernel(Particle *particles, float inc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    // Simple arithmetic to force memory load/store
    particles[idx].x += inc;
    particles[idx].y += inc;
    particles[idx].z += inc;
    particles[idx].id += 1;
}

// Kernel operating on Structure of Arrays (SoA)
__global__ void soa_kernel(float *x, float *y, float *z, int *id, float inc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    x[idx] += inc;
    y[idx] += inc;
    z[idx] += inc;
    id[idx] += 1;
}

// Helper to check CUDA errors
void check_cuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    /* ------------------------------------------------------------ */
    /* Allocate host memory for AoS and SoA                          */
    /* ------------------------------------------------------------ */
    Particle *h_particles = (Particle *)malloc(N * sizeof(Particle));
    float *h_x = (float *)malloc(N * sizeof(float));
    float *h_y = (float *)malloc(N * sizeof(float));
    float *h_z = (float *)malloc(N * sizeof(float));
    int   *h_id = (int *)malloc(N * sizeof(int));

    /* Initialize data */
    for (int i = 0; i < N; ++i) {
        h_particles[i].x = (float)i;
        h_particles[i].y = (float)i + 1.0f;
        h_particles[i].z = (float)i + 2.0f;
        h_particles[i].id = i;

        h_x[i] = (float)i;
        h_y[i] = (float)i + 1.0f;
        h_z[i] = (float)i + 2.0f;
        h_id[i] = i;
    }

    /* ------------------------------------------------------------ */
    /* Allocate device memory                                       */
    /* ------------------------------------------------------------ */
    Particle *d_particles;
    float *d_x, *d_y, *d_z;
    int *d_id;

    check_cuda(cudaMalloc((void **)&d_particles, N * sizeof(Particle)), "cudaMalloc d_particles");
    check_cuda(cudaMalloc((void **)&d_x, N * sizeof(float)), "cudaMalloc d_x");
    check_cuda(cudaMalloc((void **)&d_y, N * sizeof(float)), "cudaMalloc d_y");
    check_cuda(cudaMalloc((void **)&d_z, N * sizeof(float)), "cudaMalloc d_z");
    check_cuda(cudaMalloc((void **)&d_id, N * sizeof(int)), "cudaMalloc d_id");

    /* Copy data to device */
    check_cuda(cudaMemcpy(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice), "Memcpy h->d particles");
    check_cuda(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice), "Memcpy h->d x");
    check_cuda(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice), "Memcpy h->d y");
    check_cuda(cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice), "Memcpy h->d z");
    check_cuda(cudaMemcpy(d_id, h_id, N * sizeof(int), cudaMemcpyHostToDevice), "Memcpy h->d id");

    /* ------------------------------------------------------------ */
    /* Launch kernels and time them                                 */
    /* ------------------------------------------------------------ */
    const float INC = 1.5f;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    float elapsed_ms;

    /* ---------------- AoS kernel ---------------- */
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    check_cuda(cudaEventRecord(start, 0), "cudaEventRecord start");
    aos_kernel<<<numBlocks, BLOCK_SIZE>>>(d_particles, INC);
    check_cuda(cudaEventRecord(stop, 0), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");

    printf("AoS kernel time: %f ms\n", elapsed_ms);

    /* ---------------- SoA kernel ---------------- */
    check_cuda(cudaEventRecord(start, 0), "cudaEventRecord start");
    soa_kernel<<<numBlocks, BLOCK_SIZE>>>(d_x, d_y, d_z, d_id, INC);
    check_cuda(cudaEventRecord(stop, 0), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");

    printf("SoA kernel time: %f ms\n", elapsed_ms);

    /* ------------------------------------------------------------ */
    /* Cleanup                                                      */
    /* ------------------------------------------------------------ */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_particles);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_id);

    free(h_particles);
    free(h_x);
    free(h_y);
    free(h_z);
    free(h_id);

    return 0;
}
```