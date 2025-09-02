```cuda
/*
Aim of the program (passed in this prompt word to word):
Analyze why the SoA kernel has excellent memory performance (fully coalesced).

Thinking:
1. The goal of this program is to illustrate the memory performance
   advantage of a Structure of Arrays (SoA) layout over an Array of
   Structures (AoS) layout in a CUDA kernel.  We will implement a
   simple kernel that processes three float arrays (x, y, z) which
   together form a vector of 3D points.  In the SoA layout each
   coordinate type is stored in a separate contiguous array.  The
   kernel will read the coordinates for each point, compute the
   Euclidean norm, and write the result to an output array.

2. Because each thread accesses a single element from each of the
   three contiguous arrays, the global memory accesses are
   perfectly aligned and fully coalesced.  On modern NVIDIA GPUs,
   a coalesced access occurs when consecutive threads read or
   write consecutive memory addresses that are properly aligned.
   With SoA, thread i reads x[i], y[i], z[i] from three separate
   contiguous buffers – each of these accesses will hit the same
   memory transaction when the threads are launched in a contiguous
   block.

3. In contrast, if we used an AoS layout (e.g. struct Vec3 { float x, y, z; })
   the memory for a single point would be stored contiguously as
   x, y, z.  A thread reading the x component would read from
   the address of its own point, but the next thread would read from
   the x component of the next point, which is 12 bytes ahead.
   This pattern is still coalesced on devices with 32‑byte transaction
   granularity.  However, when accessing y and z components, the
   stride becomes 12 bytes, causing misaligned accesses and partial
   transactions.  Moreover, if the data structure were larger,
   the pattern would degrade further.  SoA keeps each coordinate
   access within its own contiguous buffer, avoiding strided
   accesses and ensuring that each memory transaction can service
   all threads in a warp.

4. Therefore the program demonstrates the SoA layout and prints
   the first few norms as a sanity check.  The comments in the
   kernel explain how the memory accesses are coalesced.

*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

/* Size of the dataset */
#define N 1024*1024   // 1M points

/* Kernel that computes the Euclidean norm of each 3D point
 * stored in SoA format (separate arrays for x, y, z).
 */
__global__ void computeNormSoA(const float *x, const float *y, const float *z, float *norm, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    /* Because x, y, z are separate contiguous arrays,
     * each thread accesses x[idx], y[idx], z[idx] which are
     * contiguous in memory for each array.  This ensures that
     * the 32 threads in a warp will read a contiguous 32*4 = 128
     * byte region from each array – a fully coalesced access.
     */
    float xi = x[idx];
    float yi = y[idx];
    float zi = z[idx];
    norm[idx] = sqrtf(xi*xi + yi*yi + zi*zi);
}

int main(void)
{
    /* Host allocations */
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));
    float *h_z = (float*)malloc(N * sizeof(float));
    float *h_norm = (float*)malloc(N * sizeof(float));

    /* Initialize host data */
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f;  // arbitrary values
        h_y[i] = 2.0f;
        h_z[i] = 3.0f;
    }

    /* Device allocations */
    float *d_x, *d_y, *d_z, *d_norm;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMalloc((void**)&d_z, N * sizeof(float));
    cudaMalloc((void**)&d_norm, N * sizeof(float));

    /* Copy data to device */
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice);

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    computeNormSoA<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, d_norm, N);
    cudaDeviceSynchronize();

    /* Copy results back to host */
    cudaMemcpy(h_norm, d_norm, N * sizeof(float), cudaMemcpyDeviceToHost);

    /* Print a few results for sanity check */
    for (int i = 0; i < 5; ++i) {
        printf("Point %d: norm = %f\n", i, h_norm[i]);
    }

    /* Clean up */
    free(h_x); free(h_y); free(h_z); free(h_norm);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_norm);

    return 0;
}
```