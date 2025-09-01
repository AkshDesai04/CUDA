```cuda
// kernel.cu
/* 
Write the kernel definition in a separate `.cu` file and the `main` function in another. Try to compile and link them together with `nvcc`. 
Thought process: The goal is to demonstrate that a CUDA kernel can live in its own .cu file and be linked together with a host main program written in a separate .cu file. The simplest, most illustrative example is a vector addition kernel that adds two float arrays element‑wise. The kernel will be defined with __global__ and perform a single addition per thread. The main program will allocate memory on both host and device, copy data to the device, launch the kernel, copy the results back, and print them. NVCC can compile and link multiple .cu files simply by listing them on the command line, e.g. `nvcc -o vecadd main.cu kernel.cu`. No external headers are required beyond <cuda_runtime.h> and <stdio.h>, so the program is self‑contained and portable across CUDA-enabled GPUs. 
*/

#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}
```

```cuda
// main.cu
/* 
Write the kernel definition in a separate `.cu` file and the `main` function in another. Try to compile and link them together with `nvcc`. 
Thought process: This file contains the host side code that will call the kernel defined in kernel.cu. We declare the kernel prototype, set up host and device arrays, copy data to the device, configure launch parameters, invoke the kernel, copy results back, and clean up. The program is deliberately simple to focus on the compile‑and‑link aspect of multiple CUDA source files. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Forward declaration of the kernel defined in kernel.cu */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N);

int main(void)
{
    const int N = 10;
    const int SIZE = N * sizeof(float);
    float h_A[N], h_B[N], h_C[N];
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    int i;

    /* Initialize host arrays */
    for (i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    /* Allocate device memory */
    cudaMalloc((void**)&d_A, SIZE);
    cudaMalloc((void**)&d_B, SIZE);
    cudaMalloc((void**)&d_C, SIZE);

    /* Copy host memory to device */
    cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice);

    /* Launch kernel: one thread per element */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Wait for GPU to finish */
    cudaDeviceSynchronize();

    /* Copy result back to host */
    cudaMemcpy(h_C, d_C, SIZE, cudaMemcpyDeviceToHost);

    /* Print results */
    printf("Result of vector addition (A + B):\n");
    for (i = 0; i < N; ++i) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
    }

    /* Clean up device memory */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```