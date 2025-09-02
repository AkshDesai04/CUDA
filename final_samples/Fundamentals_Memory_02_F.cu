/*
Write a kernel that modifies the data on the device (e.g., doubles every element) before you copy it back. Verify the result is as expected.

Thinking:
The program demonstrates a typical CUDA workflow: allocate memory on the host, copy it to the device, launch a kernel that modifies the data in-place, copy the results back, and verify correctness.
- Create a small array of floats initialized on the host.
- Allocate device memory with cudaMalloc and transfer data using cudaMemcpy.
- Define a simple kernel that doubles each element with bounds checking.
- Launch the kernel with an appropriate grid/block configuration.
- Synchronize and check for launch errors.
- Copy the modified data back to the host.
- Verify each element matches the expected doubled value and report success or any mismatch.
- Include a convenient CUDA error‑checking macro for robustness.
- Keep the example straightforward so it can be compiled and run on any CUDA‑capable GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__,                    \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

__global__ void double_elements(float *d_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_arr[idx] *= 2.0f;
}

int main()
{
    const int N = 1024;
    std::vector<float> h_arr(N);
    for (int i = 0; i < N; ++i)
        h_arr[i] = static_cast<float>(i);

    float *d_arr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_arr, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    double_elements<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_arr.data(), d_arr, N * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N; ++i)
    {
        float expected = (static_cast<float>(i)) * 2.0f;
        if (h_arr[i] != expected)
        {
            printf("Mismatch at index %d: host=%f expected=%f\n", i, h_arr[i], expected);
            ok = false;
            break;
        }
    }

    if (ok)
        printf("All elements verified successfully.\n");
    else
        printf("Verification failed.\n");

    CUDA_CHECK(cudaFree(d_arr));
    return 0;
}