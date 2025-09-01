/*
Write a kernel that modifies the data on the device (e.g., doubles every element) before you copy it back. Verify the result is as expected.

Thinking process:
1. The goal is to demonstrate a simple CUDA kernel that operates on an array on the device, modifies it (doubling each element), and then copies the data back to host for verification.
2. Host side will:
   - Allocate an array of floats of size N.
   - Initialize the array with known values (e.g., 0.0f to N-1.0f).
   - Allocate device memory and copy the host array to device.
   - Launch the kernel with a suitable grid and block configuration.
   - Copy the modified array back to host.
   - Verify that each element was doubled by comparing with expected value.
   - Print success or failure message.
3. The kernel will simply compute the global thread index and, if it falls within the array bounds, multiply the element by 2.0f.
4. CUDA error checking macro will be used to catch errors from API calls and kernel launch.
5. All resources (device memory) will be freed at the end.
6. The program will be self‑contained in a single .cu file and compile with `nvcc`.
*/

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",           \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

__global__ void doubleElements(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

int main() {
    const int N = 1024;
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_data = (float *)malloc(size);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_data, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    doubleElements<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy modified data back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify the results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = static_cast<float>(i) * 2.0f;
        if (h_data[i] != expected) {
            fprintf(stderr, "Verification failed at index %d: got %f, expected %f\n",
                    i, h_data[i], expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Verification succeeded: all elements were doubled correctly.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
