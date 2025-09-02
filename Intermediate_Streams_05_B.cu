```cpp
/*
The sequence is: `kernel<<<..., stream>>>();`, `cudaLaunchHostFunc(stream, my_callback, (void*)some_data);`.

This program demonstrates how to launch a CUDA kernel on a user‑defined stream and
immediately schedule a host callback to be executed once all preceding work on that
stream has completed.  The callback receives a pointer to user data (in this case
a small struct containing an integer), prints it, and frees the memory.  The main
routine allocates a simple integer array on the device, fills it with a kernel,
and after the callback synchronizes the stream and copies the data back to host
to verify the kernel execution.  Basic error checking is performed using a
helper macro.  The code is self‑contained and can be compiled with `nvcc`.
*/
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Simple kernel that writes the index value into the array
__global__ void fill_kernel(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] = idx;
    }
}

// Structure to hold data passed to the host callback
struct CallbackData {
    int value;
};

// Host callback function to be invoked after stream operations complete
void __host__ __cdecl my_callback(cudaStream_t stream, cudaError_t status, void *userData) {
    // Cast the user data pointer back to our struct
    CallbackData *cbData = static_cast<CallbackData*>(userData);
    if (status != cudaSuccess) {
        fprintf(stderr, "Callback error: %s\n", cudaGetErrorString(status));
    } else {
        printf("Callback executed. Received value = %d\n", cbData->value);
    }
    // Free the allocated memory for callback data
    free(cbData);
}

int main() {
    const int N = 256;
    int *d_arr = nullptr;
    int *h_arr = nullptr;
    cudaStream_t stream;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_arr, N * sizeof(int)));

    // Allocate host memory
    h_arr = (int*)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Create a non‑default stream
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Launch kernel on the custom stream
    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    fill_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_arr, N);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed\n");
        return EXIT_FAILURE;
    }

    // Allocate and set up data for the callback
    CallbackData *cbData = (CallbackData*)malloc(sizeof(CallbackData));
    if (!cbData) {
        fprintf(stderr, "Failed to allocate callback data\n");
        return EXIT_FAILURE;
    }
    cbData->value = 42;

    // Schedule host callback on the same stream
    CHECK_CUDA(cudaLaunchHostFunc(stream, my_callback, (void*)cbData));

    // Synchronize to ensure all operations (kernel + callback) are complete
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != i) {
            fprintf(stderr, "Verification failed at index %d: expected %d, got %d\n",
                    i, i, h_arr[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Kernel executed correctly. All values verified.\n");
    }

    // Clean up
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_arr));
    free(h_arr);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```