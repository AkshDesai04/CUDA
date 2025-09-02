```cpp
/*
Aim of the program:
Implement a simple program that uses CUDA Inter-Process Communication (IPC). 
Process A allocates a chunk of GPU memory and gets an IPC handle for it (`cudaIpcGetMemHandle`). 
Process A passes this handle to Process B (e.g., via a file). 
Process B uses the handle to open the memory (`cudaIpcOpenMemHandle`) and can then read/write to Process A's GPU memory directly.

Thinking:
- The program will be a single source file that can run in two modes, A or B, selected by a command line argument.
- Process A:
  * Allocate a 1K float array on the GPU.
  * Initialize it on the host with 42.0f, copy to device.
  * Obtain a cudaIpcMemHandle_t via cudaIpcGetMemHandle.
  * Persist the handle to a binary file (handle.bin) so that Process B can read it.
  * Sleep for a short period to allow Process B to run.
  * After the sleep, copy the device memory back to host and print the first element to show changes made by B.
  * Clean up: free device memory and remove the handle file.
- Process B:
  * Read the handle from handle.bin.
  * Open the remote memory with cudaIpcOpenMemHandle.
  * Copy the device memory to a host buffer and print the first element (should be 42.0f).
  * Launch a simple kernel that adds 1.0f to each element.
  * Copy the memory back to host and print the first element again (should be 43.0f).
  * Close the IPC handle with cudaIpcCloseMemHandle and free host memory.
- Error handling is performed via a CUDA_CHECK macro that aborts on any CUDA error.
- The code uses standard C file IO for handle persistence and the POSIX sleep() function for the delay.
- The program should be compiled with nvcc and can be executed as:
    nvcc -o ipc_demo ipc_demo.cu
    ./ipc_demo A   (run Process A)
    ./ipc_demo B   (run Process B)
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>  // for sleep()

// CUDA runtime
#include <cuda_runtime.h>

// Macro to check CUDA errors
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Simple kernel to add 1.0f to each element
__global__ void add_one_kernel(float *d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] += 1.0f;
    }
}

// Write a cudaIpcMemHandle_t to a binary file
bool write_handle_to_file(const char *filename, const cudaIpcMemHandle_t &handle) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror("Failed to open file for writing handle");
        return false;
    }
    size_t written = fwrite(&handle, sizeof(handle), 1, f);
    fclose(f);
    return (written == 1);
}

// Read a cudaIpcMemHandle_t from a binary file
bool read_handle_from_file(const char *filename, cudaIpcMemHandle_t &handle) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open file for reading handle");
        return false;
    }
    size_t read = fread(&handle, sizeof(handle), 1, f);
    fclose(f);
    return (read == 1);
}

// Process A: Allocate, init, export IPC handle, wait, read back
void run_process_A() {
    const int N = 1024;
    const char *handle_file = "handle.bin";

    // Allocate device memory
    float *d_buf = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_buf, N * sizeof(float)));

    // Host buffer initialization
    float *h_buf = (float *)malloc(N * sizeof(float));
    if (!h_buf) {
        fprintf(stderr, "Failed to allocate host buffer\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < N; ++i) {
        h_buf[i] = 42.0f;
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, N * sizeof(float), cudaMemcpyHostToDevice));

    // Get IPC handle
    cudaIpcMemHandle_t handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&handle, d_buf));

    // Write handle to file
    if (!write_handle_to_file(handle_file, handle)) {
        fprintf(stderr, "Failed to write handle to file\n");
        exit(EXIT_FAILURE);
    }
    printf("[Process A] IPC handle written to %s\n", handle_file);

    // Clean up host buffer
    free(h_buf);

    // Sleep to allow Process B to run
    printf("[Process A] Sleeping for 10 seconds to allow Process B to run...\n");
    sleep(10);

    // After B has potentially modified the memory, read back to host
    h_buf = (float *)malloc(N * sizeof(float));
    if (!h_buf) {
        fprintf(stderr, "Failed to allocate host buffer for readback\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(h_buf, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("[Process A] First element after B's execution: %f\n", h_buf[0]);

    // Clean up
    free(h_buf);
    CUDA_CHECK(cudaFree(d_buf));

    // Remove handle file
    if (remove(handle_file) != 0) {
        perror("Failed to remove handle file");
    } else {
        printf("[Process A] Handle file removed.\n");
    }
}

// Process B: Read IPC handle, open, read/write, close
void run_process_B() {
    const int N = 1024;
    const char *handle_file = "handle.bin";

    // Read handle from file
    cudaIpcMemHandle_t handle;
    if (!read_handle_from_file(handle_file, handle)) {
        fprintf(stderr, "Failed to read handle from file\n");
        exit(EXIT_FAILURE);
    }
    printf("[Process B] IPC handle read from %s\n", handle_file);

    // Open the remote memory
    float *d_buf = nullptr;
    CUDA_CHECK(cudaIpcOpenMemHandle((void **)&d_buf, handle,
                                    cudaIpcMemLazyEnablePeerAccess));
    printf("[Process B] Remote memory opened.\n");

    // Copy device memory to host for inspection
    float *h_buf = (float *)malloc(N * sizeof(float));
    if (!h_buf) {
        fprintf(stderr, "Failed to allocate host buffer\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(h_buf, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("[Process B] First element before modification: %f\n", h_buf[0]);

    // Launch kernel to add 1.0f to each element
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    add_one_kernel<<<blocks, threads_per_block>>>(d_buf, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back to host to verify
    CUDA_CHECK(cudaMemcpy(h_buf, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("[Process B] First element after modification: %f\n", h_buf[0]);

    // Clean up
    free(h_buf);
    CUDA_CHECK(cudaIpcCloseMemHandle(d_buf));
    printf("[Process B] IPC handle closed.\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [A|B]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (strcmp(argv[1], "A") == 0) {
        printf("[Main] Running as Process A\n");
        run_process_A();
    } else if (strcmp(argv[1], "B") == 0) {
        printf("[Main] Running as Process B\n");
        run_process_B();
    } else {
        fprintf(stderr, "Invalid argument: %s. Use 'A' or 'B'.\n", argv[1]);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```