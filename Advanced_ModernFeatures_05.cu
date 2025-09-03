/*
Implement a simple program that uses CUDA Inter-Process Communication (IPC). Process A allocates a chunk of GPU memory and gets an IPC handle for it (`cudaIpcGetMemHandle`). Process A passes this handle to Process B (e.g., via a file). Process B uses the handle to open the memory (`cudaIpcOpenMemHandle`) and can then read/write to Process A's GPU memory directly.

Thinking:
- I want a single .cu file that can act as either Process A or Process B depending on a command-line argument.
- Process A: allocate an array on the GPU, fill it with some known values (e.g., index values), obtain an IPC handle with `cudaIpcGetMemHandle`, write that handle to a temporary file (e.g., "ipc_handle.bin"). Then wait for a short time to give Process B time to read.
- Process B: read the handle from the file, open it with `cudaIpcOpenMemHandle`, copy the memory to host and print the values to confirm communication. Optionally, modify the memory to demonstrate write-back.
- Error checking: use helper macro `checkCuda` to wrap CUDA calls and exit on failure.
- Use `<fstream>` for file I/O, `<vector>` for host memory.
- Ensure proper cleanup: close the IPC handle with `cudaIpcCloseMemHandle` and free GPU memory in Process A.
- Provide simple usage instructions via `printf` in code, but not extra text outside the comment.
- The file is self-contained; compiling with `nvcc ipc_example.cu -o ipc_example` will produce the executable.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",     \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call);\
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Simple kernel to initialize GPU array
__global__ void init_kernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = idx;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <A|B>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *role = argv[1];
    const char *handleFile = "ipc_handle.bin";
    const int N = 1024; // number of ints

    if (role[0] == 'A' || role[0] == 'a') {
        // Process A
        int *d_ptr = nullptr;
        CHECK_CUDA(cudaMalloc(&d_ptr, N * sizeof(int)));
        // Initialize data
        init_kernel<<<(N + 255) / 256, 256>>>(d_ptr, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Obtain IPC handle
        cudaIpcMemHandle_t handle;
        CHECK_CUDA(cudaIpcGetMemHandle(&handle, d_ptr));

        // Write handle to file
        std::ofstream ofs(handleFile, std::ios::binary);
        if (!ofs) {
            fprintf(stderr, "Failed to open file for writing IPC handle.\n");
            return EXIT_FAILURE;
        }
        ofs.write(reinterpret_cast<const char *>(&handle), sizeof(handle));
        ofs.close();

        printf("Process A: IPC handle written to %s. Waiting for Process B to finish.\n", handleFile);

        // Sleep to allow Process B to run
        std::this_thread::sleep_for(std::chrono::seconds(10));

        // Clean up
        CHECK_CUDA(cudaFree(d_ptr));
        printf("Process A: Finished.\n");
    } else if (role[0] == 'B' || role[0] == 'b') {
        // Process B
        // Read handle from file
        std::ifstream ifs(handleFile, std::ios::binary);
        if (!ifs) {
            fprintf(stderr, "Failed to open file for reading IPC handle.\n");
            return EXIT_FAILURE;
        }
        cudaIpcMemHandle_t handle;
        ifs.read(reinterpret_cast<char *>(&handle), sizeof(handle));
        ifs.close();

        // Open the memory
        int *d_ptr = nullptr;
        CHECK_CUDA(cudaIpcOpenMemHandle(reinterpret_cast<void **>(&d_ptr), handle, cudaIpcMemLazyEnablePeerAccess));

        // Copy data back to host
        std::vector<int> host(N);
        CHECK_CUDA(cudaMemcpy(host.data(), d_ptr, N * sizeof(int), cudaMemcpyDeviceToHost));

        printf("Process B: Received data:\n");
        for (int i = 0; i < 10; ++i) { // print first 10 elements
            printf("%d ", host[i]);
        }
        printf("\n");

        // Optionally modify the memory
        std::vector<int> modify(N, 42);
        CHECK_CUDA(cudaMemcpy(d_ptr, modify.data(), N * sizeof(int), cudaMemcpyHostToDevice));

        // Close IPC handle
        CHECK_CUDA(cudaIpcCloseMemHandle(d_ptr));
        printf("Process B: Modified data and closed IPC handle.\n");
    } else {
        fprintf(stderr, "Unknown role: %s\n", role);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
