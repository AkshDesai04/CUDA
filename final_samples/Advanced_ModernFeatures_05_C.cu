```cpp
/*
Aim of the program: Use IPC to implement a simple client-server model where a server process
manages a large dataset on the GPU, and multiple client processes can attach to its memory
to perform read-only computations.

Thinking and design decisions:

1.  The program is a single .cu file that can act either as a server or as a client
    depending on the first command‑line argument ("server" or "client").
2.  The server allocates a large array on the GPU, fills it with deterministic data,
    obtains an IPC memory handle using cudaIpcGetMemHandle, and writes the handle
    (along with the array size) to a binary file called "ipc_handle.bin".  
    The file format is:
        - 4 bytes: unsigned int containing the array size N
        - sizeof(cudaIpcMemHandle_t) bytes: the IPC handle
3.  The client reads that file, opens the IPC memory handle with cudaIpcOpenMemHandle,
    and then performs a read‑only computation – in this case a reduction that sums
    all array elements using a kernel that employs atomicAdd on a double accumulator.
4.  Multiple client processes can run concurrently because each client obtains its
    own device pointer from the IPC handle and does not modify the underlying memory.
5.  Basic error checking is performed with a CUDA_CHECK macro that aborts the program
    on any CUDA API failure.
6.  The program uses CUDA runtime API only, making it portable to any CUDA capable
    device.  The kernels are intentionally simple to keep the example focused on IPC.
7.  The server waits for the user to press ENTER before exiting, giving clients time
    to attach.  In a production system, a more robust synchronization mechanism
    (e.g., a named semaphore) would be used.

The code below can be compiled with nvcc and executed as follows:

    nvcc -o ipc_demo ipc_demo.cu
    ./ipc_demo server          # start the server
    ./ipc_demo client          # run a client that attaches and sums the data

*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>

#define HANDLE_FILE "ipc_handle.bin"

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

// Kernel to fill the array with deterministic values
__global__ void fillKernel(float *d_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_arr[idx] = static_cast<float>(idx);
}

// Kernel to sum the array elements (read‑only)
__global__ void sumKernel(const float *d_arr, int N, double *d_sum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        atomicAdd(d_sum, static_cast<double>(d_arr[idx]));
}

// Server code
void runServer(int N)
{
    std::cout << "[Server] Allocating " << N << " floats on GPU.\n";
    float *d_array = nullptr;
    CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(float)));

    // Launch kernel to fill array
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    fillKernel<<<blocks, threadsPerBlock>>>(d_array, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Obtain IPC handle
    cudaIpcMemHandle_t handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&handle, d_array));

    // Write size and handle to file
    std::ofstream ofs(HANDLE_FILE, std::ios::binary);
    if (!ofs) {
        std::cerr << "[Server] Failed to open " << HANDLE_FILE << " for writing.\n";
        exit(EXIT_FAILURE);
    }
    uint32_t size_to_write = static_cast<uint32_t>(N);
    ofs.write(reinterpret_cast<const char *>(&size_to_write), sizeof(size_to_write));
    ofs.write(reinterpret_cast<const char *>(&handle), sizeof(handle));
    ofs.close();

    std::cout << "[Server] IPC handle written to " << HANDLE_FILE << ".\n";
    std::cout << "[Server] Press ENTER to exit.\n";
    std::cin.get();  // wait for user to press ENTER

    // Clean up
    CUDA_CHECK(cudaFree(d_array));
    std::cout << "[Server] Exiting.\n";
}

// Client code
void runClient()
{
    // Read size and handle from file
    std::ifstream ifs(HANDLE_FILE, std::ios::binary);
    if (!ifs) {
        std::cerr << "[Client] Failed to open " << HANDLE_FILE << " for reading.\n";
        exit(EXIT_FAILURE);
    }
    uint32_t N = 0;
    ifs.read(reinterpret_cast<char *>(&N), sizeof(N));
    cudaIpcMemHandle_t handle;
    ifs.read(reinterpret_cast<char *>(&handle), sizeof(handle));
    ifs.close();

    std::cout << "[Client] Read array size: " << N << "\n";

    // Open the IPC handle
    float *d_array = nullptr;
    CUDA_CHECK(cudaIpcOpenMemHandle((void **)&d_array, handle, cudaIpcMemLazyEnablePeerAccess));

    // Allocate device memory for sum
    double *d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));

    // Launch sum kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    sumKernel<<<blocks, threadsPerBlock>>>(d_array, N, d_sum);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy sum back to host
    double h_sum = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "[Client] Sum of array elements = " << h_sum << "\n";

    // Clean up
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaIpcCloseMemHandle(d_array));
    std::cout << "[Client] Exiting.\n";
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " server [array_size] | client\n";
        return EXIT_FAILURE;
    }

    std::string mode = argv[1];
    if (mode == "server") {
        int N = 1 << 24; // default: 16M floats (~64 MB)
        if (argc >= 3) {
            N = std::atoi(argv[2]);
            if (N <= 0) {
                std::cerr << "Invalid array size.\n";
                return EXIT_FAILURE;
            }
        }
        runServer(N);
    } else if (mode == "client") {
        runClient();
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```