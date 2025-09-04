```cuda
/*

Aim of the program:
Extend the IPC example to share a CUDA event. Process A records an event, gets an IPC handle for it, passes it to Process B, which then waits on that event.

Thoughts:
This example demonstrates how to use CUDA IPC (Inter‑Process Communication) with events. Two processes are represented in the same executable, distinguished by a command‑line argument ("producer" or "consumer"). 
- The producer creates a CUDA event, records it on a stream, obtains its IPC handle using `cudaIpcGetEventHandle`, and writes that handle to a binary file (`event_handle.bin`). 
- The consumer reads the handle from the file, imports it with `cudaIpcOpenEventHandle`, and then waits for the event using `cudaEventSynchronize`. 

The file is used as a simple communication medium; in real applications you might use sockets, shared memory, or other IPC mechanisms. All CUDA API calls are checked for errors, and the program prints informative messages. The example compiles with `nvcc`.

*/

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unistd.h>

void checkCudaErr(CUresult err, const char* msg) {
    if (err != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(err, &errStr);
        std::cerr << msg << ": " << errStr << std::endl;
        exit(EXIT_FAILURE);
    }
}

void producer() {
    CUdevice device;
    CUcontext ctx;
    CUstream stream;
    CUevent event;
    cudaIpcEventHandle_t eventHandle;

    // Initialize CUDA
    checkCudaErr(cuInit(0), "cuInit");
    checkCudaErr(cuDeviceGet(&device, 0), "cuDeviceGet");
    checkCudaErr(cuCtxCreate(&ctx, 0, device), "cuCtxCreate");

    // Create a stream
    checkCudaErr(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");

    // Create an event
    checkCudaErr(cuEventCreate(&event, CU_EVENT_DEFAULT), "cuEventCreate");

    // Record the event on the stream
    checkCudaErr(cuEventRecord(event, stream), "cuEventRecord");

    // Get IPC handle for the event
    checkCudaErr(cuIpcGetEventHandle(&eventHandle, event), "cuIpcGetEventHandle");

    // Write the handle to a file
    std::ofstream ofs("event_handle.bin", std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open file for writing IPC handle." << std::endl;
        exit(EXIT_FAILURE);
    }
    ofs.write(reinterpret_cast<const char*>(&eventHandle), sizeof(eventHandle));
    ofs.close();
    std::cout << "Producer: IPC event handle written to 'event_handle.bin'.\n";

    // Optional: Wait a bit to keep the event alive
    sleep(1);

    // Clean up
    checkCudaErr(cuEventDestroy(event), "cuEventDestroy");
    checkCudaErr(cuStreamDestroy(stream), "cuStreamDestroy");
    checkCudaErr(cuCtxDestroy(ctx), "cuCtxDestroy");
}

void consumer() {
    CUdevice device;
    CUcontext ctx;
    CUevent event;

    // Initialize CUDA
    checkCudaErr(cuInit(0), "cuInit");
    checkCudaErr(cuDeviceGet(&device, 0), "cuDeviceGet");
    checkCudaErr(cuCtxCreate(&ctx, 0, device), "cuCtxCreate");

    // Read the IPC handle from the file
    std::ifstream ifs("event_handle.bin", std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open IPC handle file." << std::endl;
        exit(EXIT_FAILURE);
    }
    cudaIpcEventHandle_t eventHandle;
    ifs.read(reinterpret_cast<char*>(&eventHandle), sizeof(eventHandle));
    ifs.close();

    // Open the event handle
    checkCudaErr(cuIpcOpenEventHandle(&event, &eventHandle), "cuIpcOpenEventHandle");
    std::cout << "Consumer: IPC event handle opened.\n";

    // Wait for the event to complete
    std::cout << "Consumer: Waiting for event to complete...\n";
    checkCudaErr(cuEventSynchronize(event), "cuEventSynchronize");
    std::cout << "Consumer: Event completed.\n";

    // Clean up
    checkCudaErr(cuEventDestroy(event), "cuEventDestroy");
    checkCudaErr(cuCtxDestroy(ctx), "cuCtxDestroy");
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [producer|consumer]" << std::endl;
        return EXIT_FAILURE;
    }

    std::string role(argv[1]);
    if (role == "producer") {
        producer();
    } else if (role == "consumer") {
        consumer();
    } else {
        std::cerr << "Unknown role: " << role << ". Use 'producer' or 'consumer'." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```