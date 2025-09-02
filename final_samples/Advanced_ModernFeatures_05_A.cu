```cuda
/*
Aim of the program:
Extend the IPC example to share a CUDA event. Process A records an event, gets an IPC handle for it, passes it to Process B, which then waits on that event.

Thinking and design decisions:

1. Use the CUDA Runtime API because it already exposes the types
   `cudaIpcEventHandle_t`, `cudaIpcGetEventHandle` and
   `cudaIpcOpenEventHandle`.  This keeps the code short and portable.

2. The program acts as both Process A and Process B depending on the
   first command line argument.  If the argument is `"A"` the program
   creates an event, records it, obtains an IPC handle, and writes the
   raw handle to a file (`event_handle.bin`).  If the argument is `"B"`
   the program reads the file, opens the event via
   `cudaIpcOpenEventHandle`, and then synchronizes on it with
   `cudaEventSynchronize`.  When the synchronization returns we know
   that the event recorded by Process A has completed.

3. The IPC handle is a small structure (usually 8 bytes).  We write it
   to a binary file using `std::ofstream` and read it back with
   `std::ifstream`.  No network communication is required.

4. To ensure that the event has indeed been recorded before the IPC
   handle is written, Process A records the event on stream 0 and
   synchronizes on the stream before writing the handle.  This means
   that when Process B waits on the event, it will immediately see the
   event finished.  The example can be adapted to delay the event
   completion by launching a kernel, but that is unnecessary for a
   minimal demonstration.

5. Basic error checking is implemented via a helper macro `cudaCheck`
   that aborts the program on any CUDA error.

6. The program assumes that both processes run on the same device
   (device 0).  For a multi-GPU system, one can modify the device
   index or pass it as a command line argument.

Compile with:
    nvcc -o ipc_event ipc_event.cu

Run Process A:
    ./ipc_event A

Run Process B (after A has finished writing the handle):
    ./ipc_event B
*/

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>

#define cudaCheck(call)                                            \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error in " << #call << ": "        \
                      << cudaGetErrorString(err) << " (file "     \
                      << __FILE__ << ", line " << __LINE__ << ")\n"; \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    } while (0)

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [A|B]\n";
        return EXIT_FAILURE;
    }

    const char *mode = argv[1];
    const int device = 0;
    cudaCheck(cudaSetDevice(device));

    if (std::strcmp(mode, "A") == 0) {
        // Process A: create event, record it, get IPC handle, write to file.
        cudaEvent_t event;
        cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

        // Record the event on stream 0
        cudaCheck(cudaEventRecord(event, 0));

        // Wait for the event to complete before obtaining the handle.
        // In a real scenario we might leave the event incomplete to
        // demonstrate waiting, but for simplicity we finish it here.
        cudaCheck(cudaEventSynchronize(event));

        // Obtain IPC handle
        cudaIpcEventHandle_t ipcHandle;
        cudaCheck(cudaIpcGetEventHandle(&ipcHandle, event));

        // Write the handle to a binary file
        std::ofstream ofs("event_handle.bin", std::ios::binary);
        if (!ofs) {
            std::cerr << "Failed to open file for writing IPC handle.\n";
            return EXIT_FAILURE;
        }
        ofs.write(reinterpret_cast<const char*>(&ipcHandle), sizeof(ipcHandle));
        ofs.close();

        std::cout << "Process A: IPC event handle written to event_handle.bin\n";

        // Cleanup
        cudaCheck(cudaEventDestroy(event));
    } else if (std::strcmp(mode, "B") == 0) {
        // Process B: read IPC handle, open event, wait on it.
        std::ifstream ifs("event_handle.bin", std::ios::binary);
        if (!ifs) {
            std::cerr << "Failed to open file containing IPC handle.\n";
            return EXIT_FAILURE;
        }
        cudaIpcEventHandle_t ipcHandle;
        ifs.read(reinterpret_cast<char*>(&ipcHandle), sizeof(ipcHandle));
        ifs.close();

        // Open the IPC event
        cudaEvent_t event;
        cudaCheck(cudaIpcOpenEventHandle(&event, ipcHandle));

        std::cout << "Process B: IPC event handle opened. Waiting for event completion...\n";

        // Wait for the event to complete
        cudaCheck(cudaEventSynchronize(event));

        std::cout << "Process B: Event has completed.\n";

        // Cleanup
        cudaCheck(cudaEventDestroy(event));
    } else {
        std::cerr << "Unknown mode: " << mode << ". Use 'A' or 'B'.\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```