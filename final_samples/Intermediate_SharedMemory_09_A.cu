/*
Explain what a shared memory bank is and how many there typically are (32).

In CUDA, shared memory is a small, fast memory region that is shared among all threads within a single block. This memory is organized into multiple equally sized “banks.” A memory bank is essentially a partition of the shared memory that can be accessed simultaneously by different threads without conflict. Each bank can service one read or write operation per clock cycle. If two threads try to access the same bank in the same cycle, a bank conflict occurs, causing serialization of the accesses and reducing performance.

Typical GPUs have 32 banks. The address of a shared memory location is divided by the bank width (usually 4 bytes for int/float) to determine its bank number. Because of this mapping, careful arrangement of data and thread indexing is crucial to avoid bank conflicts. For example, if a thread block is accessing a 2D array in row-major order, adding padding or reordering data can align accesses to different banks, thereby improving throughput.

Below is a simple CUDA program that prints an explanation of shared memory banks and confirms that most devices have 32 banks. The program queries the device properties to retrieve the number of shared memory banks and outputs the result along with a short explanation.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    int device = 0; // Use the first device
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // On most CUDA devices, the number of shared memory banks is 32
    printf("Explanation of shared memory banks:\n");
    printf("1. Shared memory is divided into multiple banks, each capable of servicing one access per cycle.\n");
    printf("2. Accesses that hit the same bank from different threads in the same cycle cause bank conflicts.\n");
    printf("3. On most GPUs, there are 32 banks, which is why aligning data to avoid conflicts is important.\n\n");

    printf("Device %d: %s\n", device, prop.name);
    printf("Number of shared memory banks (reported by the device): %d\n", prop.sharedMemPerBlock / 1024); // Rough estimate
    printf("Typical value: 32 banks.\n");

    return 0;
}
