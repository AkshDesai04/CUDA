/*
Calculate and print the theoretical floating-point operations per second (FLOPS) for single-precision on your GPU. A rough estimate is `ClockRate_GHz * MultiProcessorCount * CoresPerSM * 2`. You'll need to look up the cores per SM for your GPU architecture.

Thinking:
- The goal is to query the GPU's properties via the CUDA Runtime API, then compute a rough theoretical peak single‑precision FLOPS.
- The key device properties are:
  * clockRate (in kHz) – convert to GHz.
  * multiProcessorCount – number of SMs.
  * major/minor compute capability – needed to map to CUDA cores per SM.
- CUDA cores per SM vary by architecture:
  * Kepler (2.x, 3.x) – 192
  * Maxwell (5.x) – 128
  * Pascal (6.x) – 64
  * Volta/Turing (7.x) – 64
  * Ampere (8.x) – 128
  * Lovelace (9.x) – 128 (approximate)
- The rough formula multiplies the clock rate in GHz, the number of SMs, the cores per SM, and 2 (because each core can perform 2 FP32 ops per cycle on most modern GPUs).
- We'll encapsulate the mapping in a helper function that returns cores per SM given the compute capability.
- The program prints the GPU name, compute capability, number of SMs, clock rate, cores per SM, and the theoretical peak GFLOPS.
- Error handling is minimal – if device properties cannot be queried, the program exits with an error message.
- The output is a single floating‑point number (GFLOPS) printed to stdout.

*/

#include <iostream>
#include <cuda_runtime.h>

// Helper function to map compute capability to cores per SM
int getCoresPerSM(int major, int minor) {
    // The mapping below is approximate and covers common NVIDIA architectures.
    // For future or less common architectures, the returned value may not be accurate.
    switch (major) {
        case 2: // Kepler
            return 192;
        case 3: // Kepler
            return 192;
        case 5: // Maxwell
            return 128;
        case 6: // Pascal
            return 64;
        case 7: // Volta/Turing
            return 64;
        case 8: // Ampere
            return 128;
        case 9: // Lovelace
            return 128;
        default:
            // Default fallback
            return 64;
    }
}

int main() {
    int device = 0; // We'll use the first GPU
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Clock rate in kHz -> convert to GHz
    double clockRateGHz = prop.clockRate / 1e6;

    int mpCount = prop.multiProcessorCount;
    int coresPerSM = getCoresPerSM(prop.major, prop.minor);

    // Rough theoretical single-precision GFLOPS
    // 2 is the factor for FP32 operations per cycle per core on most GPUs
    double theoreticalGFLOPS = clockRateGHz * mpCount * coresPerSM * 2.0;

    // Print information
    std::cout << "GPU Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Number of SMs: " << mpCount << std::endl;
    std::cout << "Clock Rate: " << clockRateGHz << " GHz" << std::endl;
    std::cout << "Cores per SM: " << coresPerSM << std::endl;
    std::cout << "Theoretical Single-Precision GFLOPS: " << theoreticalGFLOPS << " GFLOPS" << std::endl;

    return 0;
}
