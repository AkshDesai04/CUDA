/*
Aim of the program:
Calculate and print the theoretical floating-point operations per second (FLOPS) for single-precision on your GPU. A rough estimate is `ClockRate_GHz * MultiProcessorCount * CoresPerSM * 2`. You'll need to look up the cores per SM for your GPU architecture.

Thinking:
1. Use CUDA Runtime API to query device properties with `cudaGetDeviceProperties`.
2. Extract:
   - `clockRate` (in kHz) → convert to GHz by dividing by 1e6.
   - `multiProcessorCount`.
   - `major` and `minor` compute capability to determine cores per Streaming Multiprocessor (SM).
3. Implement a helper function `int getCoresPerSM(int major, int minor)` that returns the number of CUDA cores per SM based on known architectures:
   - 3.x (Kepler): 192
   - 5.x (Maxwell): 128
   - 6.x (Pascal): 128
   - 7.x (Volta, Turing): 64 (Volta) or 64/128 for Turing? For simplicity use 64 for 7.x.
   - 8.x (Ampere): 64 (Ampere). For 8.x we return 64.
   - Default: 0 (unknown architecture).
4. Compute FLOPS: `clockGHz * multiProcessorCount * coresPerSM * 2`.
5. Print the result in GFLOPS (i.e., FLOPS / 1e9).
6. Handle error checking for CUDA calls.

Code below implements this logic in a single `.cu` file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper function to return CUDA cores per SM based on architecture
int getCoresPerSM(int major, int minor) {
    // Architecture to cores mapping
    // Source: NVIDIA docs / CUDA samples
    switch (major) {
        case 3: // Kepler
            return 192;
        case 5: // Maxwell
            return 128;
        case 6: // Pascal
            return 128;
        case 7: // Volta and Turing
            if (minor == 0) // Volta
                return 64;
            else if (minor == 5) // Turing
                return 64;
            else // unknown 7.x
                return 64;
        case 8: // Ampere
            // For Ampere A100 (GA100) 64 cores per SM
            // For other Ampere GPUs, also 64
            return 64;
        default:
            return 0; // Unknown architecture
    }
}

int main(void) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting CUDA device: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Clock rate in kHz, convert to GHz
    double clockGHz = prop.clockRate / 1e6; // clockRate is in kHz
    int mpCount = prop.multiProcessorCount;
    int coresPerSM = getCoresPerSM(prop.major, prop.minor);

    if (coresPerSM == 0) {
        fprintf(stderr, "Unknown architecture (%d.%d), cannot determine cores per SM.\n",
                prop.major, prop.minor);
        return -1;
    }

    double theoreticalFLOPS = clockGHz * mpCount * coresPerSM * 2.0; // single-precision

    printf("Device %d: %s\n", device, prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Clock Rate: %.3f GHz\n", clockGHz);
    printf("MultiProcessor Count: %d\n", mpCount);
    printf("Cores per SM: %d\n", coresPerSM);
    printf("Theoretical single-precision FLOPS: %.3f GFLOPS\n",
           theoreticalFLOPS / 1e9);

    return 0;
}
