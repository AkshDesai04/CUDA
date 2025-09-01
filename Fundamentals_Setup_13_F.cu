```cuda
/*
Aim: Calculate and print the theoretical floating-point operations per second (FLOPS)
for single-precision on your GPU. A rough estimate is 
`ClockRate_GHz * MultiProcessorCount * CoresPerSM * 2`. You'll need to look up 
the cores per SM for your GPU architecture.

Thinking:
1. Retrieve device properties using `cudaGetDeviceProperties`.
2. Extract `clockRate` (in kHz), `multiProcessorCount`, and compute capability (major/minor).
3. Convert `clockRate` to GHz: `clockRate_GHz = clockRate / 1e6` (since clockRate is in kHz).
4. Determine `CoresPerSM` based on architecture:
   - 2.x: 32
   - 3.x: 192
   - 5.x: 128
   - 6.x: 128
   - 7.x: 64
   - 8.x: 64 (but 8.6 and above 128)
   Use a helper function to map major/minor to cores per SM.
5. Compute FLOPS: `FLOPS = ClockRate_GHz * MultiProcessorCount * CoresPerSM * 2`.
   This yields operations per second in FLOPS (not gigaflops).
6. Print the result as a floating-point number (double) and also in GigaFLOPS.
7. Handle error checking for CUDA calls.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper function to get cores per SM based on compute capability
int getCoresPerSM(int major, int minor) {
    switch (major) {
        case 2: // Fermi
            return 32;
        case 3: // Kepler
            return 192;
        case 5: // Maxwell
            return 128;
        case 6: // Pascal
            return 128;
        case 7: // Volta/Turing
            return 64;
        case 8: // Ampere
            if (minor == 0 || minor == 6) // Ampere SMX and SMX with 128 cores
                return (minor == 6) ? 128 : 64;
            else
                return 64;
        default:
            // Default fallback
            return 64;
    }
}

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return 1;
    }

    // For simplicity, use device 0
    int device = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    double clockRateGHz = prop.clockRate / 1e6; // prop.clockRate is in kHz
    int smCount = prop.multiProcessorCount;
    int coresPerSM = getCoresPerSM(prop.major, prop.minor);

    // Calculate theoretical FLOPS
    double flops = clockRateGHz * smCount * coresPerSM * 2.0;

    printf("Device %d: %s\n", device, prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Clock Rate (GHz): %.3f\n", clockRateGHz);
    printf("SM Count: %d\n", smCount);
    printf("Cores per SM: %d\n", coresPerSM);
    printf("Theoretical single-precision FLOPS: %.2f\n", flops);
    printf("Theoretical single-precision FLOPS (GigaFLOPS): %.2f GFLOPS\n", flops / 1e9);

    return 0;
}
```