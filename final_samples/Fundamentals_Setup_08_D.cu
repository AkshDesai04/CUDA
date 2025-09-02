/*
Aim: Explain in a comment where the CUDA driver is typically installed (e.g., as part of the NVIDIA display driver) and where the CUDA runtime is installed (e.g., with the CUDA Toolkit).

Thoughts:
- The CUDA driver is a kernel-mode driver that interacts directly with the GPU hardware. It is usually bundled with the NVIDIA display driver, so installing or updating the graphics driver automatically installs or updates the CUDA driver.
  * On Linux, the driver binaries are typically found in directories such as /usr/lib/nvidia-<version>/, /usr/lib64/nvidia-<version>/, or under /usr/lib/x86_64-linux-gnu/ .
  * On Windows, the driver DLLs reside in system folders like C:\Windows\System32\DriverStore\FileRepository\nvidia_* or under the NVIDIA GPU driver installation directory (e.g., C:\Program Files\NVIDIA Corporation\NVDisplayDriver\<version>\).
- The CUDA runtime is part of the CUDA Toolkit and provides user‑level libraries and header files for developing CUDA applications. It is usually installed in a separate location from the driver.
  * On Linux, the Toolkit is commonly installed under /usr/local/cuda-<version>/ (often a symlink named /usr/local/cuda pointing to the active version).
  * On Windows, the Toolkit is typically installed under C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<version>\.

The following simple CUDA program demonstrates how to print this explanation. It uses the CUDA runtime (for illustration purposes) but the code itself is ordinary C++ and does not require GPU execution. The comments above explain the typical installation locations for the driver and runtime.

This code is intended to compile with a standard CUDA compiler (nvcc) and will print the explanation to standard output.
*/

#include <iostream>

int main() {
    std::cout << "CUDA Driver:\n"
                 "  Typically installed as part of the NVIDIA display driver.\n"
                 "  - On Linux: e.g., /usr/lib/nvidia-<version>/ or /usr/lib64/nvidia-<version>/\n"
                 "  - On Windows: e.g., C:\\Windows\\System32\\DriverStore\\FileRepository\\nvidia_* or within the NVIDIA GPU driver installation directory.\n\n"
                 "CUDA Runtime:\n"
                 "  Installed with the CUDA Toolkit.\n"
                 "  - On Linux: e.g., /usr/local/cuda-<version>/ (often symlinked as /usr/local/cuda)\n"
                 "  - On Windows: e.g., C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v<version>\\\n"
                 << std::endl;
    return 0;
}
