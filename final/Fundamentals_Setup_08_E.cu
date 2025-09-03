```cuda
/*
 * Aim: Use `nvcc --version` on the command line and compare its output to what
 * `cudaRuntimeGetVersion` reports in your program.
 *
 * Thinking process:
 * 1. The program needs to run `nvcc --version` from within the CUDA C++ program
 *    and capture its output.  We'll use `popen` to execute the command and read
 *    the resulting text stream.
 * 2. We also need to query the CUDA runtime version using the API
 *    `cudaRuntimeGetVersion`.  The function returns an integer like 11020
 *    which represents version 11.2 (major * 1000 + minor * 10).  We'll convert
 *    that to a human‑readable major.minor format.
 * 3. The program will print:
 *    - The raw output of `nvcc --version` (so the user can see the compiler
 *      version string).
 *    - The integer returned by `cudaRuntimeGetVersion`.
 *    - The parsed major/minor runtime version.
 * 4. Error handling: If `popen` fails, we will print an error message and
 *    continue.  If `cudaRuntimeGetVersion` fails, we will report the error
 *    string using `cudaGetErrorString`.
 * 5. The code is written in a single .cu file and is fully self‑contained.
 *    It can be compiled with `nvcc` and run on a machine that has the CUDA
 *    toolkit installed.  The program is trivial and does not launch any GPU
 *    kernels; it merely uses the runtime API.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// Helper function to get nvcc version string
std::string getNvccVersion()
{
    const char* cmd = "nvcc --version";
    FILE* pipe = popen(cmd, "r");
    if (!pipe)
    {
        return std::string("Error: unable to execute nvcc --version");
    }

    char buffer[128];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe))
    {
        result += buffer;
    }
    pclose(pipe);
    return result;
}

// Helper function to convert runtime version integer to major.minor string
std::string formatRuntimeVersion(int runtimeVersion)
{
    int major = runtimeVersion / 1000;
    int minor = (runtimeVersion % 1000) / 10;
    char buf[32];
    snprintf(buf, sizeof(buf), "%d.%d", major, minor);
    return std::string(buf);
}

int main()
{
    // 1. Get nvcc version output
    std::string nvccOutput = getNvccVersion();

    // 2. Query CUDA runtime version
    int runtimeVersion = 0;
    cudaError_t err = cudaRuntimeGetVersion(&runtimeVersion);
    std::string runtimeVersionStr;
    if (err == cudaSuccess)
    {
        runtimeVersionStr = formatRuntimeVersion(runtimeVersion);
    }
    else
    {
        runtimeVersionStr = std::string("Error: ") + cudaGetErrorString(err);
    }

    // 3. Print results
    printf("=== nvcc --version output ===\n%s\n", nvccOutput.c_str());
    printf("\n=== cudaRuntimeGetVersion ===\n");
    if (err == cudaSuccess)
    {
        printf("Integer value: %d\n", runtimeVersion);
        printf("Parsed version: %s\n", runtimeVersionStr.c_str());
    }
    else
    {
        printf("%s\n", runtimeVersionStr.c_str());
    }

    return 0;
}
```