/*
Aim of the program: Use `nvcc --version` on the command line and compare its output to what `cudaRuntimeGetVersion` reports in your program.

Thinking:  
The task is to query the compiler version by invoking the external command `nvcc --version` and then query the runtime API version via `cudaRuntimeGetVersion`.  
- To capture the output of `nvcc --version`, `popen` is used to execute the command and read its stdout.  
- The output of `nvcc --version` contains a line such as “Cuda compilation tools, release 11.2, V11.2.152”.  We locate the substring “release ” and then extract the numeric part (e.g., “11.2”).  
- The extracted string is parsed into major and minor integers and combined into a single integer representation similar to the one returned by `cudaRuntimeGetVersion` (which typically encodes major*1000 + minor*10).  
- We then call `cudaRuntimeGetVersion` to obtain the runtime API version number.  
- Finally, we print both versions and indicate whether they match.  
The program is simple, host‑only, and should compile with `nvcc`. It uses only standard C/C++ headers and the CUDA runtime header.  
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

int main()
{
    // Execute nvcc --version and capture its output
    FILE* fp = popen("nvcc --version", "r");
    if (!fp) {
        perror("popen failed");
        return EXIT_FAILURE;
    }

    char line[512];
    char version_str[32] = {0};
    bool found = false;

    // Read the output line by line looking for the "release " keyword
    while (fgets(line, sizeof(line), fp)) {
        const char* pos = strstr(line, "release ");
        if (pos) {
            pos += strlen("release ");
            int i = 0;
            while (*pos && (*pos == '.' || (*pos >= '0' && *pos <= '9')) && i < (int)sizeof(version_str) - 1) {
                version_str[i++] = *pos++;
            }
            version_str[i] = '\0';
            found = true;
            break;
        }
    }

    pclose(fp);

    if (!found) {
        fprintf(stderr, "Could not parse nvcc version from output.\n");
        return EXIT_FAILURE;
    }

    // Convert the extracted version string into an integer comparable to cudaRuntimeGetVersion
    int nvcc_major = 0, nvcc_minor = 0;
    if (sscanf(version_str, "%d.%d", &nvcc_major, &nvcc_minor) < 1) {
        fprintf(stderr, "Failed to parse nvcc version string '%s'.\n", version_str);
        return EXIT_FAILURE;
    }
    // cudaRuntimeGetVersion returns major*1000 + minor*10 + patch (patch is usually 0)
    int nvcc_version_int = nvcc_major * 1000 + nvcc_minor * 10;

    // Retrieve runtime version
    int runtime_version = 0;
    cudaError_t err = cudaRuntimeGetVersion(&runtime_version);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaRuntimeGetVersion failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Print results
    printf("nvcc version string: %s\n", version_str);
    printf("nvcc version int (major*1000 + minor*10): %d\n", nvcc_version_int);
    printf("cudaRuntimeGetVersion: %d\n", runtime_version);

    if (nvcc_version_int == runtime_version) {
        printf("Versions match.\n");
    } else {
        printf("Versions differ.\n");
    }

    return EXIT_SUCCESS;
}
