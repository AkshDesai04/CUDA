/*
Use `nvcc --version` on the command line and compare its output to what `cudaRuntimeGetVersion` reports in your program.
This program demonstrates how to query the CUDA runtime API version inside a CUDA C/C++ program and how to execute an external shell command (`nvcc --version`) from within the same program. It then prints both results for comparison.

Thinking:
- The CUDA runtime API provides `cudaRuntimeGetVersion`, which returns an integer encoded as MAJOR * 1000 + MINOR (e.g., 11020 for CUDA 11.2). We'll call this function and display the integer.
- To capture the output of `nvcc --version`, we use the POSIX `popen` function to execute the shell command and read its stdout stream. We read the stream into a buffer, concatenate lines, and finally close the pipe.
- We handle errors: if `popen` fails, we print an error and exit. If `nvcc` is not installed or not found, the command will likely fail, and the output will be empty or contain an error message.
- The program outputs the runtime version integer and the raw output from `nvcc --version`. The user can manually compare them to see if they match (i.e., if the installed nvcc matches the runtime library).
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>

int main(void)
{
    /* Query CUDA runtime version */
    int runtime_version = 0;
    cudaError_t err = cudaRuntimeGetVersion(&runtime_version);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaRuntimeGetVersion failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    printf("CUDA Runtime Version: %d\n", runtime_version);

    /* Execute nvcc --version and capture output */
    FILE *fp = popen("nvcc --version", "r");
    if (fp == NULL) {
        perror("popen");
        return EXIT_FAILURE;
    }

    char line[256];
    char nvcc_output[2048] = {0};
    while (fgets(line, sizeof(line), fp) != NULL) {
        strncat(nvcc_output, line, sizeof(nvcc_output) - strlen(nvcc_output) - 1);
    }

    int ret = pclose(fp);
    if (ret == -1) {
        perror("pclose");
    }

    printf("\nnvcc --version output:\n%s\n", nvcc_output);

    return EXIT_SUCCESS;
}
