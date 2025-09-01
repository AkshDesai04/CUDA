/*
Write a loop that iterates through the arrays and prints an error message if a mismatch is found, then breaks.

Thinking:
- The requirement is straightforward: we need a loop that goes through two arrays and reports a mismatch.
- The prompt does not specify that the loop must be on the device side, so a host‑side loop suffices.
- We'll create two integer arrays of a fixed size, intentionally introduce a mismatch, then iterate over them.
- As soon as a mismatch is found, we print an error message and break out of the loop.
- This example is self‑contained in a single .cu file so it can be compiled with nvcc.
- No CUDA kernel or device memory is needed for this simple demonstration, but the file extension remains .cu to follow the instruction.
- The code includes standard headers and uses `printf` for output.
- We keep the loop simple and illustrate the break after the first mismatch.
*/

#include <stdio.h>

int main(void) {
    const int N = 10;
    int a[N];
    int b[N];

    // Initialize arrays
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i;
    }

    // Introduce a mismatch at a known position
    b[5] = -1;

    // Loop through arrays, print error on mismatch and break
    for (int i = 0; i < N; ++i) {
        if (a[i] != b[i]) {
            printf("Error: mismatch found at index %d: a[%d] = %d, b[%d] = %d\n",
                   i, i, a[i], i, b[i]);
            break;
        }
    }

    return 0;
}
