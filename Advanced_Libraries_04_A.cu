/*
Aim of the program: Use `thrust::sort` to sort a single `device_vector`.

Thinking Process:
1. The requirement is to demonstrate the usage of `thrust::sort` on a single `device_vector`.
2. I need to create a CUDA source file (.cu) that compiles with nvcc.
3. Include the necessary Thrust headers: `<thrust/device_vector.h>` for the vector and `<thrust/sort.h>` for sorting.
4. Also include `<iostream>` for console output and `<vector>` and `<cstdlib>` for host-side data generation.
5. Generate a small set of random integers on the host, copy them into a `thrust::device_vector<int>`.
6. Print the unsorted values for demonstration.
7. Call `thrust::sort(dev_vec.begin(), dev_vec.end());` which sorts the vector on the GPU.
8. Copy the sorted data back to the host (via `dev_vec` directly or `thrust::host_vector`) and print the sorted result.
9. Keep the program simple and self‑contained. No external dependencies beyond Thrust and standard C++.
10. The output of the program will show the unsorted and sorted arrays, confirming that the sort worked correctly.

This design satisfies the brief: a minimal CUDA program that uses `thrust::sort` to sort a single `device_vector`. No additional commentary or support files are included; only the .cu source is output.
*/

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

int main()
{
    // Seed random number generator
    std::srand(static_cast<unsigned>(std::time(0)));

    const int N = 20; // Size of the vector
    std::vector<int> h_vec(N);

    // Populate host vector with random integers
    for (int i = 0; i < N; ++i)
        h_vec[i] = std::rand() % 100; // Random numbers between 0 and 99

    // Print unsorted host vector
    std::cout << "Unsorted host vector:" << std::endl;
    for (int i = 0; i < N; ++i)
        std::cout << h_vec[i] << " ";
    std::cout << std::endl << std::endl;

    // Transfer data to device
    thrust::device_vector<int> d_vec = h_vec;

    // Sort the device_vector using thrust::sort
    thrust::sort(d_vec.begin(), d_vec.end());

    // Copy sorted data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

    // Print sorted host vector
    std::cout << "Sorted host vector:" << std::endl;
    for (int i = 0; i < N; ++i)
        std::cout << h_vec[i] << " ";
    std::cout << std::endl;

    return 0;
}
