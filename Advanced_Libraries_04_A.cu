/*
Use `thrust::sort` to sort a single `device_vector`.

Thinking process:
1. We need a CUDA C++ program that demonstrates sorting a single `thrust::device_vector` using `thrust::sort`.
2. Include necessary headers: `<thrust/device_vector.h>`, `<thrust/sort.h>`, `<thrust/host_vector.h>`, `<iostream>`, `<algorithm>`.
3. Create a `thrust::device_vector<int>` with some size, e.g., 20 elements.
4. Populate it with random integers (for reproducibility we can use a simple deterministic sequence).
5. Copy the unsorted vector to a `thrust::host_vector` (or `std::vector`) to print it.
6. Call `thrust::sort(dev_vec.begin(), dev_vec.end());`.
7. Copy the sorted vector back to a host container and print it to verify.
8. Keep the code minimal and self‑contained so it can be compiled with `nvcc`.
9. Add a small helper function to print a vector.
10. Ensure the program compiles as a `.cu` file.
*/

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <iomanip>

// Helper to print a vector (host-side)
template <typename T>
void print_vector(const std::vector<T>& vec, const std::string& name)
{
    std::cout << name << ": ";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        std::cout << std::setw(3) << vec[i] << " ";
    }
    std::cout << "\n";
}

int main()
{
    const int N = 20;
    // Create a device_vector with N elements
    thrust::device_vector<int> d_vec(N);

    // Fill with a simple deterministic pattern (e.g., N-i)
    for (int i = 0; i < N; ++i)
    {
        d_vec[i] = N - i;
    }

    // Copy to host for printing before sort
    thrust::host_vector<int> h_vec_before = d_vec; // implicit copy
    print_vector(h_vec_before, "Before sort");

    // Sort the device_vector in ascending order
    thrust::sort(d_vec.begin(), d_vec.end());

    // Copy sorted data back to host for printing
    thrust::host_vector<int> h_vec_after = d_vec;
    print_vector(h_vec_after, "After sort");

    return 0;
}
