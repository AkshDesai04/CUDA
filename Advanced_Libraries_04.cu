```cpp
/*
Aim of the program:
Use Thrust to sort a large array of key-value pairs. Create one device_vector for keys and another for values. Use thrust::sort_by_key to sort both vectors based on the key values.

Thinking:
- Include necessary Thrust headers for device_vector, sorting, and host_vector for I/O.
- Define a reasonable size N for the example (e.g., 20). In a real large-scale scenario, this could be millions.
- Generate random keys and associated values (e.g., values can be simple integers or strings; here we use integers for simplicity).
- Allocate device_vector for keys and values, copy generated data to them.
- Use thrust::sort_by_key(keys.begin(), keys.end(), values.begin()) to sort keys and reorder values accordingly.
- After sorting, copy the sorted vectors back to host_vector for printing.
- Print the sorted keys and their corresponding values to verify correctness.
- Use a simple deterministic random seed for reproducibility.
- Wrap the main logic in a function and call it from main().
- Provide minimal error handling (e.g., check CUDA errors after kernel launch if desired).
*/

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

int main()
{
    // Size of the key-value array
    const int N = 20;

    // Host vectors to hold keys and values initially
    thrust::host_vector<int> h_keys(N);
    thrust::host_vector<int> h_values(N);

    // Seed random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Populate host vectors with random keys and values
    for (int i = 0; i < N; ++i) {
        h_keys[i] = std::rand() % 100;    // Random key between 0 and 99
        h_values[i] = i;                 // Example value (could be anything)
    }

    std::cout << "Before sorting:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "Key: " << h_keys[i] << " Value: " << h_values[i] << std::endl;
    }

    // Transfer data to device
    thrust::device_vector<int> d_keys = h_keys;
    thrust::device_vector<int> d_values = h_values;

    // Sort by key; values are reordered accordingly
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

    // Copy sorted data back to host
    thrust::copy(d_keys.begin(), d_keys.end(), h_keys.begin());
    thrust::copy(d_values.begin(), d_values.end(), h_values.begin());

    std::cout << "\nAfter sorting:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "Key: " << h_keys[i] << " Value: " << h_values[i] << std::endl;
    }

    return 0;
}
```