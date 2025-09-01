/*
 * Aim: Create a main function that accepts N as a command-line argument.
 *
 * Thinking:
 * 1. The problem statement is very simple – we just need a `main` that reads an
 *    integer `N` from the command line.  We should perform minimal error
 *    checking: if the argument is missing, print a usage message and exit
 *    with a non‑zero status code.  The argument will be parsed with
 *    `std::stoi` to convert the string to an `int`.
 * 2. The program will output the value of `N` to standard output so that
 *    it is clear the value was received and parsed correctly.
 * 3. Although the task mentions CUDA, this particular exercise does not
 *    require any GPU code.  The file extension is `.cu` and we can still
 *    use ordinary C++ code.  If desired, a dummy kernel could be defined
 *    but it is unnecessary for the task.  We keep the code simple and
 *    portable.
 * 4. The code uses standard headers (`<iostream>`, `<cstdlib>`, `<string>`)
 *    and follows the usual `int main(int argc, char *argv[])` signature.
 * 5. No external libraries or CUDA-specific constructs are needed for
 *    this straightforward command‑line parsing task.
 */

#include <iostream>
#include <cstdlib>
#include <string>

int main(int argc, char* argv[]) {
    // Expect exactly one command‑line argument (besides the program name)
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " N\n";
        return EXIT_FAILURE;
    }

    // Convert the first argument to an integer
    int N;
    try {
        N = std::stoi(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid integer '" << argv[1] << "': " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    // Output the received value
    std::cout << "Received N = " << N << std::endl;

    return EXIT_SUCCESS;
}
