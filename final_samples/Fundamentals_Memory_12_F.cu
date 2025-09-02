```cpp
/*
Aim of the program: Instantiate your class in a scope (e.g., inside a function or an `if` block) and verify that the destructor is called automatically when the object goes out of scope.

Thought process:
1. The program needs to demonstrate RAII (Resource Acquisition Is Initialization) semantics in C++ by showing that when an object goes out of scope, its destructor is invoked automatically.
2. Since the requirement is for a CUDA `.cu` file, the code will be compiled with `nvcc`. However, we do not need to use any CUDA-specific features for this demonstration; a simple host class with console output suffices.
3. To satisfy the requirement of instantiating the class within a scope, we will:
   - Create an instance inside an `if` block.
   - Create another instance inside a separate function.
4. The constructor will print a message indicating creation, and the destructor will print a message indicating destruction.
5. By running the compiled program, we should see the order of construction and destruction messages in the console, confirming that the destructor runs automatically when the objects leave scope.
6. No additional files or external resources are needed; the entire logic resides in this single `.cu` file.

The final code below implements this plan. It can be compiled with `nvcc` and executed to observe the constructor/destructor output.
*/

#include <iostream>

class TestClass {
public:
    TestClass() {
        std::cout << "TestClass constructed.\n";
    }

    ~TestClass() {
        std::cout << "TestClass destructed.\n";
    }
};

void demoFunction() {
    // Object instantiated inside a function scope
    TestClass funcObj;
    // Destructor will be called when funcObj goes out of scope (at the end of this function)
}

int main() {
    // Object instantiated inside an if block
    if (true) {
        TestClass ifObj;
        // Destructor will be called when ifObj goes out of scope (at the end of this block)
    }

    // Call the function that creates its own object
    demoFunction();

    return 0;
}
```