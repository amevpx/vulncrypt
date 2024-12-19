#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Label: bad
void unchecked_user_input() {
    char buffer[10];
    gets(buffer); // Unsafe: buffer overflow possible
    printf("User input: %s\n", buffer);
}

// Label: good
void checked_user_input() {
    char buffer[10];
    if (fgets(buffer, sizeof(buffer), stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
        printf("User input: %s\n", buffer);
    }
}

// Label: bad
void stack_buffer_overflow() {
    char buffer[5];
    strcpy(buffer, "This is too long"); // Unsafe: buffer overflow
}

// Label: good
void stack_buffer_safe() {
    char buffer[5];
    strncpy(buffer, "safe", sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0'; // Ensure null-termination
}

// Label: bad
void unchecked_malloc() {
    char *data = malloc(10);
    strcpy(data, "hello"); // Unsafe: no malloc check
    printf("%s\n", data);
    free(data);
}

// Label: good
void checked_malloc() {
    char *data = malloc(10);
    if (data) {
        strncpy(data, "hello", 9);
        data[9] = '\0';
        printf("%s\n", data);
        free(data);
    }
}

// Label: bad
void memory_leak() {
    char *data = malloc(50);
    printf("Allocated memory: %p\n", data);
    // Forgot to free memory
}

// Label: good
void memory_properly_freed() {
    char *data = malloc(50);
    if (data) {
        printf("Allocated memory: %p\n", data);
        free(data);
    }
}

// Label: bad
void uncontrolled_format_string() {
    char input[50];
    gets(input); // Unsafe: allows malicious input
    printf(input); // Format string vulnerability
}

// Label: good
void controlled_format_string() {
    char input[50];
    fgets(input, sizeof(input), stdin);
    printf("%s", input); // Safe usage of input
}

// Label: bad
void integer_overflow() {
    unsigned int max = 0xFFFFFFFF;
    unsigned int result = max + 1; // Overflow
    printf("Result: %u\n", result);
}

// Label: good
void integer_overflow_check() {
    unsigned int a = 4000000000;
    unsigned int b = 500000000;
    if (a > UINT_MAX - b) {
        printf("Addition would overflow!\n");
    } else {
        printf("Result: %u\n", a + b);
    }
}

// Label: bad
void dangling_pointer() {
    char *data = malloc(50);
    free(data);
    printf("Dangling pointer: %p\n", data); // Unsafe: accessing freed memory
}

// Label: good
void null_pointer_check() {
    char *data = malloc(50);
    free(data);
    data = NULL; // Safe practice
    if (data == NULL) {
        printf("Pointer is NULL\n");
    }
}

// Label: bad
void double_free() {
    char *data = malloc(50);
    free(data);
    free(data); // Unsafe: double free
}

// Label: good
void avoid_double_free() {
    char *data = malloc(50);
    free(data);
    data = NULL; // Avoid double free
}

// Label: bad
void out_of_bounds_read() {
    int arr[5] = {1, 2, 3, 4, 5};
    printf("Out of bounds: %d\n", arr[10]); // Unsafe: out-of-bounds read
}

// Label: good
void safe_array_access() {
    int arr[5] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5; i++) {
        printf("Array element: %d\n", arr[i]);
    }
}

// Label: bad
void unvalidated_user_input() {
    char input[50];
    gets(input); // Unsafe: input not validated
    printf("Command: %s\n", input);
    system(input); // Unsafe: potential command injection
}

// Label: good
void validate_user_input() {
    char input[50];
    fgets(input, sizeof(input), stdin);
    printf("Safe command: %s\n", input); // Safe: no command execution
}
