/**
 * 04_function_pointers.c - Callbacks and Dynamic Dispatch
 * 
 * Function pointers enable:
 * - Callbacks (like PyTorch hooks)
 * - Plugin systems
 * - Dispatch tables (like PyTorch's dispatcher)
 * 
 * Compile: gcc -o func_ptr 04_function_pointers.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================
// SECTION 1: Basic Function Pointers
// ============================================================

// Two functions with same signature
float add(float a, float b) { return a + b; }
float mul(float a, float b) { return a * b; }

void demonstrate_basics() {
    printf("=== BASIC FUNCTION POINTERS ===\n\n");
    
    // Declare function pointer
    float (*op)(float, float);
    
    // Point to add
    op = add;
    printf("add(3, 4) via pointer: %.1f\n", op(3, 4));
    
    // Point to mul
    op = mul;
    printf("mul(3, 4) via pointer: %.1f\n", op(3, 4));
    printf("\n");
}

// ============================================================
// SECTION 2: Callbacks (like PyTorch hooks)
// ============================================================

typedef void (*HookFn)(float* data, int n);

void apply_hook(float* data, int n, HookFn hook) {
    if (hook != NULL) {
        hook(data, n);
    }
}

void print_hook(float* data, int n) {
    printf("  Hook called: data[0] = %.2f\n", data[0]);
}

void scale_hook(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] *= 2;
    }
    printf("  Scaled data by 2\n");
}

void demonstrate_callbacks() {
    printf("=== CALLBACKS (LIKE PYTORCH HOOKS) ===\n\n");
    
    float data[5] = {1, 2, 3, 4, 5};
    
    printf("Before:\n");
    apply_hook(data, 5, print_hook);
    
    printf("Apply scale hook:\n");
    apply_hook(data, 5, scale_hook);
    
    printf("After:\n");
    apply_hook(data, 5, print_hook);
    
    printf("\nPyTorch equivalent:\n");
    printf("  model.register_forward_hook(my_hook)\n\n");
}

// ============================================================
// SECTION 3: Dispatch Table (like PyTorch dispatcher)
// ============================================================

typedef float (*ActivationFn)(float);

float relu(float x) { return x > 0 ? x : 0; }
float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
float tanh_fn(float x) { return tanhf(x); }

// Dispatch table
ActivationFn activation_table[] = {
    relu,
    sigmoid,
    tanh_fn
};

const char* activation_names[] = {"relu", "sigmoid", "tanh"};

void apply_activation(float* data, int n, int activation_id) {
    ActivationFn fn = activation_table[activation_id];
    for (int i = 0; i < n; i++) {
        data[i] = fn(data[i]);
    }
}

void demonstrate_dispatch() {
    printf("=== DISPATCH TABLE ===\n\n");
    
    float data[5] = {-2, -1, 0, 1, 2};
    
    for (int act = 0; act < 3; act++) {
        float test[5] = {-2, -1, 0, 1, 2};
        apply_activation(test, 5, act);
        
        printf("%s: [", activation_names[act]);
        for (int i = 0; i < 5; i++) {
            printf("%.2f%s", test[i], i < 4 ? ", " : "");
        }
        printf("]\n");
    }
    
    printf("\nPyTorch dispatcher does this for:\n");
    printf("  CPU vs CUDA vs MPS backends\n");
    printf("  Different dtypes\n");
    printf("  Autograd vs no-grad\n");
}

int main() {
    demonstrate_basics();
    demonstrate_callbacks();
    demonstrate_dispatch();
    
    printf("\n=== ML APPLICATIONS ===\n\n");
    printf("1. PyTorch hooks: forward_hook, backward_hook\n");
    printf("2. Dispatcher: Route ops to correct backend\n");
    printf("3. Custom ops: Register C functions\n");
    printf("4. Optimizers: Different update rules\n");
    
    return 0;
}
