#include <vector>
#include <cstdlib>
#include <random>



void init4D_vector(int N, std::vector<float> &A, std::vector<float> &B){
    // float min = -30.0f, max = 100.0f;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                for (int l = 0; l < N; ++l) {
                    // Calculate the index in row-major format
                    int index = i * N * N * N + j * N * N + k * N + l;
                    
                    // Assign random value to h_a and 0.0f to h_b
                    A[index] = 10.0f; 
                    B[index] = 0.0f;
                }
            }
        }
    }
}