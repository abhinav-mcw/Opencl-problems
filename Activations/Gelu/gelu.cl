__kernel void gelu(const int N, __global float* A, __global float* B) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    if (i < N && j < N && k < N) {
        for (int l = 0; l < N; l++) {
            int index = i*N*N*N + j*N*N + k*N + l;
            B[index] = .5f * A[index] * (1 + tanh(sqrt(2/3.14f) * (A[index] * 0.044715 * pow(A[index],3))));
        }
    }
}
