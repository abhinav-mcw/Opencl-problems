
__kernel void sigmoid(const int N, __global float* A, __global float* B) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    if (i < N && j < N && k < N) {
        for (int l = 0; l < N; l++) {
            int index = i*N*N*N + j*N*N + k*N + l;
            B[index] = 1 / (1 + exp(A[index] * -1));
        }
    }
}
