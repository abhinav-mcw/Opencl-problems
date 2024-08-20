__kernel void relu6(const int N, __global float* A, __global float* B) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    if (i < N && j < N && k < N) {
        for (int l = 0; l < N; l++)
            B[i*N*N*N + j*N*N + k*N + l] = min(max(A[i*N*N*N + j*N*N + k*N + l], 0.0f), 6.0f);
    }
}
