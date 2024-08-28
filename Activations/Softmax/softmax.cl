__kernel void softmax(const int N, __global float* A, __global float* B) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    if (i < N && j < N && k < N) {
        long sum=0.0l;
        int index = i*N*N*N + j*N*N + k*N;
        for (int l = 0; l < N; l++) {
            index = index + l;
            B[index] = exp(A[index]);
            sum+=B[index];
        }
        index = i*N*N*N + j*N*N + k*N;
        for (int l = 0; l < N; l++) {
            index = index + l;
            B[index]=B[index]/sum;
        }
    }
}
