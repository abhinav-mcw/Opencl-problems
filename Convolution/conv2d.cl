__kernel void conv2d(__global float* input, __global float* filter, __global float* output, int N, int M, int L) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < L && y < L) {
        float sum = 0;
        for(int j=0; j<M; j++){
            for(int i=0; i<M; i++){
                int inputIdx = (y+j)*N + (x+i);
                int filterIdx = j*M + i;
                sum = sum +  input[inputIdx]*filter[filterIdx];
            }
        }
        int outputIdx = y*L + x;
        output[outputIdx] = sum;
    }
}