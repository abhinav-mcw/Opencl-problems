__kernel void conv3d(__global float* input, __global float* filter, __global float* output, int N, int M, int L) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    if (x < L && y < L && z < L ) {
        float sum = 0;
        for(int k=0; k<M; k++){
            for(int j=0; j<M; j++){
                for(int i=0; i<M; i++){
                    int inputIdx = (z+k)*N*N + (y+j)*N + (x+i);
                    int filterIdx = k*M*M + j*M + i;
                    sum = sum +  input[inputIdx]*filter[filterIdx];
                }
            }
        }
        int outputIdx = z*L*L + y*L + x;
        output[outputIdx] = sum;
    }
}