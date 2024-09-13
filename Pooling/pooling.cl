__kernel void max_pooling(__global float* input, __global float* output, int N, int M, int f, int s) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < N && y < N && x%s==0 && y%s==0) {
        float max = -9999.0f;
        for(int j=0; j<f; j++){
            for(int i=0; i<f; i++){
                int inputIdx = (y+j)*N + (x+i);
                if(input[inputIdx] > max)
                    max = input[inputIdx];
            }
        }
        int outputY, outputX;
        outputY = y/s;
        outputX = x/s;

        int outputIdx = outputY*M + outputX;
        output[outputIdx] = max;
    }
}

__kernel void avg_pooling(__global float* input, __global float* output, int N, int M, int f, int s) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < N && y < N && x%s==0 && y%s==0) {
        float sum = 0.0f;
        for(int j=0; j<f; j++){
            for(int i=0; i<f; i++){
                int inputIdx = (y+j)*N + (x+i);
                    sum = sum + input[inputIdx];
            }
        }
        int outputY, outputX;
        outputY = y/s;
        outputX = x/s;

        int outputIdx = outputY*M + outputX;
        output[outputIdx] = sum/(f*f);
    }
}