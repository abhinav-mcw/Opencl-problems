__kernel void horizontal_mirror_image(__global const uchar *input, __global uchar *output, int width, int height) {
    int gidX = get_global_id(0);
    int gidY = get_global_id(1);

    if (gidX < width && gidY < height) {
        int newWidth = 2 * width;
        output[gidY * newWidth + gidX] = input[gidY * width + gidX];
        output[gidY * newWidth + newWidth - gidX] = input[gidY * width + gidX];
    }
}
