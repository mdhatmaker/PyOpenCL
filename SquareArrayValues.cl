__kernel void squareArrayValues(__global int* a, __global int* b)
{
    unsigned int i = get_global_id(0);
    b[i] = a[i]*a[i];
}