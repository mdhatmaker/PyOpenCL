__kernel void matrix_dot_vectort(__global const float4 *matrix, __global const float4 *vector, __global float *result)
{
	int gid = get_global_id(0);
	result[gid] = dot(matrix[gid], vector[0]);
}
