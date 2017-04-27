import pyopencl as cl
import pyopencl.array as cl_array
import numpy
 
if __name__ == "__main__":
    platform = cl.get_platforms()[0]
     
    device = platform.get_devices()[0]
     
    context = cl.Context([device])
     
    queue = cl.CommandQueue(context)
     
    n = 40000
     
    x_origin_gpu = cl_array.to_device(queue, numpy.arange(0, n, 1, dtype=numpy.float32))
     
    # x_origin[i] + r * cos(2 * PI * i / n)
    from pyopencl.elementwise import ElementwiseKernel
    calculate_polygon_vertices_x = ElementwiseKernel(context,
            "float r, float *x_origin, float *x ",
            operation="x[i] = x_origin[i] + (r * cos(2 * M_PI * i / n))",
            name="calculate_polygon_vertices_x", preamble="#define M_PI 3.14159265358979323846")
     
    x_gpu = cl_array.empty_like(x_origin_gpu)
     
    # long n is included in the element-wise kernel
    # float r, float *x_origin, float *x
    event = calculate_polygon_vertices_x(50.0, x_origin_gpu, x_gpu)
     
    print(x_gpu)