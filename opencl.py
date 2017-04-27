import pyopencl as cl
from pyopencl import array
import numpy

def clDotVector():
	return cl.Program(context, """
		__kernel void matrix_dot_vector(__global const float4 *matrix,
		__global const float4 *vector, __global float *result)
		{
			int gid = get_global_id(0);
			result[gid] = dot(matrix[gid], vector[0]);
		}
		""").build();
def clGrayScale():
	return cl.Program(context, """
		__kernel void imagingTest(__read_only  image2d_t srcImg,
							__write_only image2d_t dstImg)
		{
		const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
			CLK_ADDRESS_CLAMP_TO_EDGE | //Clamp to zeros
			CLK_FILTER_LINEAR;
		int2 coord = (int2)(get_global_id(0), get_global_id(1));
		uint4 bgra = read_imageui(srcImg, smp, coord); //The byte order is BGRA
		float4 bgrafloat = convert_float4(bgra) / 255.0f; //Convert to normalized [0..1] float
		//Convert RGB to luminance (make the image grayscale).
		float luminance =  sqrt(0.241f * bgrafloat.z * bgrafloat.z + 0.691f * 
							bgrafloat.y * bgrafloat.y + 0.068f * bgrafloat.x * bgrafloat.x);
		bgra.x = bgra.y = bgra.z = (uint) (luminance * 255.0f);
		bgra.w = 255;
		write_imageui(dstImg, coord, bgra);
		}	
		""").build();
		
		
		
if __name__ == "__main__":
	vector = numpy.zeros((1, 1), cl.array.vec.float4)
	matrix = numpy.zeros((1, 4), cl.array.vec.float4)
	matrix[0, 0] = (1, 2, 4, 8)
	matrix[0, 1] = (16, 32, 64, 128)
	matrix[0, 2] = (3, 6, 9, 12)
	matrix[0, 3] = (5, 10, 15, 25)
	
	vector[0, 0] = (1, 2, 4, 8)
	
	## Step #1. Obtain an OpenCL platform.
	platform = cl.get_platforms()[0]
	
	## It would be necessary to add some code to check th
	## necessary platform extensions with platform.extensions
	
	## Step #2. Obtain a device id for at least one device (accelerator).
	device = platform.get_devices()[0]
	
	## It would be necessary to add some code to check the support for
	## the necessary device extensions with device.extensions
	
	## Step #3. Create a context for the selected device.
	context = cl.Context([device])
	
	## Step #4. Create the accelerator program from source code.
	## Step #5. Build the program.
	## Step #6. Create one or more kernels from the program functions.
	program = clDotVector()
		
	## Step #7. Create a command queue for the target device.
	queue = cl.CommandQueue(context)
	
	## Step #8. Allocate device memory and move input data from the host to the device memory.
	mem_flags = cl.mem_flags
	matrix_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=matrix)
	vector_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=vector)
	matrix_dot_vector = numpy.zeros(4, numpy.float32)
	destination_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, matrix_dot_vector.nbytes)
	
	## Step #9. Associate the arguments to the kernel with the kernel object.
	## Step #10. Deploy the kernel for device execution.
	program.matrix_dot_vector(queue, matrix_dot_vector.shape, None, matrix_buf, vector_buf, destination_buf)
	
	## Step #11. Move the kernel's output data to host memory.
	cl.enqueue_copy(queue, matrix_dot_vector, destination_buf)
	
	## Step #12. Release context, program, kernels and memory.
	## PyOpenCL performs this step for you, and therefore,
	## you don't need to worry about cleanup code
	
	print(matrix_dot_vector)
	
	
	