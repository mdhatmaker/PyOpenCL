import pyopencl as cl
from pyopencl import array
import numpy
		
if __name__ == "__main__":
	
	## Step #0. Prepare the data we want to work on (using the NumPy module)
	
	# initialize two numpy arrays with values from 0 to 9
	a = numpy.array(range(10), dtype=numpy.float32)
	b = numpy.array(range(10), dtype=numpy.float32)
	
	
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
	cl_code_file = "AddArrays.cl"
	
	f = open(cl_code_file, 'r')
	fstr = "".join(f.readlines())
	program = cl.Program(context, fstr).build()
	
	## Step #7. Create a command queue for the target device.
	queue = cl.CommandQueue(context)
	
	## Step #8. Allocate device memory and move input data from the host to the device memory.
	
	# create two OpenCL buffers where we pass in data to be copied to the device right away
	# we also create a "destination" buffer which we will use to store the results of our computation
	mf = cl.mem_flags
	a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
	b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
	dest_buf = cl.Buffer(context, mf.WRITE_ONLY, b.nbytes)
	
	# these buffers are now ready to be used by the kernel
	
	## Step #9. Associate the arguments to the kernel with the kernel object.
	## Step #10. Deploy the kernel for device execution.
	
	# A method has been added to our program instance with the name of our kernel ("addArrays").
	# So now we call it just like any other function, passing in our command queue, the global
	# and local worksizes (in this case our global size is the size of our arrays, and we don't
	# specify a local worksize, leaving it up to the implementation).
	# We then pass in the three parameters to our kernel - the three OpenCL buffers we created.
	program.addArrays(queue, a.shape, None, a_buf, b_buf, dest_buf)
	
	## Step #11. Move the kernel's output data to host memory.
	#cl.enqueue_copy(queue, matrix_dot_vector, destination_buf)
	
	# To look at the results of our computation, we need to read back the data from the dest_buf
	# and print it out:
	
	# read data from the destination buffer into our c array which is an empty numpy array of the
	# correct size and type
	c = numpy.empty_like(a)
	
	# the wait() on the end of the enqueue_read_buffer ensures we are done copying the data before
	# we try to print it out
	cl.enqueue_read_buffer(queue, dest_buf, c).wait()
	# (alternatively, we could have put queue.finish() on this subsequent line - same result)
	
	## Step #12. Release context, program, kernels and memory.
	## PyOpenCL performs this step for you, and therefore,
	## you don't need to worry about cleanup code
	
	print "a", a
	print "b", b
	print "c", c
	
	
	
	
	