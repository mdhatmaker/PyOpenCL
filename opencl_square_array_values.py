# This program generates a Numpy array with all integers between 0 and 9, passes it
# to an OpenCL kernel that computes the square of an array of integers, dynamically
# compiles and executes this kernel, and copies back the output from the context memory
# to Python.


import pyopencl as cl
import numpy as np

# create an OpenCL context
#ctx = cl.create_some_context()
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])

# create the command queue
queue = cl.CommandQueue(ctx)

# create the kernel input
a = np.array(np.arange(10), dtype=np.int32)

# kernel output placeholder
b = np.empty(a.shape, dtype=np.int32)

# create context buffers for a and b arrays
# for a (input), we need to specify that this buffer should be populated from a
mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)

# for b (output), we just allocate an empty buffer
b_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

# OpenCL kernel code
#code = """
#__kernel void squareArrayValues(__global int* a, __global int* b)
#{
#	int i = get_global_id(0);
#	b[i] = a[i] * a[i];
#}
#"""

# read the OpenCL "kernel" code from a file
cl_code_file = "SquareArrayValues.cl"	
f = open(cl_code_file, 'r')
fstr = "".join(f.readlines())

# compile the kernel
#prg = cl.Program(ctx, code).build()
prg = cl.Program(ctx, fstr).build()

# launch the kernel
event = prg.squareArrayValues(queue, a.shape, None, a_buf, b_buf)
event.wait()

# copy the output from the context to the Python process
cl.enqueue_copy(queue, b, b_buf)

# if everything went fine, b should contain the squares of integers
print(b)
 