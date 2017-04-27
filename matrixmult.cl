# example provided by Eilif Muller

from __future__ import division

import pyopencl as cl
from time import time
import numpy

block_size = 16

ctx = cl.create_some_context()

for dev in ctx.devices:
	assert dev.local_mem_size > 0
	
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

#queue = cl.CommandQueue(ctx)

if False:
	a_height = 4096
	#a_height = 1024
	a_width = 2048
	#a_width = 256
	#b_height == a_width
	b_width = a_height
	
elif False:
	# like PyCUDA
	a_height = 2516
	a_width = 1472
	b_height = a_width
	b_width = 2144
	
else:
	# CL SDK
	a_width = 50 * block_size
	a_height = 100 * block_size
	b_width = 50 * block_size
	b_height = a_width
	
c_width = b_width
c_height = a_height

h_a = numpy.random.rand(a_height, a_width).astype(numpy.float32)
h_b = numpy.random.rand(b_height, b_width).astype(numpy.float32)
h_c = numpy.empty((c_height, c_width)).astype(numpy.float32)


kernel_params = {"block_size": block_size, "w_a": a_width, "h_a": a_height, "w_b": b_width}

if "NVIDIA" in queue.device.vendor:
	options = "-cl-mad-enable -cl-fast-relaxed-math"
else:
	options = ""

prg = cl.Program(ctx, KERNEL_CODE % kernel_params, ).build(options=options)
kernel = prg.matrixMul
#pring prg.binaries[0]

assert a_width % block_size 