from __future__ import absolute_import
from __future__ import print_function

import time

import numpy as np

import pyopencl as cl
from six.moves import range

# uncomment one of the three lines to test the three variations

# set width and height of window (more pixels take longer to calculate)
w = 2048
h = 2048

def calc_fractal_opencl(q, maxiter):
	ctx = cl.create_some_context()
	queue = cl.CommandQueue(ctx)
	
	output = np.empty(q.shape, dtype=np.uint16)
	
	mf = cl.mem_flags
	q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
	output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)
	
	prg = cl.Program(ctx, """
	#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
	__kernel void mandelbrot(__global float2 *q, __global ushort *output, ushort const maxiter)
	{
		int gid = get_global_id(0);
		float nreal, real = 0;
		float imag = 0;
		
		output[gid] = 0;
		
		for (int curiter = 0; curiter < maxiter; curiter++)
		{
			nreal = real * real - imag * imag + q[gid].x;
			imag = 2 * real * imag + q[gid].y;
			real = nreal;
			
			if (real * real + imag * imag > 4.0f)
				output[gid] = curiter;
		}
	}
	""").build()
	
	prg.mandelbrot(queue, output.shape, None, q_opencl, output_opencl, np.uint16(maxiter))
	
	cl.enqueue_copy(queue, output, output_opencl).wait()
	
	return output
		
def calc_fractal_serial(q, maxiter):
	# calculate x using pure python on a numpy array
	z = np.zeros(q.shape, complex)
	output = np.resize(np.array(0,), q.shape)
	for i in range(len(q)):
		for iter in range(maxiter):
			z[i] = z[i] * z[i] + q[i]
			if abs(z[i]) > 2.0:
				output[i] = iter
				break
	return output
	
def calc_fractal_numpy(q, maxiter):
	# calculate x using numpy
	output = np.resize(np.array(0,), q.shape)
	z = np.zeros(q.shape, p.complex64)
	
	for it in range(maxiter):
		z = z * z + q
		done = np.greater(abs(z), 2.0)
		q = np.where(done, 0+0j, q)
		z = np.where(done, 0+0j, z)
		output = np.where(done, it, output)
	return output
	
# choose your calculation routine here by uncommenting one of the options
#calc_fractal = calc_fractal_opencl
calc_fractal = calc_fractal_serial
#calc_fractal = calc_fractal_numpy

print("before MAIN")

if __name__ == "__main__":
	print("in MAIN")
	try:
		import six.moves.tkinter as tk
	except ImportError:
		# Python 3
		print("IMPORT ERROR")
		import tkinter as tk
	from PIL import Image, ImageTk
	
	class Mandelbrot(object):
		def __init__(self):
			# create window
			self.root = tk.Tk()
			self.root.title("Mandelbrot Set")
			self.create_image()
			self.create_label()
			# start event loop
			self.root.mainloop()
			
		def draw(self, x1, x2, y1, y2, maxiter = 30):
			# draw the Mandelbrot set, from numpy example
			xx = np.arange(x1, x2, (x2-x1)/w)
			yy = np.arange(y2, y1, (y1-y2)/h) * 1j
			q = np.ravel(xx+yy[:, np.newaxis]).astype(np.complex64)
			
			start_main = time.time()
			output = calc_fractal(q, maxiter)
			end_main = time.time()
			
			secs = end_main - start_main
			print("Main took", secs)
			
			self.mandel = (output.reshape((h, w)) / float(output.max()) * 255.).astype(np.uint8)
			
		def create_image(self):
			# create the image from the draw() string
			# you can experiment with these x and y ranges
			self.draw(-2.13, 0.77, -1.3, 1.3)
			self.im = Image.fromarray(self.mandel)
			self.im.putpalette([i for rgb in ((j, 0, 0) for j in range(255)) for i in rgb])
			
		def create_label(self):
			# put the image on a label widget
			self.iage = ImageTk.PhotoImage(self.im)
			self.label = tk.Label(self.root, image=self.im)
			self.label.pack()
			
	# test the class
	print("test here")
	test = Mandelbrot()
		
		