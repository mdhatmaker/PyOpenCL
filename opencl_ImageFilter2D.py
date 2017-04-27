# (from OpenCL Programming Guide)
# (http://www.openclprogrammingguide.com)
# This example demonstrates performing gaussian filtering on a
# 2D image using OpenCL.

import pyopencl as cl
import sys
from PIL import Image	#, ImageFilter
import numpy

#
# Create an OpenCL context on the first available platform using
# either a GPU or CPU depending on what is available.
#
def CreateContext():
	platforms = cl.get_platforms()
	if len(platforms) == 0:
		print "Failed to find any OpenCL platforms."
		return None
	# Next, create an OpenCL context on the first platform.  Attempt to
	# create a GPU-based context, and if that fails, try to create
	# a CPU-based context.
	devices = platforms[0].get_devices(cl.device_type.GPU)
	if len(devices) == 0:
		print "Could not find GPU device, trying CPU..."
		devices = platforms[0].get_devices(cl.device_type.CPU)
		if len(devices) == 0:
			print "Could not find OpenCL GPU or CPU device."
			return None
			
	# Create a context using the first device
	context = cl.Context([devices[0]])
	return context, devices[0]
	
#
# Create an OpenCL program from the kernel source file
#
def CreateProgram(context, device, filename):
	kernelFile = open(filename, 'r')
	kernelStr = kernelFile.read()
	
	# Load the program source
	program = cl.Program(context, kernelStr)
	
	# Build the program and check for errors
	program.build(devices=[device])
	
	return program
	
	
#
# Load an image using the Python Image Library and create an OpenCL
# image out of it
#
def LoadImage(context, filename):
	im = Image.open(filename)
	# Make sure the image is RGBA formatted
	if im.mode != "RGBA":
		im = im.convert("RGBA")
		
	# Convert to uint8 buffer
	buffer = im.tostring()
	clImageFormat = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)
	
	clImage = cl.Image(context,
						cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
						clImageFormat,
						im.size,
						None,
						buffer
						)
						
	return clImage, im.size
	
#
# Save an image using the Python Image Library (PIL)
#
def SaveImage(filename, buffer, imgSize):
	im = Image.fromstring("RGBA", imgSize, buffer.tostring())
	im.save(filename)
	
#
# Round up to the nearest multiple of the group size
#
def RoundUp(groupSize, globalSize):
	r = globalSize % groupSize
	if r == 0:
		return globalSize
	else:
		return globalSize + groupSize - r
		
def main():
	imageObjects = [0, 0]
	
	# Main
	if len(sys.argv) != 3:
		print "USAGE: " + sys.argv[0] + " <iputImageFile> <outputImageFile>"
		return 1
		
	# Create an OpenCL context on first available platform
	context, device = CreateContext()
	if context == None:
		print "Failed to create OpenCL context."
		return 1
		
	# Create a command-queue on the first device available
	# on the created context
	commandQueue = cl.CommandQueue(context, device)
	
	# Make sure the device supports images, otherwise exit
	if not device.get_info(cl.device_info.IMAGE_SUPPORT):
		print "OpenCL device does not support images."
		return 1
		
	# Load input image from file and load it into
	# an OpenCL image object
	imageObjects[0], imgSize = LoadImage(context, sys.argv[1])
	
	# Create output image object
	clImageFormat = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)
	
	imageObjects[1] = cl.Image(context, cl.mem_flags.WRITE_ONLY, clImageFormat, imgSize)
	
	# Create sample for sampling image object
	sampler = cl.Sampler(context,
							False, # Non-normalized coordinates
							cl.addressing_mode.CLAMP_TO_EDGE,
							cl.filter_mode.NEAREST)
							
	# Create OpenCL program
	program = CreateProgram(context, device, "ImageFilter2D.cl")
	
	# Call the kernel directly
	localWorkSize = (16, 16)
	globalWorkSize = (RoundUp(localWorkSize[0], imgSize[0]), RoundUp(localWorkSize[1], imgSize[1]))
	
	program.gaussian_filter(commandQueue,
							globalWorkSize,
							localWorkSize,
							imageObjects[0],
							imageObjects[1],
							sampler,
							numpy.int32(imgSize[0]),
							numpy.int32(imgSize[1]))
							
	# Read the output buffer back to the Host
	buffer = numpy.zeros(imgSize[0] * imgSize[1] * 4, numpy.uint8)
	origin = (0, 0, 0)
	region = (imgSize[0], imgSize[1], 1)
	
	cl.enqueue_read_image(commandQueue, imageObjects[1], origin, region, buffer).wait()
	
	print "Executed program successfully."
	
	# Save the image to disk
	SaveImage(sys.argv[2], buffer, imgSize)
	
main()
