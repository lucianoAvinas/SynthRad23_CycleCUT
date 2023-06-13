import numpy as np


def ct_rescale(x):
	return 513.261 * x - 722.48


def unpad_img(x_pad, new_shape):
	# x_pad: padded numpy array
	# new_shape: original shape of x
	slice_tuple = tuple()
	for d1,d2 in zip(x_pad.shape, new_shape):
		# padded array should not be smaller than new shape
		assert d1 >= d2

        hdiff = (d1 - d2)//2
        slice_tuple += (slice(hdiff, hdiff + d2),)

    return x_pad[slice_tuple]
