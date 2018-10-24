from PIL import Image
import numpy as np
import time

def get_image(fname,size):
	im = Image.open(fname)
	im.thumbnail(size,Image.LANCZOS)
	# if fname == './ICDAR/test/char/28/2724.jpg':
	# im.sve('test_'+fname.replace('/','-'),'JPEG')
	pix = im.getdata()
	pix = np.array(pix).reshape((im.size[1],im.size[0],3))
	# print(fname,pix.shape)
	# exit()
	result = np.zeros((size[1],size[0],3))
	rowpad = size[0]-pix.shape[0]
	colpad = size[1]-pix.shape[1]
	result[rowpad//2:rowpad//2+pix.shape[0],colpad//2:colpad//2+pix.shape[1],:] = pix
	image = Image.fromarray(result.astype('uint8'), 'RGB')
	# if fname == './ICDAR/test/char/28/2724.jpg':
	# 	np.set_printoptions(threshold=np.nan)
	# 	print((result)[0,24,:])
	# 	image.save('test'+fname.replace('/','-'),'JPEG')
	return result
