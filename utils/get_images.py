from PIL import Image
import numpy as np

def get_image(fname,size):
    im = Image.open(fname)
  
    im.thumbnail(size,Image.LANCZOS)
   	if (fname == './ICDAR/train/char/1/100.jpg') :
    	im.show()
    pix = im.getdata()
    pix = np.array(pix).reshape((im.size[0],im.size[1],3))

    result = np.zeros((size[0],size[1],3))
    result[:pix.shape[0],:pix.shape[1],:] = pix
    return result

