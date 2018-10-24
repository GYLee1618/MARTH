from PIL import Image
import numpy as np
import time

def get_image(fname,size):
    im = Image.open(fname)
    im.thumbnail(size,Image.LANCZOS)
    im.save('test'+str(time.time()),'JPEG')
    pix = im.getdata()
    pix = np.array(pix).reshape((im.size[0],im.size[1],3))

    result = np.zeros((size[0],size[1],3))
    result[:pix.shape[0],:pix.shape[1],:] = pix
    return result

