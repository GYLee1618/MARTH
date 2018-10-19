from PIL import Image
import numpy as np

def get_image(fname):
    im = Image.open(fname)
    pix = im.getdata()
    
    pix = np.array(pix).reshape((im.size[0],im.size[1],3))

    return pix

