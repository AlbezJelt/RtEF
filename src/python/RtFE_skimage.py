from skimage.io import (imread, imsave)
from skimage.color import (rgb2hsv, hsv2rgb)
import numpy as np
from itertools import product

def fuse_v_channel(Vov, Vun, th1, th2):
    diff = Vov - Vun
    if Vun < th1: # class = black saturated
        return _Vfu_black(Vov, diff)
    elif Vov > th2: # class = white saturated
        return _Vfu_white(Vun, diff)
    else:
        return ((_Vfu_black(Vov, diff) * (1 - Vov)) + (_Vfu_white(Vun, diff) * Vun)) / (1 - diff)
    
def _Vfu_black(Vov, diff):
    return Vov - (diff / 4)

def _Vfu_white(Vun, diff):
    return Vun + (diff / 4)

def visual_saturation_factor_ov(Sov, Vfu):
    return Sov * (1 - Vfu)

def visual_saturation_factor_un(Sun, Vfu):
    return Sun * (Vfu)

def fuse_pixel(Hov, Sov, Vov, Hun, Sun, Vun, th1, th2):
    Vfu = fuse_v_channel(Vov, Vun, th1, th2)
    VSFun = visual_saturation_factor_un(Sun, Vfu)
    VSFov = visual_saturation_factor_ov(Sov, Vfu)
    if VSFov <= VSFun:
        return Hun, VSFun, Vfu
    else:
        return Hov, VSFov, Vfu
    
if __name__ == "__main__":
   
    ov = imread("../../images/window/ov.jpg") 
    un = imread("../../images/window/un.jpg")

    ov = rgb2hsv(ov)
    un = rgb2hsv(un)

    # New image, zeros array with same the same shape as input
    fused_image = np.zeros(un.shape, np.float64)

    # Thresholds
    th1 = np.amax(un[:, :, 2]) * 0.05
    th2 = np.amax(ov[:, :, 2]) * 0.95

    # Iter on each pixel
    width, height, _ = ov.shape
    for x, y in product(range(width), range(height)):
        fused_image[x, y] = fuse_pixel(*ov[x, y, ], *un[x, y, ], th1, th2) 
        # print(f"{x}, {y}: ov => {img1[x, y, ]} \t un => {img2[x, y, ]} \t fu => {fused_image[x, y, ]}")

    # Reconvert to RGB color space
    fused_image = hsv2rgb(fused_image)

    # Save the image
    imsave('fu.png', (fused_image * 255).astype(np.uint8))