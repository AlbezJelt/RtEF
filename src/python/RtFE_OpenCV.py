import cv2 as cv
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
   
    # OpenCV read the images as uint8, BGR color space. Cast to float32
    ov = cv.imread("../../images/window/ov.jpg").astype(np.float32)   
    un = cv.imread("../../images/window/un.jpg").astype(np.float32)

    # Convert BGR to HSV_FULL. 
    # OpenCV should return values in the range [0, 1] for V channel (https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_hsv)
    # But it returns 255*V, so its divided by 255
    factors = np.array([1., 1., 255.], np.float32)[np.newaxis, np.newaxis, :]
    ov = cv.cvtColor(ov, cv.COLOR_BGR2HSV_FULL) / factors
    un = cv.cvtColor(un, cv.COLOR_BGR2HSV_FULL) / factors

    # New image, zeros array with same the same shape as input
    fused_image = np.zeros(un.shape, np.float32)

    # Thresholds
    th1 = np.amax(un[:, :, 2]) * 0.05
    th2 = np.amax(ov[:, :, 2]) * 0.95

    # Iter on each pixel
    width, height, _ = ov.shape
    for x, y in product(range(width), range(height)):
        fused_image[x, y] = fuse_pixel(*ov[x, y, ], *un[x, y, ], th1, th2) 
        # print(f"{x}, {y}: ov => {img1[x, y, ]} \t un => {img2[x, y, ]} \t fu => {fused_image[x, y, ]}")

    # Reconvert to BGR color space
    fused_image = cv.cvtColor(fused_image * factors, cv.COLOR_HSV2BGR_FULL)

    # Save the image
    cv.imwrite('fu.png', fused_image)