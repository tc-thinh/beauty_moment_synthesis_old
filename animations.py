import cv2
import imageio as io

def cover_animation(img1, img2, filename, color_mode = "RGB", speed = 6, save_mode = True):
    
    if color_mode == "BGR":
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
 
    h = min(h1, h2)
    w = min(w1, w1)

    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))
    
    results = []
    for D in range(0, w+1, speed):
        result = img1.copy()
        result[:,0:w-D,:] = img1[:,D:w,:]
        result[:,w-D:w,:] = img2[:,0:D,:]
        # create gif file
        results.append(result)

    if save_mode:
        io.mimsave(filename, results)
    else:
        return results