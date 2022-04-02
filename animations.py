import cv2
import imageio as io

def cover_animation(img1, img2, filename, from_right = True, color_mode = "BGR", fps = 30, speed = 1):
    
    if color_mode == "RGB":
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    
    h1, w1, c1 = image1.shape
    h2, w2, c2 = image2.shape
    
    h = min(h1, h2)
    w = min(w1, w1)
    
    frameSize = (w, h)
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)    
    
    img1 = cv2.resize(image1, (w,h))
    img2 = cv2.resize(image2, (w,h))
    
    results = []
    
    if from_right:
        for D in range(0, w+1, speed):
            
            result = img1.copy()
            
            result[:,0:w-D,:] = img1[:,D:w,:]
            result[:,w-D:w,:] = img2[:,0:D,:]
            
            out.write(result)
            
    else:
        for D in range(0, w+1, speed):
            result = img1.copy()
            
            result[:,0:D,:] = img1[:,w-D:w,:]
            result[:,D:w,:] = img2[:,0:w-D]

            out.write(result)
            
    # io.mimsave(filename, results, fps = 60)
    out.release()
    
# cover_animation(img1, img2, filename = "results/output_video.avi", from_right = False, color_mode = "BGR", fps = 256, speed = 1) 