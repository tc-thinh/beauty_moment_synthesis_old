import cv2
import tensorflow as tf

def cover_animation(folder_name, filename, from_right = True, fps = 30, effect_speed = 2, duration = 1): # change speed to time
    
    images = []
    
    for i in range(len(tf.io.gfile.listdir(folder_name))):
        image = cv2.imread(r"{}".format(tf.io.gfile.join(folder_name, tf.io.gfile.listdir(folder_name)[i])))
        images.append(image)
    
    h = []
    w = []
    
    for i in range(len(images)):

        height, width, _ = images[0].shape
        h.append(height)
        w.append(width)
    
    h = min(h)
    w = min(w)
    
    if w%speed == 0:
        k = w//effect_speed
    else:
        k = w//effect_speed + 1
    
    assert duration - k/fps > 0, "change your parameters"
    
    frameSize = (w, h)
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)
    
    img_list = []
    for i in range(len(images)):
        img = cv2.resize(images[i], (w,h))
        img_list.append(img)
    
    if from_right:
        for i in range(len(img_list)-1):
            j = 0
            for D in range(0, w+1, effect_speed):
                result = img_list[i].copy()

                result[:,0:w-D,:] = img_list[i][:,D:w,:]
                result[:,w-D:w,:] = img_list[i+1][:,0:D,:]

                out.write(result)
                j += 1
            
            # static image in the remaining frames
            for i in range(fps*duration - j):
                out.write(result)

    else:
        for i in range(len(img_list)-1):
            j = 0
            for D in range(0, w+1, effect_speed):
                result = img_list[i].copy()

                result[:,0:D,:] = img_list[i+1][:,w-D:w,:]
                result[:,D:w,:] = img_list[i][:,0:w-D]

                out.write(result)
                j += 1
            
            # static image in the remaining frames
            for i in range(fps*duration - j):
                out.write(result)
                
    # io.mimsave(filename, results, fps = 60)
    out.release()

# cover_animation(img1, img2, filename = "results/output_video.avi", from_right = False, color_mode = "BGR", fps = 256, speed = 1)
