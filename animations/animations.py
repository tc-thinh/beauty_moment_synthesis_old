import cv2
import tensorflow as tf

def process_images_for_vid(folder_name, effect_speed, duration, fps):
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
    
    if w%effect_speed == 0:
        k = w//effect_speed
    else:
        k = w//effect_speed + 1
    
    assert duration - k/fps > 0, "change your parameters"
    
    img_list = []
    for i in range(len(images)):
        img = cv2.resize(images[i], (w,h))
        img_list.append(img)
    
    return img_list, w, h

def cover_animation(folder_name, filename, from_right = True, fps = 30, effect_speed = 2, duration = 1): # change speed to time
    
    img_list, w, h = process_images_for_vid(folder_name = folder_name, 
                                            effect_speed = effect_speed, 
                                            duration = duration, 
                                            fps = fps)
    
    # initialize video
    out = cv2.VideoWriter(r"{}".format(filename), cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))    
    
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
    
# cover_animation(folder_name = "test", filename = "results/output_video.avi", from_right = False, fps = 120, effect_speed = 1, duration = 3)

def comb_animation(folder_name, filename, fps = 30, effect_speed = 2, duration = 1): 
    
    img_list, w, h = process_images_for_vid(folder_name = folder_name, 
                                            effect_speed = effect_speed, 
                                            duration = duration, 
                                            fps = fps)
    
    # initialize video
    out = cv2.VideoWriter(r"{}".format(filename), cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))    
    
    h1 = h // lines
    for i in range(len(img_list)-1):
        j = 0
        for D in range(0, w + 1, effect_speed):
            result = img_list[0].copy()
            for L in range(0, lines, 2):
                result[h1*L:h1*(L+1), 0:D, :] = img_list[i+1][h1*L:h1*(L+1), w - D:w, :]
                result[h1*L:h1*(L+1), D:w, :] = img_list[i][h1*L:h1*(L+1), 0:w - D]
                result[h1*(L+1):h1*(L+2), 0:w - D, :] = img_list[i][h1*(L+1):h1*(L+2), D:w, :]
                result[h1*(L+1):h1*(L+2), w - D:w, :] = img_list[i+1][h1*(L+1):h1*(L+2), 0:D, :]

            out.write(result)
            j += 1

        # static image in the remaining frames
        for k in range(fps*duration - j):
            out.write(img_list[i+1])
                
    # io.mimsave(filename, results, fps = 60)
    out.release()
    
# comb_animation(folder_name = "test", filename = "results/output_video4.avi", fps = 75, effect_speed = 2, duration = 3)
