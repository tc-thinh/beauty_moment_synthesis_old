import random
import time as t

import cv2
import tensorflow as tf
from tqdm import tqdm


class Image:

    def __init__(self, filename, time=500, size=500):

        self.size = size
        self.time = time
        self.shifted = 0.0
        self.img = cv2.imread(r"{}".format(filename))
        self.height, self.width, _ = self.img.shape

        if self.width < self.height:

            self.height = int(self.height * size / self.width)
            self.width = size
            self.img = cv2.resize(self.img, (self.width, self.height))
            self.shift = self.height - size
            self.shift_height = True

        else:

            self.width = int(self.width * size / self.height)
            self.height = size
            self.shift = self.width - size
            self.img = cv2.resize(self.img, (self.width, self.height))
            self.shift_height = False

        self.delta_shift = self.shift / self.time

    def reset(self):

        if random.randint(0, 1) == 0:

            self.shifted = 0.0
            self.delta_shift = abs(self.delta_shift)

        else:

            self.shifted = self.shift
            self.delta_shift = -abs(self.delta_shift)

    def get_frame(self):

        if self.shift_height:
            roi = self.img[int(self.shifted):int(self.shifted) + self.size, :, :]

        else:
            roi = self.img[:, int(self.shifted):int(self.shifted) + self.size, :]

        self.shifted += self.delta_shift

        if self.shifted > self.shift:
            self.shifted = self.shift

        if self.shifted < 0:
            self.shifted = 0

        return roi


def fade_process(folder, output_name, fps = 30, duration = 3, size = 500):
    start = time.time()
    
    filenames = []
    for i in range(len(tf.io.gfile.listdir(folder))):
        filenames.append(tf.io.gfile.join(folder, tf.io.gfile.listdir(folder)[i]))
        
    images = []
    for filename in filenames:
        print(filename)
        img = Image(filename, time = 500, size = size)
        images.append(img)
        
    prev_image = images[0]
    prev_image.reset()
    out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (size, size))
    
    for j in range(1, len(images)):
        img = images[j]
        img.reset()
        # number of frames - time = number of frames/fps
        for i in tqdm(range((duration*fps)//3)):
            alpha = i/(duration*fps)
            beta = 1.0 - alpha
            dst = cv2.addWeighted(img.get_frame(), alpha, prev_image.get_frame(), beta, 0.0)
            out.write(dst)
        prev_image = img
        for _ in tqdm(range(2*(duration*fps)//3)): # number of frames
            out.write(img.get_frame())
            
    out.release()
    end = time.time()
    print(f"Duration: {end - start}s")

# process(folder="test/bboxes", output_name=r'results\output_video1.avi')
