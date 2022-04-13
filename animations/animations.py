import random

import cv2
from tqdm import tqdm


def process_images_for_vid(img_list, effect_speed, duration, fps, fraction):
    images = []

    for i in range(len(img_list)):
        image = cv2.imread(r"{}".format(img_list[i]))
        images.append(image)

    h = []
    w = []

    for image in images:
        height, width, _ = image.shape
        h.append(height)
        w.append(width)

    h = int(min(h)/fraction)
    w = int(min(w)/fraction)

    if w % effect_speed == 0:
        k = w // effect_speed
    else:
        k = w // effect_speed + 1

    assert duration - k / fps > 0, "change your parameters"

    img_list = []
    for image in images:
        img = cv2.resize(image, (w, h))
        img_list.append(img)

    return img_list, w, h


def cover_animation(img_list, w, h, from_right=random.randint(0, 1), fps=30, effect_speed=2,
                    duration=1):  # change speed to time

    frames = []

    if from_right:
        for i in range(len(img_list) - 1):
            j = 0
            for D in range(0, w + 1, effect_speed):
                result = img_list[i].copy()

                result[:, 0:w - D, :] = img_list[i][:, D:w, :]
                result[:, w - D:w, :] = img_list[i + 1][:, 0:D, :]

                frames.append(result)
                j += 1

            # static image in the remaining frames
            for _ in range(fps * duration - j):
                frames.append(img_list[i + 1])
    else:
        for i in range(len(img_list) - 1):
            j = 0
            for D in range(0, w + 1, effect_speed):
                result = img_list[i].copy()

                result[:, 0:D, :] = img_list[i + 1][:, w - D:w, :]
                result[:, D:w, :] = img_list[i][:, 0:w - D]

                frames.append(result)
                j += 1

            # static image in the remaining frames
            for _ in range(fps * duration - j):
                frames.append(result)

    return frames


def comb_animation(img_list, w, h, fps=30, effect_speed=2, duration=1):
    lines = random.randint(1, 6)
    frames = []
    h1 = h // lines

    for i in range(len(img_list) - 1):
        j = 0
        for D in range(0, w + 1, effect_speed):
            result = img_list[0].copy()
            for L in range(0, lines, 2):
                result[h1 * L:h1 * (L + 1), 0:D, :] = img_list[i + 1][h1 * L:h1 * (L + 1), w - D:w, :]
                result[h1 * L:h1 * (L + 1), D:w, :] = img_list[i][h1 * L:h1 * (L + 1), 0:w - D]
                result[h1 * (L + 1):h1 * (L + 2), 0:w - D, :] = img_list[i][h1 * (L + 1):h1 * (L + 2), D:w, :]
                result[h1 * (L + 1):h1 * (L + 2), w - D:w, :] = img_list[i + 1][h1 * (L + 1):h1 * (L + 2), 0:D, :]

            frames.append(result)
            j += 1

        # static image in the remaining frames
        for k in range(fps * duration - j):
            frames.append(img_list[i + 1])


def push_animation(img_list, w, h, fps=30, effect_speed=2, duration=1):
    frames = []

    for i in range(len(img_list) - 1):
        j = 0
        for D in range(0, h + 1, effect_speed):
            result = img_list[i].copy()
            result[0:h - D, :, :] = img_list[i][D:h, :, :]
            result[h - D:h, :, :] = img_list[i + 1][0:D, :, :]

            frames.append(result)
            j += 1

        # static image in the remaining frames
        for k in range(fps * duration - j):
            frames.append(img_list[i + 1])

    return frames


def uncover_animation(img_list, w, h, fps=30, effect_speed=2, duration=1):
    frames = []

    for i in range(len(img_list) - 1):
        j = 0
        for D in range(0, w + 1, effect_speed):
            result = img_list[i].copy()
            result[:, 0:w - D, :] = img_list[i][:, D:w, :]
            result[:, w - D:w, :] = img_list[i + 1][:, w - D:w, :]

            frames.append(result)
            j += 1

        # static image in the remaining frames
        for k in range(fps * duration - j):
            frames.append(img_list[i + 1])

    return frames


def split_animation(img_list, w, h, fps=30, effect_speed=2, duration=1):
    frames = []

    for i in range(len(img_list) - 1):
        j = 0
        for D in range(0, w // 2, effect_speed):
            result = img_list[i].copy()
            result[:, w // 2 - D:w // 2 + D, :] = img_list[i + 1][:, w // 2 - D:w // 2 + D, :]
            result[:, 0:w // 2 - D, :] = img_list[i][:, 0:w // 2 - D, :]
            result[:, w // 2 + D:w, :] = img_list[i][:, w // 2 + D:w, :]

            frames.append(result)
            j += 1

        # static image in the remaining frames
        for k in range(fps * duration - j):
            frames.append(img_list[i + 1])

    return frames


# comb_animation(folder_name = "test", filename = "results/output_video4.avi", fps = 75, effect_speed = 2, duration = 3)


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


def fade_animation(img_list, w, h, fps=30, effect_speed=2, duration=1):
    frames = []
    prev_image = img_list[0]
    prev_image.reset()

    for j in range(1, len(img_list)):
        img = img_list[j]
        img.reset()
        # number of frames - time = number of frames/fps
        for i in tqdm(range((duration * fps) // 3)):
            alpha = i / (duration * fps)
            beta = 1.0 - alpha
            dst = cv2.addWeighted(img.get_frame(), alpha, prev_image.get_frame(), beta, 0.0)
            frames.append(dst)
        prev_image = img
        for _ in tqdm(range(2 * (duration * fps) // 3)):  # number of frames
            frames.append(img.get_frame())

    return frames


def extract_vid(frames, output_path, w=500, h=500, fps=30):
    out = cv2.VideoWriter(r"{}".format(output_path), cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))

    for image in frames:
        for frame in image:
            out.write(frame)

    out.release()
