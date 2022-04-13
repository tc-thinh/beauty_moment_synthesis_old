import cv2


def get_target_bbox(img_path, bboxes, p=0.1):
    img = cv2.imread(img_path)
    data = []
    for bbox in bboxes:
        x, y = int(bbox[0]), int(bbox[1])  # top-left x, y corrdinates
        w, h = int(bbox[2]), int(bbox[3])  # w, h values

        if y - int(p * w) < 0 or x - int(p * h) < 0 or y + int(p * w) > img.shape[0] or y + int(p * w) > img.shape[1] \
                or x + int(p * w) > img.shape[1] or x + int(p * w) > img.shape[0]:
            data.append(img[y:y + w, x:x + h])
        else:
            data.append(img[y - int(p * w):y + w + int(p * w), x - int(p * h):x + h + int(p * h)])  # target box

    return data
