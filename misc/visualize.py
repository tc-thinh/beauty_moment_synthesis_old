import cv2
import numpy as np


def drawing_boxes(row, img):
    row = row[1]
    img_name = row['filename']
    fiqa_score = row['fiqa scores']
    smile_score = row['smile scores']

    print(img_name)

    bboxes = row['bboxes']
    face_scores = row['face scores']
    ids = row['ids']

    img = cv2.putText(img, text='Image name: {}'.format(img_name), org=(00, 185), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                      fontScale=1,
                      color=[0, 255, 0], thickness=2, lineType=cv2.LINE_AA)

    for bbox_index in range(len(bboxes)):
        current_box = bboxes[bbox_index]

        start_point = (current_box[0], current_box[1])
        end_point = (current_box[2], current_box[3])

        img = cv2.rectangle(img, start_point, end_point, [0, 0, 255], thickness=3)

        img = cv2.putText(img, text='Face scores : {}'.format(face_scores[bbox_index]),
                          org=(start_point[0], start_point[1] - 30),
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 255, 0], thickness=2,
                          lineType=cv2.LINE_AA)
        img = cv2.putText(img, text='FIQA scores : {}'.format(fiqa_score), org=(start_point[0], start_point[1] - 20),
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 255, 0], thickness=2,
                          lineType=cv2.LINE_AA)
        img = cv2.putText(img, text='Smile scores : {}'.format(smile_score), org=(start_point[0], start_point[1] - 10),
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 255, 0], thickness=2,
                          lineType=cv2.LINE_AA)
        img = cv2.putText(img, text='Name : {}'.format(ids[bbox_index]), org=(start_point[0], start_point[1] - 40),
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 255, 0], thickness=2,
                          lineType=cv2.LINE_AA)

    return img


def visualizing_bounding_boxes(df, img_list):
    img_list = list(map(drawing_boxes, list(df.iterrows()), img_list))
    img_list = np.array(img_list)

    return img_list
    