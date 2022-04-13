import tensorflow as tf
from keras.models import load_model
import cv2
import os
import numpy as np
import pandas as pd
from misc.extract_bbox import *

def load_model(model_path):
    model = load_model(model_path)
    model.compile(optimizer = tf.keras.optimizers.Adam(0.0001), 
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])
    return model

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
def get_smile_score(df, model):
    smile_score = []
    filename = []
    for i in range(len(df)):
        if df["bboxes"][i][0] is not None:
            input_data = get_target_bbox(os.path.join("test/img", df["filename"][i]), df["bboxes"][i], p=0.15)
            score = []
            for j in input_data:
                img = cv2.cvtColor(j, cv2.COLOR_BGR2RGB)

                img = cv2.resize(img, (139, 139))
                img = np.reshape(img, [1, 139, 139, 3])

                preditions = model.predict(img)
                
                score.append(preditions[0][0]*100)
            filename.append(df["filename"][i])
            smile_score.append(sum(score))
    new_df = pd.DataFrame({'filename': filename, 'score': smile_score})
    sorted_df = new_df.sort_values(by='score', ascending=False)

    return sorted_df[filename]
