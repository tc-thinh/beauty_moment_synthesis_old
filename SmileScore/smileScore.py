import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import pandas as pd
from misc.extract_bbox import *


def load_smile_model(model_path):
    model = load_model(model_path)
    model.compile(optimizer = tf.keras.optimizers.Adam(0.0001), 
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    return model


def get_smile_score(df, img_list, model):
    smile_score = []
    filename = []
    final_img = []
    for i in range(len(df)):
        if df["bboxes"][i][0] is not None:
            input_data = get_target_bbox(img_list[i], df["bboxes"][i], p=0.15)
            score = []
            for j in input_data:
                img = cv2.resize(j, (139, 139))
                img = np.reshape(img, [1, 139, 139, 3])

                preditions = model.predict(img)

                score.append(preditions[0][0] * 100)

            filename.append(df["filename"][i])
            final_img.append(img_list[i])
            smile_score.append(sum(score) / len(score))

    new_df = pd.DataFrame({'filename': filename, 'score': smile_score})
    new_df.sort_values(by="score", ascending=False, inplace=True)
    old_index = list(new_df.index)
    final_img = [final_img[i] for i in old_index]

    new_df.reset_index(drop=True)
    return new_df, np.array(final_img)
