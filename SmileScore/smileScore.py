from keras.models import load_model
from misc.extract_bbox import *
from config import *
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import os
from deepface import DeepFace
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_smile_model(model_path):
  model = load_model(model_path)
  model.compile(optimizer = tf.keras.optimizers.Adam(0.0001),
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
  return model


def get_smile_score(df, img_list, model):
  smile_score_avg = []
  final_img = []
  smile_scores = []

  for i in range(len(df)):
    input_data = get_target_bbox(img_list[i], df["bboxes"][i], p = CFG_FIQA.EXTEND_RATE)
    scores = []
    for j in input_data:
      predictions = DeepFace.analyze(j)
      scores.append(predictions['emotion']['happy'])

    smile_scores.append([[score] for score in scores])
    smile_score_avg.append(sum(scores) / len(scores))
    final_img.append(img_list[i])

  new_df = df.copy()
  new_df['smile score average'] = smile_score_avg
  new_df['smile scores'] = smile_scores
  new_df.sort_values(by = 'smile score average', ascending = False, inplace = True)
  old_index = list(new_df.index)
  final_img = [final_img[i] for i in old_index]

  new_df.reset_index(drop = True)
  return new_df, np.array(final_img)