import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np

def load_model(model_path):
    model = load_model(model_path)
    model.compile(optimizer = tf.keras.optimizers.Adam(0.0001), 
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])
    return model

def get_smile_score(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (139, 139))
    img = np.reshape(img, [1, 139, 139, 3])

    preditions = model.predict(img)
    max_index = np.argmax(preditions[0])
    emotions = ('happy', 'unhappy')
    predicted_emotion = emotions[max_index]
    return preditions[0][0]*100
