from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms
import numpy as np
import cv2
import pandas as pd
import math
import os
from sklearn.neighbors import KNeighborsClassifier
from numpy import dot
from numpy.linalg import norm


def read_images(path):
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_input_images(img_list, fraction=1):
    all_width = [img_list[i].shape[-3] for i in range(len(img_list))]
    mean_width = int(sum(all_width) / len(all_width) * fraction)
    all_height = [img_list[i].shape[-2] for i in range(len(img_list))]
    mean_height = int(sum(all_height) / len(all_height) * fraction)

    img_list = np.stack([cv2.resize(img_list[i], (mean_height, mean_width)) for i in range(len(img_list))], axis=0)

    return img_list


def resize_anchor_images(path):
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 600))

    return img


def read_input_images(path, purpose='anchor', different_sizes=True):
    """
    This function reads all the images in the input dataset.

    Parameters
    ----------
    + path: str .
    A path to your input dataset

    + purpose: {'anchor', 'input'}, optional, default 'anchor'.
    Since this function will be reused in the read_anchor_images, please don't modify this argument.

    Return
    ---------
    Return all the images in the input dataset in the Python Numpy Array format, as well as their names.

    """
    all_path = [os.path.join(path, name) for name in os.listdir(path)]
    file_name = [name for name in os.listdir(path)]

    if purpose == 'input':
        img_list = list(map(read_images, all_path))
        if different_sizes:
            img_list = resize_input_images(img_list)
        return file_name, img_list

    elif purpose == 'anchor':
        img_list = list(map(resize_anchor_images, all_path))
        return img_list


def read_anchor_images(path):
    """
    This function reads all the images in the anchor dataset.

    Parameters
    ----------
    + path: str .
    A path to your anchor dataset

    Return
    ---------
    + img_flatten_list: np.ndarray
    Return all the images in the anchor dataset in the Python Numpy Array format.

    + img_label: np.ndarray
    Return people's ids for those images (needed for Face Recognition).

    """
    folder_name = [name for name in os.listdir(path) if name[-3:] != 'txt']
    folder_name.sort()
    folder_path = [os.path.join(path, name) for name in folder_name]
    img_list = list(map(read_input_images, folder_path))

    img_flatten_list = []
    img_label = []
    for ind in range(len(img_list)):
        sublist = img_list[ind]
        for img in sublist:
            img_label.append(folder_name[ind])
            img_flatten_list.append(img)

    img_flatten_list = np.stack(img_flatten_list, axis=0)
    img_label = np.array(img_label).reshape(-1, )

    return img_label, img_flatten_list


def create_facenet_models():
    """
  This function returns an MTCNN model base - which was used to detect human
  faces in images.
  Original GitHub Repository: https://github.com/timesler/facenet-pytorch

  To have a better understanding of this model's parameters,
  use the Python built-in help () function
  >> help (mtcnn_model_name)
  Example
  >> model_A = create_mtcnn_model()
  >> help (model_A)

  To calibrate again MTCNN model's parameters after calling out this function:
  >> mtcnn_model_name.parameters_want_to_change = ...
  Example
  >> model_A = create_mtcnn_model()
  >> model_A.image_size = 200
  >> model_A.min_face_size = 10
  """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Used device: {}'.format(device))
    infer_model = InceptionResnetV1(pretrained='vggface2', device=device).eval()

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=50,
        thresholds=[0.7, 0.7, 0.8], post_process=False,
        device=device, selection_method='largest_over_threshold'
    )

    if mtcnn is not None and infer_model is not None:
        print('Successfully create a MTCNN + InceptionResnet model base')
    else:
        print('Fail to create a MTCNN + InceptionResnet model base')

    mtcnn.keep_all = False

    return mtcnn, infer_model


def get_bounding_box(mtcnn_model, frames, batch_size=32):
    """
  This function detects human faces in the given batch of images / video frames
  in the Python Numpy Array format. It will return 3 lists - bounding box coordinates
  list, confidence score list, and facial landmarks list. See details on Return section.

  To save GPU's memory, this function will detect faces in
  separate mini-batches. It has been shown that mini-batched detection has the
  same efficiency as full-batched detection.

  For each detected face, the function will return:
  * 4 bounding box coordinates (x left, y top, x right, y bot)
  * 1 confidence score for that bounding box.
  * 5 landmarks - marking that person's eyes, nose, and mouth.

  Parameters
  ----------
  + mtcnn_model: a facenet_pytorch.models.mtcnn.MTCNN model.
      Passing a MTCNN model base that has been created beforehand.

  + frames: np.ndarray.
      Given batch of images to detect human faces. Must be a Numpy Array that
      has 4D shape.
      --> The ideal shape should be:
      (number_of_samples, image_width, image_height, channels)

      All images in the Frames must be of equal size, and has all pixel values
      in scale [0-255].
      All images must have 3 color channels (RGB-formatted images).

  + batch_size: int > 0, optional, default is 32.
      The size of the mini-batch. The larger the batch size, the more GPU memory
      needed for detection.

  Return
  ----------
  + bboxes_pred_list: list .
      The list that contains all the predicted bounding boxes in the OpenCV format
      [x_left, y_top, x_right, y_bot]

  + box_probs_list: list .
      The list that contains the confidence scores for all predicted bounding
      boxes.

  + landmark_list: list .
      The list that contains facial landmarks for all predicted bounding boxes/
  """

    assert (type(frames) == np.ndarray and frames.ndim == 4), "Frames must be a 4D np.array"
    assert (frames.shape[-1] == 3), "All images must have 3 color channels - R, G, and B"
    assert (type(batch_size) == int and batch_size > 0), "Batch size must be an integer number, larger than 0"

    size_checking = all(frame.shape == frames[0].shape for frame in frames)
    assert size_checking, "All the images must be of same size"

    frames = frames.astype(np.uint8)
    steps = math.ceil(len(frames) / batch_size)
    frames = np.array_split(frames, steps)

    bboxes_pred_list = []
    box_probs_list = []
    landmark_list = []

    for batch_file in frames:
        with torch.no_grad():
            bb_frames, box_probs, landmark = mtcnn_model.detect(batch_file, landmarks=True)

        for ind in range(len(bb_frames)):
            if bb_frames[ind] is not None:
                bboxes_pred_list.append(bb_frames[ind].tolist())
                box_probs_list.append(box_probs[ind].tolist())
                landmark_list.append(landmark[ind].tolist())

            else:
                bboxes_pred_list.append([None])
                box_probs_list.append([None])
                landmark_list.append([None])

    return bboxes_pred_list, box_probs_list, landmark_list


def convert_bounding_box(box, input_type, change_to):
    """
    This function converts an input bounding box to either YOLO, COCO, or OpenCV
    format.
    However, the function only converts the input bounding box if it already belongs
    to one of the three formats listed above.

    Note:
    + OpenCV-formatted bounding box has 4 elements [x_left, y_top, x_right, y_bot]
    + YOLO-formatted bounding box has 4 elements [x_center, y_center, width, height]
    + COCO-formatted bounding box has 4 elements [x_left, y_top, width, height]

    Parameters
    ----------
    + box : list.
        The provided bounding box in the Python list format. The given bounding box
        must have 4 elements, corresponding to its format.

    + input_type : {'opencv', 'yolo', 'coco'}
        The format of the input bounding box.
        Supported values are 'yolo' - for YOLO format, 'coco' - for COCO format,
        and 'opencv' - for OpenCV format.

    + change_to : {'opencv', 'yolo', 'coco'}.
        The type of format to convert the input bounding box to.
        Supported values are 'yolo' - for YOLO format, 'coco' - for COCO format,
        and 'opencv' - for OpenCV format.

    Return
    ----------
        Returns a list for the converted bounding box.

    """
    assert (type(box) == list), 'The provided bounding box must be a Python list'
    assert (len(box) == 4), 'Must be a bounding box that has 4 elements: [x_left, y_top, x_right, y_bot] (OpenCV format)'
    assert (input_type == 'yolo' or input_type == 'coco' or input_type == 'opencv'), "Must select either 'yolo', 'coco', or 'opencv' as a format of your input bounding box"
    assert (change_to == 'yolo' or change_to == 'coco' or change_to == 'opencv'), "Must select either 'yolo', 'coco', or 'opencv' as a format you want to convert the input bounding box to"
    assert (input_type != change_to), "The format of your input bounding box must be different from your output bounding box."

    if input_type == 'opencv':
        x_left, y_top, x_right, y_bot = box[0], box[1], box[2], box[3]

        if change_to == 'yolo':
            x_center = int((x_left + x_right) / 2)
            y_center = int((y_top + y_bot) / 2)
            width = int(x_right - x_left)
            height = int(y_bot - y_top)

            return [x_center, y_center, width, height]

        elif change_to == 'coco':
            width = int(x_right - x_left)
            height = int(y_bot - y_top)

            return [x_left, y_top, width, height]

    elif input_type == 'yolo':
        x_center, y_center, width, height = box[0], box[1], box[2], box[3]

        if change_to == 'opencv':
            x_left = int(x_center - width / 2)
            x_right = int(x_center + width / 2)
            y_top = int(y_center - height / 2)
            y_bot = int(y_center + height / 2)

            return [x_left, x_right, y_top, y_bot]

        elif change_to == 'coco':
            x_left = int(x_center - width / 2)
            y_top = int(y_center - height / 2)

            return [x_left, y_top, width, height]

    elif input_type == 'coco':
        x_left, y_top, width, height = box[0], box[1], box[2], box[3]

        if change_to == 'opencv':
            x_right = int(x_left + width)
            y_bot = int(y_top + height)

            return [x_left, x_right, y_top, y_bot]

        elif change_to == 'yolo':
            x_center = int(x_left + width / 2)
            y_center = int(y_top + height / 2)

            return [x_center, y_center, width, height]


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def transform(img):
    normalized = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return normalized(img)


def filter_images(name, img_list, boxes):
    #   discard_index = [i for i, x in enumerate(boxes) if x[0] == [None]]
    #   if discard_index != [None]:
    #       discard_name = [name[i] for i in discard_index]

    keep_index = [i for i, x in enumerate(boxes) if x[0] != [None]]

    img_final = [img_list[i] for i in keep_index]
    box_list_final = [boxes[i] for i in keep_index]
    name_final = [name[i] for i in keep_index]
    return np.array(img_final), box_list_final, name_final


def clipping(img_list, boxes):
    """
    This function clips the predicted bounding boxes to appropriate ranges.

    Parameters
    ----------
    + img_list: np.ndarray.
    The given input image list (output of the read_input_images function)

    + boxes: np.ndarray.
    The predicted bounding boxes
    """

    def clipping_method(img, box, format='opencv'):
        if format == 'opencv':
            x_left, y_top, x_right, y_bot = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            x_left = min(max(x_left, 0), img.shape[-2])
            y_top = min(max(y_top, 0), img.shape[-3])  # (h, w, 3)
            x_right = min(max(x_right, 0), img.shape[-2])
            y_bot = min(max(y_bot, 0), img.shape[-3])

            return [x_left, y_top, x_right, y_bot]

        elif format == 'coco':
            x_left, y_top, width, height = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            x_left = min(max(x_left, 0), img.shape[-3])
            y_top = min(max(y_top, 0), img.shape[-2])
            width = min(max(width, 0), img_list.shape[-3] - x_left)
            height = min(max(height, 0), img_list.shape[-2] - y_top)

            return [x_left, y_top, width, height]

        elif format == 'yolo':
            x_center, y_center, width, height = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            x_center = min(max(x_center, 0), img.shape[0])
            y_center = min(max(y_center, 0), img.shape[0])
            width = min(max(width, 0), img_list.shape[0])
            height = min(max(height, 0), img_list.shape[1])

            return [x_center, y_center, width, height]

    box_clipping = []

    for i in range(len(img_list)):
        if len(boxes[i]) > 1:
            img_list_map = np.expand_dims(img_list[i], axis=0)
            img_list_map = np.repeat(img_list_map, repeats=len(boxes[i]), axis=0)
            box_clipping.append(list(map(clipping_method, img_list_map, boxes[i])))

        elif len(boxes[i]) == 1:
            if boxes[i][0] is not None:
                box_clipping.append([clipping_method(img_list[i], box=boxes[i][0], format='opencv')])

            else:
                box_clipping.append([[None]])

    return box_clipping


def cropping_face(img_list, box_clipping, percent=0, purpose='input'):
    def crop_with_percent(img, box, percent=0):
        x_left, y_top, x_right, y_bot = box[0], box[1], box[2], box[3]  # [x_left, y_top, x_right, y_bot]

        x_left -= percent * (x_right - x_left)
        x_right += percent * (x_right - x_left)
        y_top -= percent * (y_bot - y_top)
        y_bot += percent * (y_bot - y_top)
        target_img = img[int(y_top): int(y_bot), int(x_left): int(x_right)]

        target_img = cv2.resize(target_img, (240, 300), interpolation=cv2.INTER_CUBIC)  # cv2 resize (height, width)

        return np.array(target_img).astype('int16')

    if purpose == 'input':
        faces = []
        for i in range(len(img_list)):
            if len(box_clipping[i]) > 1:
                img_list_map = np.expand_dims(img_list[i], axis=0)
                img_list_map = np.repeat(img_list_map, repeats=len(box_clipping[i]), axis=0)
                faces.append(list(map(crop_with_percent, img_list_map, box_clipping[i])))

            elif len(box_clipping[i]) == 1:
                faces.append([crop_with_percent(img_list[i], box_clipping[i][0])])

    elif purpose == 'anchor':
        faces = [crop_with_percent(img_list[i], box_clipping[i][0], percent) for i in range(len(img_list))]

    return faces


def vector_embedding(infer_model, img_list, purpose='input'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def extract_vector(batch):
        batch = batch.to(device)
        embed = infer_model(batch)
        embed = embed.cpu().detach().numpy()
        return embed

    if purpose == 'anchor':
        img_list = list(map(transform, img_list))
        img_list = torch.stack(img_list, dim=0)

        batch_size = 32
        steps = math.ceil(len(img_list) / batch_size)
        img_list = torch.split(img_list, steps)

        vector_embeddings = list(map(extract_vector, img_list))
        vector_embeddings = np.concatenate(vector_embeddings).reshape(-1, 512)

    elif purpose == 'input':

        vector_embeddings = []

        for img in img_list:
            mini_list = list(map(transform, img))
            mini_list = torch.stack(mini_list, dim=0)

            embedding = extract_vector(mini_list)

            if embedding.shape[0] == 1:
                embedding = list(embedding)
                vector_embeddings.append(embedding)
            else:
                embedding = np.concatenate(embedding).reshape(-1, 512)
                embedding = list(embedding)
                vector_embeddings.append(embedding)

    return vector_embeddings


def get_knn(x, y):
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', algorithm='brute')
    knn.fit(x, y)

    return knn


def knn_predict(model, embeddings):
    predicted_ids = [list(model.predict(embedding)) for embedding in embeddings]

    final_ids = []
    for ids in predicted_ids:
        collected_ids = np.array(ids)
        collected_ids = np.array_split(collected_ids, len(ids))
        collected_ids = [id.tolist() for id in collected_ids]
        final_ids.append(collected_ids)

    return final_ids


def names_to_integers(list_name):
    unique_names = np.unique(list_name)
    label_to_int = {label: i for i, label in enumerate(unique_names)}
    int_to_label = {i: label for i, label in enumerate(unique_names)}
    mapped_name = np.array([label_to_int[name] for name in list_name]).astype('int16')

    return int_to_label, mapped_name


def euclidean_distance(row1, row2):
    euclidean_dist = np.linalg.norm(row1 - row2[:-1])

    return (euclidean_dist, int(row2[-1]))


def cosine_distance(row1, row2):
    return dot(row1, row2) / (norm(row1) * norm(row2))


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    test_rows = np.array([test_row] * len(train))
    euclidean_distances = list(map(euclidean_distance, test_rows, train))
    euclidean_distance_index = euclidean_distances.copy()
    euclidean_distance_index = sorted(range(len(euclidean_distance_index)),
                                      key=lambda tup: euclidean_distance_index[tup])
    euclidean_distances.sort(key=lambda tup: tup[0])
    neighbors = list()
    for i in range(num_neighbors):
        cos_dist = cosine_distance(test_row, train[euclidean_distance_index[i]][:-1])
        if cos_dist <= 0.7:
            neighbors.append(None)
        else:
            neighbors.append(euclidean_distances[i][1])
    return neighbors


# Make a prediction with neighbors
def classification(mapping, train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    if prediction is not None:
        prediction = mapping[prediction]
    return prediction


# KNN Algorithm
def k_nearest_neighbors(label, train, test, num_neighbors):
    int_to_label, anchor_mapped_label = names_to_integers(label)

    new_shape = list(train.shape)
    new_shape[-1] += 1
    new_shape = tuple(new_shape)

    anchor_mega = np.empty(new_shape)
    for i in range(len(anchor_mega)):
        anchor_mega[i] = np.append(train[i], anchor_mapped_label[i])

    predictions = list()
    for row in test:
        output = classification(int_to_label, anchor_mega, row, num_neighbors)
        predictions.append(output)

    return predictions


def knn_prediction(anchor_label, anchor_embed, input_embed):
    predicted_ids = [k_nearest_neighbors(anchor_label, anchor_embed, embed, 1) for embed in input_embed]

    final_ids = []
    for ids in predicted_ids:
        collected_ids = np.array(ids)
        collected_ids = np.array_split(collected_ids, len(ids))
        collected_ids = [id.tolist() for id in collected_ids]
        final_ids.append(collected_ids)

    return final_ids


def clear_results(names, boxes, ids, name):
    for i in range(len(ids)):
        keep_index = [ind for ind in range(len(ids[i])) if ids[i][ind] != [None]]
        boxes[i] = [boxes[i][ind] for ind in keep_index]
        ids[i] = [ids[i][ind] for ind in keep_index]

    new_names = [names[i] for i in range(len(names)) if boxes[i] != []]
    new_boxes = list(filter(None, boxes))
    new_ids = list(filter(None, ids))

    df_new = pd.DataFrame({'filename': new_names, 'bboxes': new_boxes, 'ids': new_ids})
    if name is not None:
        find_index = [i for i in range(len(new_ids)) if [name] in new_ids[i]]
        df_new = df_new.iloc[find_index]

    df_new = df_new.reset_index(drop=True)

    return df_new


def face_detection(original_path, anchor_path, finding_name):
    """
    This function performs face detection in the given image dataset.

    Parameters
    ----------
    + original_path : str.
    The path to your input image dataset.

    + anchor_path : str.
    The path to your anchor image dataset.

    Return
    ----------
    + df : Pandas Dataframe.
    A dataframe contained the filenames for input images, as well as predicted bounding boxes.
    """

    input_name, input_img = read_input_images(original_path, purpose='input')
    anchor_label, anchor_img = read_anchor_images(anchor_path)

    mtcnn, infer_model = create_facenet_models()

    input_boxes, _, _ = get_bounding_box(mtcnn, input_img, 64)
    anchor_boxes, _, _ = get_bounding_box(mtcnn, anchor_img, 64)

    input_boxes = clipping(input_img, input_boxes)
    anchor_boxes = clipping(anchor_img, anchor_boxes)

    input_img, input_boxes, input_name = filter_images(input_name, input_img, input_boxes)
    anchor_img, anchor_boxes, anchor_label = filter_images(anchor_label, anchor_img, anchor_boxes)

    cropped_img_anchor = cropping_face(anchor_img, anchor_boxes, purpose='anchor')
    cropped_img_input = cropping_face(input_img, input_boxes, purpose='input')

    anchor_embed = vector_embedding(infer_model, cropped_img_anchor, purpose='anchor')
    input_embed = vector_embedding(infer_model, cropped_img_input, purpose='input')

    final_ids = knn_prediction(anchor_label, anchor_embed, input_embed)

    df = clear_results(input_name, input_boxes, final_ids, finding_name)

    return df, input_image #return input images để không phải đọc hình nhiều lần
