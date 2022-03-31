from facenet_pytorch import MTCNN
import torch
import numpy as np
import math
from tqdm.notebook import tqdm


def create_mtcnn_model():
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

  mtcnn = MTCNN(
      image_size=160, margin=0, min_face_size=10,
      thresholds=[0.6, 0.7, 0.7], post_process=False,
      device=device, selection_method='largest_over_threshold'
  )

  if mtcnn is not None:
      print('Successfully create a MTCNN model base')
  else:
      print('Fail to create a MTCNN model base')

  return mtcnn


def get_bounding_box(mtcnn_model, frames, batch_size = 32):
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

  for batch_file in tqdm(frames):
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
