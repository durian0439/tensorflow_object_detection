import pathlib
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#버전 호환을 위한 코드
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile

#Loader
def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))

  return model

#함수 테스트: 유닛테스트

#Detection 
#Load an object detection model:
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)


#Loading label map

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'C:/Users/bear0/Documents/GitHub/Tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


#테스트 이미지 준비 : 왼쪽 폴더 열어보면, models 폴더 안에 테스트 이미지 이미 있다. 이것 활용한다.
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = pathlib.Path('data/models/images')
# TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))




def run_inference_for_single_image(model, image):
    #np.array로 바꿔주는 것
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    #모델이 tensor로 들어가야하기에 바꿔줌
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = tf.convert_to_tensor(output_dict['detection_masks'], dtype=tf.float32)
        output_dict['detection_boxes'] = tf.convert_to_tensor(output_dict['detection_boxes'], dtype=tf.float32)
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    output_dict['detection_masks'], output_dict['detection_boxes'],
                    image.shape[0], image.shape[1])  
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

# #함수테스트
# image_np = cv2.imread('data/models/images/image1.jpg')
# output_dict = run_inference_for_single_image(detection_model, image_np)
# print(output_dict)

def show_inference(model, image_np):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.

    # image_np = np.array(Image.open(image_path)
    
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.array(output_dict['detection_boxes']),
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed',None),
        use_normalized_coordinates=True,
        line_thickness=8)

    cv2.imshow('result', image_np)



#이미지 경로에 있는 이미지 실행
# for image_path in TEST_IMAGE_PATHS:
#     show_inference(detection_model, image_path)

#비디오 실행 코드
cap = cv2.VideoCapture('data/models/video/dashcam2.mp4')

#카메라의 영상 실행 코드
# cap = cv2.VideoCapture(0)

if cap.isOpened() == False:
    print('Error opening video stream or file')

else:
    #반복문이 필요한 이유? 비디오는 여러 사진으로 구성되어있음.
    while cap.isOpened():
        #사진을 1장씩 가져와서 
        ret, frame = cap.read()
        #제대로 가져왔으면 화면 표시.
        if ret == True:
            # cv2.imshow("Frame", frame)
            #추론하고 화면에 보여주는 코드
            show_inference(detection_model, frame)

            #키보드에서 esc를 누르면 exit하라
            if cv2.waitKey(25) & 0xFF ==27:
                break
        else:
            break

cap.release()
cv2.destroyAllWindow()

# cv2.waitKey(0)
# cv2.destroyALLWindow()
