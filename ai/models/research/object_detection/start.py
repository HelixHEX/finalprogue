import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import zerosms

from pyfirmata import Arduino, util
import serial
import serial.tools.list_ports

from twilio.rest import Client

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import time

import cv2
cap = cv2.VideoCapture(0)

#
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('output4.mp4',fourcc, 20.0, (1280,720))
#

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
MODEL_NAME = 'gun_graph'


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1


# ## Download Model

# In[5]:




# ## Load a (frozen) Tensorflow model into memory.

# In[6]:
count = 0

number_of_threats_detected = 0;

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 9) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)




# In[10]:
# def aruduino:
#     board = Arduino("COM3")
#     loopTimes = input(5)
#     print("Blinking " + loopTimes + " times.")
#
#     for x in range(int(loopTimes)):
#         board.digital[13].write(1)
#         time.sleep(0.2)
#         board.digital[13].write(0)
#         time.sleep(0.2)
def arduino():
    ser = serial.Serial('/dev/cu.usbmodemFA131', 9800, timeout=1)
    loopTime = 1
    time.sleep(1)

    for x in range(int(loopTime)):
        ser.writelines(b'H')
        time.sleep(0.5)
        ser.writelines(b'L')
        time.sleep(0.5)
    ser.close()
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    gunfound = tf.Variable(0, name='gunfound')
    assert gunfound.graph is detection_graph
    tf.global_variables_initializer().run()
    while True:
      ret, image_np = cap.read()
      if ret == True:
       # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
       image_np_expanded = np.expand_dims(image_np, axis=0)
       image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
       boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

       scores = detection_graph.get_tensor_by_name('detection_scores:0')
       classes = detection_graph.get_tensor_by_name('detection_classes:0')
       num_detections = detection_graph.get_tensor_by_name('num_detections:0')
       gunfound = detection_graph.get_tensor_by_name('gunfound:0')
       # Actual detection.
       (boxes, scores, classes, num_detections, gunfound) = sess.run(
          [boxes, scores, classes, num_detections, gunfound],
          feed_dict={image_tensor: image_np_expanded})
       predic = scores[0]
       print(predic[0])
       if (predic > 0.50).any():
           print('gun found')
           count+=1
       elif (predic > 0.25).any() and (predic < 0.50).any():
           print('possible threat')
       else:
           print('no threat')
       if count == 10:
           account_sid = 'AC4c1c280aaaca67c35eee56cdf73f7a66'
           auth_token = 'e25f674e89bbe95fba2842b6e741aef2'
           client = Client(account_sid, auth_token)
           message = client.messages \
              .create(
                 body='There is gun in the building',
                 from_='+1938444-8619',
                 to='+17325956989'
              )
           print(message.sid)
           ccount = 0
           loopTime = 5
           number_of_threats_detected+=1
           create_file = open('number-of-threats.txt', 'w+')
           create_file.write(str("Number of threats stopped: %d\r\n" % (number_of_threats_detected)))
           arduino()
           arduino()
       # ans = np.amax(scores)
       # print(ans)

       vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)



       # out.write(image_np)
       cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
       if cv2.waitKey(15) & 0xFF == ord('q'):

         break
      else:
        break

# Release everything if job is finished
cap.release()
# out.release()
cv2.destroyAllWindows()
