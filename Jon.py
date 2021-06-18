import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
from google.colab.patches import cv2_imshow
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ImageOps

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.15)
# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = '/content/trained_model/exported_models/my_model/saved_model'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '/content/trained_model/annotations/label_map.pbtxt'
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
# Enable GPU dynamic memory allocation
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)


value = st.sidebar.radio('Radio', ["Main App", "About"])
if value == "Main App":
    st.title("Ageing sign detection")

    st.text("AI and deep learning assisted.")

    image = st.file_uploader('Upload an image here...', type=["jpg"])
    if image is None:
        st.text("Please upload a jpg")

    else:
        image = Image.open(image)
        image = np.asarray(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_with_detections = image.copy()

        # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=500,
            min_score_thresh=0.15,
            agnostic_mode=False)

        # DISPLAYS OUTPUT IMAGE
        # st.image(image_with_detections, caption="Image with detected signs.", width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
        # CLOSES WINDOW ONCE KEY IS PRESSED

    if st.button("Detect..") and image is not None:

        st.image(image_with_detections, caption="Image with detected signs.", width=None, use_column_width=None,
                 clamp=False, channels='RGB', output_format='auto')
        st.success("Thank you for trying our model..")
    else:
        st.error("Please upload an image first")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")

    if st.button("Exit"):
        st.stop()
elif value == "About":
    st.title("About")
    st.text("Author : Verzeo Batch-16 Team (April-May-2021)")
    st.text(" ")
    st.text("About us..")
    st.text("""We are a team of enthusiastic programmers trying make world a 
          better place.""")
    st.text("Hope you enjoy our app :-)")
else:
    pass
