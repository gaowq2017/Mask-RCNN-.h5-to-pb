#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings

import keras.backend as K
import tensorflow as tf

warnings.filterwarnings('ignore', category=FutureWarning)
# suppress warning and error message tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Root directory of the project
ROOT_DIR = 'F:\MaskRcnn\My_Mask_Rcnn_5class'
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn.config import Config
# from samples.coco import coco

K.clear_session()
K.set_learning_phase(0)

##############################################################################
# Load model
##############################################################################


# Model Directory
# MODEL_DIR = os.path.join(os.path.dirname(__file__), "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "samples/coco/mask_rcnn_coco.h5")
DEFAULT_WEIGHTS = os.path.join(ROOT_DIR, "mask_rcnn_diangan_0030.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(DEFAULT_WEIGHTS):
    utils.download_trained_weights(DEFAULT_WEIGHTS)


##############################################################################
# Load configuration
##############################################################################

class DianganConfig(Config):
    """
    Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "diangan"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # COCO has 80 classes

'''
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
'''

##############################################################################
# Save entire model function
##############################################################################

def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_"):
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))
    sess = K.get_session()
    init_graph = sess.graph.as_graph_def()
    main_graph = tf._api.v1.graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    with tf.gfile.GFile(os.path.join(output_dir, model_name), "wb") as filemodel:
        filemodel.write(main_graph.SerializeToString())
    print("pb model: ", {os.path.join(output_dir, model_name)})


if __name__ == "__main__":
    config = DianganConfig()
    config.display()
    # Create model in inference mode
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Set path to model weights
    weights_path = DEFAULT_WEIGHTS  # model.find_last()
    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    model.keras_model.summary()

    # make folder for full model
    model_dir = os.path.join(ROOT_DIR, "Model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # save h5 full model
    name_model = os.path.join(model_dir, "mask_rcnn_landing.h5")
    if not os.path.exists(name_model):
        model.keras_model.save(name_model)
        print("save model: ", name_model)

    # export pb model
    pb_name_model = "mask_rcnn_landing.pb"
    h5_to_pb(model.keras_model, output_dir=model_dir, model_name=pb_name_model)
    K.clear_session()
    sys.exit()