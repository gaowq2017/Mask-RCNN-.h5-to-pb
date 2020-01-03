import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format

def convert_pb_to_pbtxt(filename):
    with gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, './', 'protobuf.pbtxt', as_text=True)
    return

filepath = 'F:\MaskRcnn\My_Mask_Rcnn_5class\Model\mask_rcnn_landing.pb'
convert_pb_to_pbtxt(filepath)
