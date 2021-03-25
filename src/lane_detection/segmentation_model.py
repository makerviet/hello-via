import tensorflow as tf
import cv2
import os
import numpy as np

from ..utils.image_stream import image_streamer

class LaneLineSegmentationModel:

    def __init__(self, model_path, use_gpu=False):

        self.graph = tf.Graph()
        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
            self.sess = tf.InteractiveSession(graph = self.graph)

        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read()) 

        # Print out nodes
        # print([n.op + "=>" + n.name for n in graph_def.node])

        self.input = tf.placeholder(np.float32, shape = [None, 144, 144, 3], name='input_1')
        tf.import_graph_def(graph_def, {'input_1': self.input})


    def predict(self, origin_img):
        """Lane line segmentation from bgr image

        Args:
            origin_img: Original image

        Returns:
            np.array: Lane line mask. Value for each pixel 0->255.
                See lane_segmentation/model_output in Visualizer
        """

        img = cv2.resize(origin_img,(144,144))
        img = (img[...,::-1].astype(np.float32)) / 255.0
        img = np.reshape(img, (1, 144, 144, 3))
        output_tensor = self.graph.get_tensor_by_name("import/softmax/truediv:0")
        output = self.sess.run(output_tensor, feed_dict={self.input: img})

        # output0 = (output[0][:, :, 0] * 255).astype(np.uint8)
        # output1 = (output[0][:, :, 1] * 255).astype(np.uint8)
        output2 = (output[0][:, :, 2] * 255).astype(np.uint8)

        # image_streamer.set_image("lane_segmentation/output0", output0)
        # image_streamer.set_image("lane_segmentation/output1", output1)
        image_streamer.set_image("lane_segmentation/model_output", output2)

        output2 = cv2.resize(output2, (origin_img.shape[1], origin_img.shape[0]))
        return output2