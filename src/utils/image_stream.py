import cv2
import threading

class ImageStream():
    """Image stream"""

    def __init__(self):
        self.image = None
        self.mutex = threading.Lock()

    def set_image(self, image):
        self.mutex.acquire()
        self.image = image.copy()
        self.mutex.release()

    def get_image(self):
        if self.image is None:
            return None
        self.mutex.acquire()
        image = self.image.copy()
        self.mutex.release()
        return image


class ImageStreamManager():
    """Image stream manager. This class got the idea from ROS topics"""

    def __init__(self):
        self.image_streams = {}
        self.placeholder = cv2.imread("data/placeholder.png")
        self.current_topic = ""

    def set_current_topic(self, topic):
        if topic in self.get_topics():
            self.current_topic = topic
            return True, ""
        else:
            return False, "Topic not found: {}".format(topic)

    def get_current_topic(self):
        topics = self.get_topics()
        if self.current_topic in self.get_topics():
            return self.current_topic
        elif len(topics) > 0:
            self.current_topic = topics[0]
            return self.current_topic
        else:
            self.current_topic = ""
            return ""

    def create_stream(self, topic):
        self.image_streams[topic] = ImageStream()

    def set_image(self, topic, image):
        if topic not in self.image_streams.keys():
            self.image_streams[topic] = ImageStream()
        self.image_streams[topic].set_image(image)

    def get_image(self, topic=None):
        if topic is None:
            topic = self.get_current_topic()
        if topic in self.image_streams.keys():
            return self.image_streams[topic].get_image()
        else:
            return self.placeholder

    def get_topics(self):
        return list(self.image_streams.keys())

image_streamer = ImageStreamManager()