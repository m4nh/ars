import prediction
import os
import tensorflow as tf
import numpy as np

class tensorflow_detector_wrapper(object):
    """
    Object to wrap a tensorflow object detcter trained using the object detection API.
    """
    def __init__(self,graph_path,label_map=None):
        """
        Setup tensorflow and all the inference graph, returns an object ready to be used for object detction
        Args:
            - graph_path: path to the pb file with the graph and weight definition
            - label_map: path to the pbtxt containing the label definition
        """
        #check file existence
        for p in [graph_path,label_map]:
            if not os.path.exists(p):
                raise Exception('Unable to find file: {}'.format(path))
        
        #setup inner fields
        self._graph = graph_path
        self._label_map = label_map
        self._setup_inference_graph()
        if self._label_map is None:
            self._labelsDictionary={}
        else:
            self._setup_label_map()
        self._sess = tf.Session(graph=self._detection_graph)

        #fetch tensorflow ops
        self._image_tensor = self._detection_graph.get_tensor_by_name('image_tensor:0')
        self._boxes = self._detection_graph.get_tensor_by_name('detection_boxes:0')
        self._scores = self._detection_graph.get_tensor_by_name('detection_scores:0')
        self._classes = self._detection_graph.get_tensor_by_name('detection_classes:0')
        self._num_detections = self._detection_graph.get_tensor_by_name('num_detections:0')

        #fake detection to setup tensorflow pipeline
        immy = np.zeros((224,224,3),dtype=np.uint8)
        self.detect(immy)

        print('Tensorflow detector succesfully created, ready to detect {} class'.format(len(self._labelsDictionary.keys())))

    def getClassDictionary(self):
        """
        Returns a dictionary with keys class_id and values human readable name of the classes
        """
        return self._labelsDictionary

    def _setup_inference_graph(self):
        self._detection_graph = tf.Graph()
        with self._detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self._graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
    
    def _setup_label_map(self):
        with open(self._label_map) as f_in:
            lines = f_in.readlines()
    
        current_id=-1
        self._labelsDictionary={}
        self._labelsDictionary[0]='Backgorund'
        for l in lines:
            if 'id' in l:
                current_id = int(l.split(':')[-1])
            if 'name' in l:
                class_name=l.split(':')[-1].strip().replace("'","")
                self._labelsDictionary[current_id]=class_name

    def _getClassName(self,id):
        if len(self._labelsDictionary.keys())>0:
            if id in self._labelsDictionary.keys():
                return self._labelsDictionary[id]
        return "object"

    
    def detect(self,image):
        """
        Perform object detction on image and returns the result as a list of detections.prediction
        Args:
            image: numpy array with shape [height,width,#channels] in RGB order
        Returns:
            list of detected object in detections.prediciton format
        """
        assert len(image.shape)==3 and (image.shape[2]==1 or image.shape[2]==3)

        #image preprocessing
        image = image.astype(np.uint8)
        image = np.expand_dims(image,axis=0)

        #detections
        (boxes, scores, classes, num_detections) = self._sess.run([self._boxes, self._scores, self._classes, self._num_detections],feed_dict={self._image_tensor: image})
        
        #remove extra dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        #convert to detections.prediction
        result=[None]*int(num_detections)
        for i in range(num_detections):
            result[i]=prediction.prediction(boxes[i],classes[i],scores[i],self._getClassName(classes[i]))
        return result