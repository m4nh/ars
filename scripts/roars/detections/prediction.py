class prediction(object):
    """
    Object that represent a generic detection obtained from an object detection system.
    Each detection is associated with a bounding box 
    """
    def __init__(self,coordinates,class_id,confidence, class_name=None):
        """
        Create a detection
        Args:
            coordinates: 4 float representing the coordinates of the predicted bounding box in realitve dimension [ymin,xmin,ymax,xmax].
            class_id: id of the predicted class.
            confidence: confidence associated with the prediction
            class_name: optional human readable name of the detected class
        """
        assert len(coordinates)==4 
        self.box = coordinates
        self.classId=class_id
        self.confidence = confidence
        self.className=class_name
    
    def __str__(self):
        if self.className is None:
            return "Class: {}, confidence: {:.2f}, coordinates: {}".format(self.classId, self.confidence, self.box)
        else:
            return "Class: {}-{}, confidence: {:.2f}, coordinates: {}".format(self.classId, self.className, self.confidence, self.box)
    
    def __repr__(self):
        return self.__str__()

    def getClassName(self):
        if self.className is None:
            return str(self.classId)
        else:
            return self.className