import numpy as np

class prediction(object):
    """
    Object that represent a generic detection obtained from an object detection system.
    Each prediction is associated with a bounding box 
    """
    def __init__(self,coordinates,class_id,confidence, class_name=None):
        """
        Create a prediction
        Args:
            coordinates: 4 float representing the coordinates of the predicted bounding box in realitve dimension [ymin,xmin,ymax,xmax].
            class_id: id of the predicted class.
            confidence: confidence associated with the prediction
            class_name: optional human readable name of the detected class
        """
        assert len(coordinates)==4 
        self.box = coordinates
        self.classId=int(class_id)
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
    
    def toArray(self):
        """
        Encode the detction in 6 floats
        """
        return [self.classId,self.confidence,self.box[0],self.box[1],self.box[2],self.box[3]]

    @classmethod
    def fromArray(cls,array):
        """
        Construct a prediction from an array of six floats
        """
        assert(len(array)==6)
        return cls(array[2:],array[0],array[1])

    @classmethod
    def fromMatrix(cls,matrix):
        """
        Construct a list of predictions from a matrix of float
        """
        result=[]
        for i in range(matrix.shape(0)):
            result.append(cls.fromArray(matrxi[i]))
        return result

    @staticmethod
    def toMatrix(predictions):
        """
        Serialize a bunch of predictions in a matrix of float
        """
        result = np.ndarray(shape=(len(predictions),6),dtype=float)
        for i,d in enumerate(predictions):
            result[i]=d.toArray()
        return result