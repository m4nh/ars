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
        self.ymin = coordinates[0]
        self.xmin = coordinates[1]
        self.ymax = coordinates[2]
        self.xmax = coordinates[3]
        self.classId=int(class_id)
        self.confidence = confidence
        self.className=class_name
    
    def __str__(self):
        if self.className is None:
            return "Class: {}, confidence: {:.2f}, coordinates: {}".format(self.classId, self.confidence, self.box())
        else:
            return "Class: {}-{}, confidence: {:.2f}, coordinates: {}".format(self.classId, self.className, self.confidence, self.box())
    
    def __repr__(self):
        return self.__str__()

    def getClassName(self):
        if self.className is None:
            return str(self.classId)
        else:
            return self.className
    
    def box(self,center=False):
        """
        Get coordinates of the bounding box as a 4 float array either with [ymin,xmin,ymax,xmax] or if center=True [x_center,y_center,w,h]
        """
        if not center:
            return [self.ymin,self.xmin,self.ymax,self.xmax]
        else:
            return [(self.xmax-self.xmin)/2,(self.ymax-self.ymin)/2,self.xmax-self.xmin,self.ymax-self.ymin]

    def getArea(self):
        w=self.xmax-self.xmin
        h=self.ymax-self.ymin
        return w*h

    def intersect(self,other):
        return (self.xmin <= other.xmax and self.xmax >= other.xmin and self.ymin <= other.ymax and self.ymax >= other.ymin)

    def intersectionArea(self,other):
        if not self.intersect(other):
            return 0
        else:
            i_xmin = max(self.xmin,other.xmin)
            i_xmax = min(self.xmax,other.xmax)
            i_ymin = max(self.ymin,other.ymin)
            i_ymax = min(self.ymax,other.ymax)
            i_w = i_xmax-i_xmin
            i_h = i_ymax-i_ymin
            return i_w*i_h

    def toArray(self):
        """
        Encode the detection in 6 floats
        """
        return [self.classId,self.ymin,self.xmin,self.ymax,self.xmax,self.confidence]

    @classmethod
    def fromArray(cls,array,centers=False):
        """
        Construct a prediction from an array of six floats.
        The six floats encode [class_id, ymin, xmin, ymax, xmax, confidence] if centers=False, else [class_id, x_center, y_center, width, height, confidence]
        """
        assert(len(array)==6)
        if not centers:
            return cls(array[1:5],array[0],array[-1])
        else:
            x_c,y_c,w,h=array[1:5]
            xmin=x_c-(w/2)
            xmax=x_c+(w/2)
            ymin=y_c-(h/2)
            ymax=y_c+(h/2)
            return cls([ymin,xmin,ymax,xmax],array[0],array[-1])

    @classmethod
    def fromMatrix(cls,matrix):
        """
        Construct a list of predictions from a matrix of float
        """
        result=[]
        for i in range(matrix.shape(0)):
            result.append(cls.fromArray(matrix[i]))
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
