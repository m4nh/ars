from abc import ABCMeta, abstractmethod, abstractproperty
import os

#TODO: li abbiamo da qualche parte questi?
LABEL_FOLDER_NAME = 'labels'
LABEL_FILES_EXTENSION = 'txt'

IMAGE_FOLDER_NAME = 'images'
IMAGE_FILES_EXTENSION = 'jpg'

CLASS_FILE = 'class_list.txt'

class meta_trainer():
    """
    Meta class for a generic CNN trainer, it will have different implementation according to the framework used
    """
    __metaclass__=ABCMeta

    def __init__(self, **kwargs):
        self._ready=False
        self._setup_trainer(kwargs)

    #abstract stuff that should be implemented in the inheriting classes
    @abstractproperty
    def _default_batch_size(self):
        pass


    @abstractproperty
    def _default_max_iteration(self):
        pass
        

    @abstractmethod
    def _prepare_dataset(self):
        """Setup the dataset to be used during the training phase, return False if the dataset already exist in the detsination folder, True otherways"""
        pass

    @abstractmethod
    def _setup_training_parameters(self):
        """Create additional configuration file if needed"""
        pass

    @abstractmethod
    def _train(self):
        """Proper training"""
        pass

    @abstractmethod
    def _export(self):
        """Save the best model found"""
        pass


    #Common high order methods
    def _setup_trainer(self,args):
        """
        Check the args to see if everything is properly configured and save needed info in internal fields
        """
        print('='*50)
        print('=                 Checking Arguments             =')
        print('='*50)

        if 'input_folder' not in args:
            print('ERROR: Please specify an input directory')
            raise Exception('"input_folder" is missing')
        
        if 'detector_name' not in args:
            print('ERROR: "detector_name" not specified, this should not happen')
            raise Exception('Detector_name not specified')
        
        if 'batch_size' not in args:
            print('WARNING: "batch_size" not specified, using default {}'.format(self._default_batch_size))
            args['batch_size']=self._default_batch_size
        
        if 'max_iteration' not in args:
            print('WARNING: "max_iteration" not specified, using default {}'.format(self._default_max_iteration))
            args['max_iteration']=self._default_max_iteration
        

        #map args to objet fields
        self._input_folder = args['input_folder']
        self._detector_name = args['detector_name']
        self._batch_size = args['batch_size']
        self._max_iteration = args['max_iteration']

    def _check_ready(self):
        """
        If the trainer is not ready raise an exception
        """
        if not self._ready:
            print('ERROR: trainer not correctly configured, you are not supposed to end here!')
            raise Exception('trainer not correctly configured')

    @property
    def _class_map(self):
        """
        Read class list files and return a list containing class names
        """
        self._check_ready()

        if not hasattr(self,'_c_map'):
            if os.path.exists(os.path.join(self._input_folder,CLASS_FILE)):
                with open(os.path.join(self._input_folder,CLASS_FILE)) as f_in:
                    classes =[c.strip() for c in  f_in.readlines()]
            
                self._c_map = classes
            else:
                raise Exception('Unable to find {}'.format(CLASS_FILE))
        
        return self._c_map


    def train_detector(self,output_folder):
        self._check_ready()
        
        self._output_folder = output_folder
        
        #setup data
        self._prepare_dataset()

        #setup config parameters
        self._setup_training_parameters()

        #lunch training
        result=self._train()

        if result==0:
            #save final model
            self._export()
            print('All DONE!')
        else:
            raise Exception('Train Failed')