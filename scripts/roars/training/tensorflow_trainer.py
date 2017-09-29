from __future__ import print_function
import meta_trainer
import os
import glob
import io
import hashlib
from PIL import Image
import tensorflow as tf
from abc import ABCMeta, abstractmethod
import subprocess

DATASET_TMP_FOLDER = '/tmp/'
DATASET_TMP_NAME = 'Train.tfrecord'
DATASET_CLASS_MAP = 'classes.pbtxt'
CONFIG_FOLDER = os.path.join(os.path.abspath(os.path.join(__file__,os.pardir)),'sample_tensorflow_config')
DEFAULT_CONFIG_NAME = 'temp_detector.config'
TRAIN_SCRIPT = os.path.join(os.path.abspath(os.path.join(__file__,os.pardir)),'train_script/tensorflow_train.sh')
EXPORT_SCRIPT = os.path.join(os.path.abspath(os.path.join(__file__,os.pardir)),'train_script/tensorflow_export.sh')

################################################################
##                          UTILS                             ##
################################################################
def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def get_tf_record(image_path, label_path, class_names):
    """
    Utility function to encode image and label in a single tfrecord
    """

    #load image
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_img = fid.read()

    encoded_img_io = io.BytesIO(encoded_img)
    image = Image.open(encoded_img_io)
    key = hashlib.sha256(encoded_img).hexdigest()

    width, height = image.size

    #read annotation
    with open(label_path, 'r') as li:
        annotations = li.readlines()

    #convert annotations to tensorflow format
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for a in annotations:
        c_id, x, y, w, h = a.strip().split(' ')
        c_id=int(c_id)
        x=float(x)
        y=float(y)
        w=float(w)
        h=float(h)

        xmin.append(float(x - (w / 2)))
        ymin.append(float(y - (h / 2)))
        xmax.append(float(x + (w / 2)))
        ymax.append(float(y + (h / 2)))
        #class 0 is for background?
        classes.append(c_id+1)
        classes_text.append(class_names[c_id].encode('utf8'))
        #????????????????????????????????????
        truncated.append(0)
        poses.append(''.encode('utf8'))
        difficult_obj.append(int(False))

    #create tfrecords
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(image_path.encode('utf8')),
        'image/source_id': bytes_feature(image_path.encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_img),
        'image/format': bytes_feature(image_path[-3:].encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
        'image/object/difficult': int64_list_feature(difficult_obj),
        'image/object/truncated': int64_list_feature(truncated),
        'image/object/view': bytes_list_feature(poses),
    }))

    return example


################################################################
##                      ABSTACT MAIN CLASS                    ##
################################################################


class tensorflow_trainer(meta_trainer.meta_trainer):

    def __init__(self,**kwargs):
        super(tensorflow_trainer,self).__init__(**kwargs)

    def _setup_trainer(self,args):
        #call parent class setup
        super(tensorflow_trainer,self)._setup_trainer(args)

        #check class specific args
        if 'tf_models_base_dir' not in args:
            print('ERROR: no "tf_models_base_dir" specified, unable to use tensorflow!! ')
            raise Exception('tf_models_base_dir is missing')
        
        if 'starting_weight' not in args:
            print('WARNING: "starting_weight" not specified, starting from random initialization')
            args['starting_weight']=None
        
        if 'temp_dataset_dir' not in args:
            print('WARNING: no "temp_dataset_dir" using default '+DATASET_TMP_FOLDER)
            args['temp_dataset_dir']=DATASET_TMP_FOLDER
        

        #map args to object fields and set ready to True
        self._tf_models_base_dir = args['tf_models_base_dir']
        self._starting_weight = args['starting_weight']
        self._temp_dataset_dir = args['temp_dataset_dir']
        self._ready=True
    
    def _prepare_dataset(self):
        print('='*50)
        print('=          Starting Dataset Creation             =')
        print('='*50)
        
        #create image/label lists
        label_list = sorted(glob.glob(os.path.join(self._input_folder,meta_trainer.LABEL_FOLDER_NAME,'*.'+meta_trainer.LABEL_FILES_EXTENSION)))
        image_list = sorted(glob.glob(os.path.join(self._input_folder,meta_trainer.IMAGE_FOLDER_NAME,'*.'+meta_trainer.IMAGE_FILES_EXTENSION)))

        #check for same size
        assert(len(label_list)==len(image_list))
        print('Found a dataset with {} sampless'.format(len(image_list)))

        #output files path
        self._training_set_path = os.path.join(DATASET_TMP_FOLDER,DATASET_TMP_NAME)
        self._label_map_path = os.path.join(DATASET_TMP_FOLDER,DATASET_CLASS_MAP)

        #check if the dataset has already been created
        if os.path.exists(self._training_set_path) and os.path.exists(self._label_map_path):
            print('Dataset already existing, skipping recreation')
            return False
        else:
            print('Going to save temporary dataset in: {}'.format(self._training_set_path))
        
        writer = tf.python_io.TFRecordWriter(self._training_set_path)

        #start tfrecord creation
        for idx,(img,lbl) in enumerate(zip(image_list,label_list)):
            #get a single tfrecord sample
            example = get_tf_record(img,lbl,self._class_map)

            #serialize it
            writer.write(example.SerializeToString())
            
            print('{}/{}'.format(idx,len(image_list)),end='\r')
        print('Dataset Conversion Complete')

        #save class_map.pbtxt
        with open(self._label_map_path, 'w+') as f_out:
            proto_string="\nitem{{\n\tid: {}\n\tname: '{}' \n }}\n"
            for i,c in enumerate(self._class_map):
                f_out.write(proto_string.format(i+1,c))

        print('Label Map Saved')
        print('Dataset Ready')
        print('='*50)

    
    def _setup_training_parameters(self):
        """
        Create the proper configuration file for the detction
        """
        print('='*50)
        print('=            Creating configuration file         =')
        print('='*50)
        config_path = os.path.join(CONFIG_FOLDER,self._detector_name+'.config')
        
        with open(config_path,'r') as f_in:
            lines = f_in.readlines()
        

        #TODO: rewrite this method better
        new_lines=[]
        for l in lines:
            skip=False
            if 'fine_tune_checkpoint' in l:
                if self._starting_weight is None:
                    skip=True
                else:
                    l=l.replace('PATH_TO_BE_CONFIGURED',self._starting_weight)
            elif 'from_detection_checkpoint' in l and self._starting_weight is None:
                skip=True
            elif 'num_classes' in l:
                l=l.replace('??',str(len(self._class_map)))
            elif 'train_input_reader' in l:
                in_training=True
            elif 'eval_input_reader' in l:
                in_validation=True
            elif 'input_path' in l:
                if in_training:
                    l = l.replace('PATH_TO_BE_CONFIGURED',self._training_set_path)
                    in_training=False
                elif in_validation:
                    l = l.replace('PATH_TO_BE_CONFIGURED', self._training_set_path)
                    in_validation=False
            elif 'label_map_path' in l:
                l = l.replace('PATH_TO_BE_CONFIGURED', self._label_map_path)
            elif 'num_step' in l:
                l = l.replace('??',str(self._max_iteration))
            elif 'batch_size' in l:
                l = l.replace('??',str(self._batch_size))
            
            if not skip:
                new_lines.append(l)

        self._config_file = os.path.join(DATASET_TMP_FOLDER,DEFAULT_CONFIG_NAME)
        with open(self._config_file,'w+') as f_out:
            f_out.writelines(new_lines)

        print('DONE')
        print('='*50)

    def _train(self):
        """
        Launch training of the object detector
        """
        print('='*50)
        print('=                Starting Training               =')
        print('='*50)

        #setup enviromentall variables
        os.environ['TF_MODEL_DIR']=os.path.join(self._tf_models_base_dir,'research')
        os.environ['CONFIG_PATH']=self._config_file
        os.environ['OUTPUT_FOLDER']=self._output_folder

        #call tensorflow_train.sh
        print('Intermediate results and logs will be saved in {}'.format(self._output_folder))
        print(TRAIN_SCRIPT)
        exit_status=subprocess.call(['bash',TRAIN_SCRIPT])
        
        #training done
        print('Process complete with exit status: {}'.format(exit_status))
        return exit_status
        
    
    def _export(self):
        """
        Export trained model as freezed graph
        """
        print('='*50)
        print('=     Exporting trained graph     =')
        print('='*50)

        #read checkpoint gile
        with open(os.path.join(self._output_folder,'checkpoint')) as f_in:
            self._ckpt_path=f_in.readline().split(':')[1].replace('"','').strip()
        self._export_folder = os.path.join(self._output_folder,'exported')

        #setup enviromentall variables
        os.environ['TF_MODEL_DIR']=os.path.join(self._tf_models_base_dir,'research')
        os.environ['CONFIG_PATH']=self._config_file
        os.environ['OUT_FOLDER']=self._export_folder
        os.envirom['CHECKPOINT_FILE']=self._ckpt_path

        #call tesnorflow_export.sh
        print('Resulting detector will be saved in {}'.format(self._output_folder))
        exit_status=subprocess.call(['bash',EXPORT_SCRIPT])

        #export done
        print('Frozen graph exported')
        return exit_status


################################################################
##                   IMPLEMENTATION CLASSES                   ##
################################################################

class rcnn_based_trainer(tensorflow_trainer):
    def __init__(self,**kwargs):
        super(rcnn_based_trainer,self).__init__(**kwargs)

    @property
    def _default_batch_size(self):
        return 1
    
    @property
    def _default_max_iteration(self):
        return 100000

class ssd_based_trainer(tensorflow_trainer):
    def __init__(self,**kwargs):
        super(ssd_based_trainer,self).__init__(**kwargs)

    @property
    def _default_batch_size(self):
        return 24
    
    @property
    def _default_max_iteration(self):
        return 100000

class rfcn_based_trainer(tensorflow_trainer):
    def __init__(self,**kwargs):
        super(rfcn_based_trainer,self).__init__(**kwargs)

    @property
    def _default_batch_size(self):
        return 1

    @property
    def _default_max_iteration(self):
        return 100000