from roars.training import meta_trainer
from roars.training import tensorflow_trainer
from functools import partial


#model name --> constructor
TRAINER_FACTORY = {
    'faster_rcnn_inception_v2_atrous': tensorflow_trainer.rcnn_based_trainer,
    'faster_rcnn_resnet101':tensorflow_trainer.rcnn_based_trainer,
    'faster_rcnn_resnet50':tensorflow_trainer.rcnn_based_trainer,
    'faster_rcnn_resnet152':tensorflow_trainer.rcnn_based_trainer,
    'rfcn_resnet101':tensorflow_trainer.rfcn_based_trainer,
    'ssd_inception_v2':tensorflow_trainer.ssd_based_trainer,
    'ssd_mobilenet_v1':tensorflow_trainer.ssd_based_trainer
}

def get_trainer(params):
    if 'detector_name' not in params:
        raise Exception('detector_name missing')
    elif params['detector_name'] not in TRAINER_FACTORY.keys():
        raise Exception('Unrecognized trainer {}'.format(params['detector_name']))
    else:
        trainer_maker = TRAINER_FACTORY[params['detector_name']]
        return trainer_maker(**params)