from roars import training

args = {}
args['input_folder']='/home/alessio/Desktop/dataset_fruit/generic_scene'
args['detector_name']='ssd_inception_v2'
args['batch_size']=2
args['max_iteration']=10
args['tf_models_base_dir']='/home/alessio/sources/models'


trainer = training.get_trainer(args)

trainer.train_detector('/home/alessio/Desktop/trained_ssd')