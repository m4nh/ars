
# -*- encoding: utf-8 -*-

import PyKDL
import numpy as np
import json
import os
import random
import shutil
import roars.geometry.transformations as transformations
import roars.vision.colors as colors
from roars.vision.augmentereality import *


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'toJSON'):
            data = obj.toJSON()
            data['_type'] = obj.__class__.__name__
            return data
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            data = obj.__dict__
            data['_type'] = obj.__class__.__name__
            return data

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################


class CustomJSONDecoder(object):

    @staticmethod
    def decode(s):
        if '_type' in s:
            tp = s['_type']
            cl = eval(tp)
            if hasattr(cl, 'fromJSON'):
                return cl.fromJSON(s)
            else:
                inst = cl()
                inst.__dict__ = s
                return inst
        return s

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################


class JSONHelper(object):

    @staticmethod
    def loadFromFile(filename):
        try:
            with open(filename, 'r') as handle:
                data = json.load(handle, object_hook=CustomJSONDecoder.decode)
                return data
        except:
            print("JSONHelper:  Error loading '{}'".format(filename))
            return {}

    @staticmethod
    def saveToFile(filename, data):
        try:
            with open(filename, 'w') as handle:
                handle.write(json.dumps(
                    data,
                    indent=4,
                    cls=CustomJSONEncoder
                ))
                return True
        except:
            print("JSONHelper: Error saving '{}'".format(filename))
            return False

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################


class TrainingCamera(object):

    def __init__(self, configuration_file=''):
        self.configuration_file = configuration_file
        self.width = 0
        self.height = 0
        self.fx = 0
        self.fy = 0
        self.cx = 0
        self.cy = 0
        self.k1 = 0
        self.k2 = 0
        self.p1 = 0
        self.p2 = 0
        self.camera_matrix = np.array([])
        self.camera_matrix_inv = np.array([])
        self.distortion_coefficients = np.array([])

        if os.path.exists(self.configuration_file):
            raw = np.loadtxt(self.configuration_file)
            self.width = raw[0]
            self.height = raw[1]
            self.fx = raw[2]
            self.fy = raw[3]
            self.cx = raw[4]
            self.cy = raw[5]
            self.k1 = raw[6]
            self.k2 = raw[7]
            self.p1 = raw[8]
            self.p2 = raw[9]
            self.buildCameraMatrix()

    def buildCameraMatrix(self):
        self.camera_matrix = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.distortion_coefficients = np.array([
            self.k1, self.k2, self.p1, self.p2
        ])
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

    def getCameraFile(self):
        return self.configuration_file

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################


class TrainingScene(object):
    DEFAULT_CAMERA_POSE_NAME = 'camera_transform.txt'
    DEFAULT_CAMERA_PARAMS_NAME = 'camera_params.txt'

    def __init__(self, scene_path='#', image_topic_name='aaa', robot_pose_name='#', camera_intrisics_file='', camera_extrinsics_file=''):
        self.classes = {}
        self.scene_path = scene_path
        self.image_topic_name = image_topic_name
        self.robot_pose_name = robot_pose_name
        self.image_filenames_lists = []
        self.robot_to_camera_pose = [0, 0, 0, 0, 0, 0, 1]
        self.camera_intrisics_file = camera_intrisics_file
        self.camera_extrinsics_file = camera_extrinsics_file
        self.camera_params = {
            "camera_matrix": [],
            "camera_matrix_inv": [],
            "distortions": []
        }
        self.robot_poses = []
        self.initialized = False

    def initialize(self):
        ''' Initializes the Scene searching for Images and camera poses '''

        #⬢⬢⬢⬢⬢➤ Checks for Scene path
        if os.path.exists(self.scene_path):
            images_path = os.path.join(self.scene_path, self.image_topic_name)
        else:
            print("Scene path doesn't exist: {}".format(self.scene_path))
            return

        #⬢⬢⬢⬢⬢➤ Checks for Images folder path
        if os.path.exists(images_path):
            # Lists all Images in scene
            files = []
            for (dirpath, dirnames, filenames) in os.walk(images_path):
                files.extend(filenames)
                break
        else:
            print("Images path doesn't exist: {}".format(images_path))
            return

        #⬢⬢⬢⬢⬢➤ Checks for camera params
        if self.camera_intrisics_file == '':
            camera_param_path = os.path.join(
                self.scene_path,
                TrainingScene.DEFAULT_CAMERA_PARAMS_NAME
            )
        else:
            camera_param_path = self.camera_intrisics_file
        if os.path.exists(camera_param_path):
            # Load Camera Params
            try:
                self.camera_params = TrainingCamera(
                    configuration_file=camera_param_path)

            except Exception, e:
                print(e)
                return
        else:
            print("Camera params path doesn't exist: {}".format(camera_param_path))
            return

        #⬢⬢⬢⬢⬢➤ Checks for camera poses path
        if self.camera_extrinsics_file == '':
            camera_pose_path = os.path.join(
                self.scene_path,
                TrainingScene.DEFAULT_CAMERA_POSE_NAME
            )
        else:
            camera_pose_path = self.camera_extrinsics_file
        if os.path.exists(camera_pose_path):
            # Load Camera Pose Frame
            try:
                self.robot_to_camera_pose = np.loadtxt(camera_pose_path)
            except Exception, e:
                print(e)
                return
        else:
            print("Camera transform path doesn't exist: {}".format(camera_pose_path))
            return

        #⬢⬢⬢⬢⬢➤ Stores images paths A-Z
        for f in sorted(files):
            full_path = os.path.join(
                self.scene_path, self.image_topic_name, f)
            self.image_filenames_lists.append(full_path)

        #⬢⬢⬢⬢⬢➤ Checks for Robot poses path
        robot_pose_path = os.path.join(
            self.scene_path,
            self.robot_pose_name if '.txt' in self.robot_pose_name else self.robot_pose_name + ".txt"
        )
        if os.path.exists(robot_pose_path):
            robot_poses = np.loadtxt(robot_pose_path)
            for p in robot_poses:
                self.robot_poses.append(p.tolist())
        else:
            print("Robot poses path doesn't exist: {}".format(robot_pose_path))
            return

        #⬢⬢⬢⬢⬢➤ Checks for arrays consistency
        if len(self.robot_poses) == len(self.image_filenames_lists):
            self.initialized = True
            return
        else:
            print(
                "Robot poses size mismatches with Images number {}/{}".format(
                    len(self.robot_poses),
                    len(self.image_filenames_lists)
                )
            )
        return

    def getName(self):
        return os.path.basename(self.scene_path)

    def clearClasses(self):
        self.classes = {}

    def classesNumber(self):
        return len(self.classes)

    def getTrainingClass(self, index):
        if index in self.classes:
            return self.classes[index]
        if str(index) in self.classes:
            return self.classes[str(index)]

    def getAllInstances(self):
        instances = []
        for _, cl in self.classes.iteritems():
            instances.extend(cl.instances)
        return instances

    def getAllFrames(self):
        frames = []
        for i in range(0, self.size()):
            frames.append(TrainingFrame(scene=self, internal_index=i))
        return frames

    def pickTrainingFrame(self, index=-1):
        if index < 0:
            index = random.randint(0, self.size() - 1)
        index = index % self.size()
        return TrainingFrame(self, index)

    def getImagePath(self, index):
        index = index % self.size()
        return self.image_filenames_lists[index]

    def getCameraPose(self, index):
        index = index % self.size()
        robot_pose = self.robot_poses[index]
        camera_pose_frame = transformations.KDLFromArray(
            self.robot_to_camera_pose,
            fmt='XYZQ'
        )
        robot_pose_frame = transformations.KDLFromArray(
            robot_pose,
            fmt='XYZQ'
        )
        camera_pose_frame = robot_pose_frame * camera_pose_frame
        return camera_pose_frame

    def isImagesConsistens(self):
        return len(self.robot_poses) == len(self.image_filenames_lists)

    def isValid(self):
        return self.initialized and self.isImagesConsistens()

    def size(self):
        return len(self.image_filenames_lists)

    def save(self, filename):
        scene_js = json.dumps(
            self,
            cls=CustomJSONEncoder,
            sort_keys=True, indent=4
        )
        with open(filename, "w") as outfile:
            outfile.write(scene_js)

    @staticmethod
    def loadFromFile(filename):
        f = open(filename)
        sc = json.loads(f.read(), object_hook=CustomJSONDecoder.decode)
        return sc

    def getTrainingClassByLabel(self, label):
        if label in self.classes:
            return self.classes[label]
        return None

    def createTrainingClass(self, label, name=""):
        new_class = TrainingClass(label=label, name=name)
        self.classes[label] = new_class
        return new_class

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################


class TrainingClass(object):
    CLASSES_COLOR_MAP = {
        0: colors.getColor(name='red', out_type='BGR'),
        1: colors.getColor(name='blue', out_type='BGR'),
        2: colors.getColor(name='teal', out_type='BGR'),
        3: colors.getColor(name='lime', out_type='BGR'),
        4: colors.getColor(name='orange', out_type='BGR'),
        5: colors.getColor(name='amber', out_type='BGR'),
        6: colors.getColor(name='indigo', out_type='BGR')
    }

    def __init__(self, label=-1, name=""):
        self.label = label
        self.name = name
        self.instances = []

    def createTrainingInstance(self, frame=PyKDL.Frame(), size=np.array([0.1, 0.1, 0.1])):
        instance = TrainingInstance(
            frame=frame,
            size=size,
            label=self.label
        )
        self.instances.append(instance)
        return instance

    def getName(self):
        return self.name

    @staticmethod
    def getColorByLabel(label):
        if label in TrainingClass.CLASSES_COLOR_MAP:
            return TrainingClass.CLASSES_COLOR_MAP[label]
        if int(label) in TrainingClass.CLASSES_COLOR_MAP:
            return TrainingClass.CLASSES_COLOR_MAP[int(label)]

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################


class TrainingInstance(PyKDL.Frame):

    def __init__(self, frame=PyKDL.Frame(), size=[0.1, 0.1, 0.1], label=-1):
        super(TrainingInstance, self).__init__()
        self.M = frame.M
        self.p = frame.p
        self.size = size
        self.label = label

    def getRPY(self):
        return self.M.GetRPY()

    def setRPY(self, r2, p2, y2):
        r, p, y = self.getRPY()
        r2 = r if r2 == None else r2
        p2 = p if p2 == None else p2
        y2 = y if y2 == None else y2
        self.M = PyKDL.Rotation.RPY(r2, p2, y2)

    def toJSON(self):
        data = {}
        q = self.M.GetQuaternion()
        frame_enroll = [
            self.p.x(),
            self.p.y(),
            self.p.z(),
            q[0], q[1], q[2], q[3]
        ]
        data['frame'] = frame_enroll
        data['size'] = self.size
        data['label'] = self.label
        return data

    def __str__(self):
        return "Instance[label={}, p={}]".format(self.label, self.p)

    @staticmethod
    def fromJSON(json_data):
        inst = TrainingInstance()
        inst.M = PyKDL.Rotation.Quaternion(
            json_data["frame"][3],
            json_data["frame"][4],
            json_data["frame"][5],
            json_data["frame"][6]
        )
        inst.p = PyKDL.Vector(
            json_data["frame"][0],
            json_data["frame"][1],
            json_data["frame"][2]
        )
        inst.size = json_data["size"]
        inst.label = json_data["label"]
        return inst

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################


class TrainingFrame(object):

    def __init__(self, scene=TrainingScene(), internal_index=-1):
        self.scene = scene
        self.internal_index = internal_index
        self.image_path = self.scene.getImagePath(self.internal_index)
        self.camera_pose = self.scene.getCameraPose(self.internal_index)

    def getImagePath(self):
        return self.image_path

    def getCameraPose(self):
        return self.camera_pose

    def getCameraParams(self):
        return self.scene.camera_params

    def getInstancesGT(self):
        instances = self.scene.getAllInstances()
        gts = []
        for inst in instances:
            vobj = VirtualObject(frame=inst, size=inst.size, label=inst.label)
            img_pts = vobj.getImagePoints(
                camera_frame=self.getCameraPose(),
                camera=self.getCameraParams()
            )

            img_frame = vobj.getImageFrame(
                img_pts,
                camera=self.getCameraParams(),
                only_top_face=False
            )

            if VirtualObject.isValidFrame(img_frame):
                img_frame.insert(0, inst.label)
                gts.append(img_frame)

        return gts

    def __str__(self):
        return "TrainingFrame[{},{}]".format(self.scene.getName(), self.internal_index)

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################


class TrainingDataset(object):

    def __init__(self, scenes):
        self.scenes = scenes

    def getAllFrames(self):
        frames = []
        for s in self.scenes:
            frames.extend(s.getAllFrames())
        return frames

    def generateRandomFrameSet(self, test_percentage, validity_percentage=0.0):
        frames = self.getAllFrames()
        random.shuffle(frames)

        all_count = len(frames)
        test_count = int(all_count * test_percentage)
        val_count = int(all_count * validity_percentage)
        train_count = all_count - test_count - val_count

        trains = frames[:train_count]
        remains = frames[train_count:]

        tests = remains[:test_count]
        vals = remains[test_count:]
        return trains, tests, vals

    @staticmethod
    def buildDatasetFromManifestsFolder(manifests_folder):
        #⬢⬢⬢⬢⬢➤ Load manifests files
        manifests = []
        for root, directories, files in os.walk(manifests_folder):
            manifests = files
            break

        #⬢⬢⬢⬢⬢➤ Load scenes
        scenes = []
        frames_counter = 0
        for man in manifests:
            filename = os.path.join(manifests_folder, man)
            scene = TrainingScene.loadFromFile(filename)
            print("Loading: {}".format(man))
            if not scene.isValid():
                print("Scene '{}' is not valid!", filename)
                return None
            scenes.append(scene)
            frames_counter += scene.size()

        return TrainingDataset(scenes)


########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################


class DatasetBuilder(object):

    def __init__(self, training_dataset, dest_folder):
        self.training_dataset = training_dataset
        self.dest_folder = dest_folder

    def build(self, options={}):
        print("Build not implemented for '{}'".format(type(self)))


class YoloDatasetBuilder(DatasetBuilder):

    def __init__(self, training_dataset, dest_folder):
        super(YoloDatasetBuilder, self).__init__(training_dataset, dest_folder)

    def build(self, options={}):
        test_percentage = options["test_percentage"] if "test_percentage" in options else 0.05
        validity_percentage = options["validity_percentage"] if "validity_percentage" in options else 0.0

        #⬢⬢⬢⬢⬢➤ GENERATES SUBSETS
        trains, tests, vals = self.training_dataset.generateRandomFrameSet(
            test_percentage, validity_percentage)

        print("Train: ", len(trains))
        print("Test: ", len(tests))
        print("Val: ", len(vals))
        print("Total {}/{}".format(len(self.training_dataset.getAllFrames()),
                                   len(trains) + len(tests) + len(vals)))

        images_folder = os.path.join(self.dest_folder, 'images')
        labels_folder = os.path.join(self.dest_folder, 'labels')

        try:
            os.mkdir(images_folder)
            os.mkdir(labels_folder)
        except:
            pass

        train_file = open(os.path.join(self.dest_folder, 'train.txt'), 'a')
        test_file = open(os.path.join(self.dest_folder, 'test.txt'), 'a')
        val_file = open(os.path.join(self.dest_folder, 'val.txt'), 'a')

        sets = {
            'trains': trains,
            'tests': tests,
            'val': vals
        }

        sets_manifests = {
            'trains': train_file,
            'tests': test_file,
            'val': val_file
        }

        counter = 0
        for set_name, set_list in sets.iteritems():
            for frame in set_list:
                name = str(counter).zfill(6)
                _, extension = os.path.splitext(frame.getImagePath())

                image_path = os.path.join(
                    images_folder,
                    name + extension
                )
                label_path = os.path.join(
                    labels_folder,
                    name + ".txt"
                )

                shutil.copyfile(frame.getImagePath(), image_path)

                gts = np.array(frame.getInstancesGT())
                np.savetxt(label_path, gts, fmt='%d %1.4f %1.4f %1.4f %1.4f')

                sets_manifests[set_name].write(image_path + "\n")

                counter += 1
                print(set_name, counter)

            # image_filename = os.path.join(
            #            filename, 'rgb_{}.{}'.format(str(i).zfill(image_number_padding), image_format))

        train_file.close()
        test_file.close()
        val_file.close()
