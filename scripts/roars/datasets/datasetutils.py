
# -*- encoding: utf-8 -*-

import PyKDL
import numpy as np
import json
import os
import glob
import random
import shutil
import roars.geometry.transformations as transformations
import roars.vision.colors as colors
from roars.vision.augmentereality import *
import collections


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

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################


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

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################


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

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################


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

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################


class TrainingScene(object):
    DEFAULT_CAMERA_POSE_NAME = 'camera_extrinsics.txt'
    DEFAULT_CAMERA_PARAMS_NAME = 'camera_intrinsics.txt'
    AVAILABLE_FILE_FORMATS = ['jpg', 'png', 'JPG', 'PNG', 'bmp']

    def __init__(self, scene_path='#', images_path='aaa', images_depth_path=None, robot_pose_name='#', camera_intrisics_file='', camera_extrinsics_file='', relative_path=None):
        self.classes = {}
        self.scene_path = scene_path
        self.images_path = images_path
        self.images_depth_path = images_depth_path
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
        self.relative_path = relative_path

    def setRelativePath(self, filename):
        self.relative_path = os.path.dirname(filename)

    def initFromManifest(self, manifest_file):
        if not self.scene_path.startswith('/'):
            self.setRelativePath(manifest_file)
            self.scene_path = os.path.join(self.relative_path, self.scene_path)

    def initialize(self):
        ''' Initializes the Scene searching for Images and camera poses '''

        #⬢⬢⬢⬢⬢➤ Checks for Scene path
        if os.path.exists(self.scene_path):
            images_path = os.path.join(self.scene_path, self.images_path)
        else:
            print("Scene path doesn't exist: {}".format(self.scene_path))
            return

        #⬢⬢⬢⬢⬢➤ Checks for Images folder path
        if os.path.exists(images_path):
            # Lists all Images in scene
            files_temp = glob.glob(os.path.join(images_path, '*.jpg'))
            if len(files_temp) == 0:
                files_temp = glob.glob(os.path.join(images_path, '*.png'))
            files = []
            for f in files_temp:
                files.append(os.path.basename(f))
            # TODO: enable other image extensions
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
        self.image_filenames_lists = sorted(files)

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

    def getScenePath(self, only_base=True):
        if only_base:
            path = self.scene_path
            if path.endswith("/"):
                path = path[:-1]
            return os.path.basename(path)
        else:
            return self.scene_path

    def getName(self):
        return os.path.basename(self.scene_path)

    def setClasses(self, classes):
        if isinstance(classes, TrainingClassesMap):
            classes = classes.getClasses()

        temp_instances = self.getAllInstances()
        temp_classes = collections.OrderedDict(
            sorted(classes.items())
        )
        self.classes = {}
        for k, v in temp_classes.iteritems():
            self.classes[int(k)] = v
            self.classes[int(k)].clearInstances()
        for inst in temp_instances:
            if inst.label in self.classes:
                self.classes[inst.label].instances.append(inst)
            else:
                if -1 in self.classes:
                    self.classes[-1].instances.append(inst)

    def clearInstances(self):
        for k, ti in self.classes.iteritems():
            ti.clearInstances()

    def setInstances(self, instances):
        self.clearInstances()
        for inst in instances:
            self.getTrainingClass(inst.label).addInstances([inst])

    def getClassesAsList(self):
        return TrainingClassesMap.transformClassesInStringList(self.classes)

    def generateClassesMap(self):
        return TrainingClassesMap(class_list=self.getClassesAsList())

    def clearClasses(self):
        self.classes = {}

    def classesNumber(self):
        return len(self.classes)

    def getTrainingClass(self, index, force_creation=False):
        if index in self.classes:
            return self.classes[index]
        else:
            if force_creation:
                self.classes[index] = TrainingClass(label=-1)
                return self.getTrainingClass(index)
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
        return os.path.join(self.scene_path, self.images_path, self.image_filenames_lists[index])

    def getImageDepthPath(self, index):
        try:
            if self.images_depth_path == None:
                return None
        except:
            return None

        index = index % self.size()
        path = os.path.join(
            self.scene_path, self.images_depth_path, self.image_filenames_lists[index])

        if not os.path.exists(path):
            for ext in TrainingScene.AVAILABLE_FILE_FORMATS:
                path = os.path.splitext(path)[0] + '.' + ext
                if os.path.exists(path):
                    return path
        else:
            return path
        return None

    def getFrameByIndex(self, index):
        return TrainingFrame(scene=self, internal_index=index)

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

    def save(self, filename, force_relative=False):
        if force_relative:
            self.setRelativePath(filename)

        #⬢⬢⬢⬢⬢➤ Remove Relative Path if any
        if self.relative_path != None:
            self.scene_path = os.path.relpath(
                self.scene_path, self.relative_path)

        #⬢⬢⬢⬢⬢➤ Save
        scene_js = json.dumps(
            self,
            cls=CustomJSONEncoder,
            sort_keys=True, indent=4
        )
        with open(filename, "w") as outfile:
            outfile.write(scene_js)

        #⬢⬢⬢⬢⬢➤ Restore relative path if any
        if self.relative_path != None:
            self.scene_path = os.path.join(self.relative_path, self.scene_path)

    def validateImport(self):
        self.setClasses(self.classes)

    @staticmethod
    def loadFromFile(filename):
        try:
            f = open(filename)
            sc = json.loads(f.read(), object_hook=CustomJSONDecoder.decode)
            if not isinstance(sc, TrainingScene):
                return None
            sc.validateImport()
            sc.initFromManifest(filename)
            return sc
        except ValueError, e:
            return None

    def getTrainingClassByLabel(self, label):
        if label in self.classes:
            return self.classes[label]
        return None

    def createTrainingClass(self, label, name=""):
        new_class = TrainingClass(label=label, name=name)
        self.classes[label] = new_class
        return new_class

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################


class TrainingClass(object):
    CLASSES_COLOR_MAP = {
        -1: 'white',
        0: 'red',
        1: 'blue',
        2: 'teal',
        3: 'lime',
        4: 'purple',
        5: 'green',
        6: 'cyan'
    }

    def __init__(self, label=-1, name=""):
        self.label = label
        self.name = name
        self.instances = []

    def clearInstances(self):
        self.instances = []

    def addInstances(self, instances):
        self.instances.extend(instances)

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
    def generateClassListFromInstances(instances, classes_map=None):
        classes = {}
        for inst in instances:
            if inst.label not in classes:
                name = ""
                if classes_map:
                    name = classes_map.getClassName(inst.label)
                classes[inst.label] = TrainingClass(
                    label=inst.label, name=name)
            classes[inst.label].instances.append(inst)
        return classes

    @staticmethod
    def getColorByLabel(label, output_type="BGR"):
        if label not in TrainingClass.CLASSES_COLOR_MAP:
            label = label % len(TrainingClass.CLASSES_COLOR_MAP)

        name = TrainingClass.CLASSES_COLOR_MAP[label]
        return colors.getColor(name=name, out_type=output_type)

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################


class TrainingClassesMap(object):
    DEFAULT_NOCLASS_NAME = "No-Class"

    def __init__(self, class_list):

        class_map = {}
        az_classes = sorted(class_list)
        az_classes.insert(0, TrainingClassesMap.DEFAULT_NOCLASS_NAME)
        for i in range(0, len(az_classes)):
            class_map[i - 1] = az_classes[i]
        class_map = collections.OrderedDict(
            sorted(class_map.items())
        )

        self.classes = {}
        for k, name in class_map.iteritems():
            self.classes[k] = TrainingClass(k, name)

    def getClasses(self):
        return self.classes

    def getClassName(self, class_index):
        if class_index in self.class_map:
            return self.class_map[class_index].name
        else:
            return None

    def getClassIndex(self, class_name):
        for k, v in self.class_map.iteritems():
            if v.name == class_name:
                return k
        return None

    def getUserFriendlyStringList(self):
        l = []
        for k, v in self.class_map.iteritems():
            l.append("{} [{}]".format(v.name, k))
        return l

    def userFriendlyStringToPair(self, ustr):
        ustr = ustr.replace("[", "").replace("]", "")
        k = int(ustr.split(" ")[0])
        v = str(ustr.split(" ")[0])
        return k, v


##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################


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

    def setSize(self, sx, sy, sz):
        self.size = np.array([sx, sy, sz])

    def grows(self, label, delta):
        if label == 'x':
            self.grow(delta, 0.0, 0.0)
        if label == 'y':
            self.grow(0.0, delta, 0.0)
        if label == 'z':
            self.grow(0.0, 0.0, delta)

    def grow(self, sx, sy, sz):
        self.size = self.size + np.array([sx, sy, sz])

    def setFrameProperty(self, name, value):
        if name == 'cx':
            self.p.x(value)
        if name == 'cy':
            self.p.y(value)
        if name == 'cz':
            self.p.z(value)
        if name == 'roll':
            self.setRPY(value, None, None)
        if name == 'pitch':
            self.setRPY(None, value, None)
        if name == 'yaw':
            self.setRPY(None, None, value)

    def relativeTranslations(self, label, delta):
        if label == 'x':
            self.relativeTranslation(x=delta)
        if label == 'y':
            self.relativeTranslation(y=delta)
        if label == 'z':
            self.relativeTranslation(z=delta)

    def relativeTranslation(self, x=None, y=None, z=None):
        x = x if x != None else 0.0
        y = y if y != None else 0.0
        z = z if z != None else 0.0
        tv = PyKDL.Frame(PyKDL.Vector(x, y, z))
        current = PyKDL.Frame()
        current.M = self.M
        current.p = self.p
        current = current * tv
        self.M = current.M
        self.p = current.p

    def getFrameProperty(self, name):
        if name == 'cx':
            return self.p.x()
        if name == 'cy':
            return self.p.y()
        if name == 'cz':
            return self.p.z()
        if name == 'roll':
            return self.getRPY()[0]
        if name == 'pitch':
            return self.getRPY()[1]
        if name == 'yaw':
            return self.getRPY()[2]

    def getFrameProperties(self):
        return {
            'cx': self.p.x(),
            'cy': self.p.y(),
            'cz': self.p.z(),
            'roll': self.getRPY()[0],
            'pitch': self.getRPY()[1],
            'yaw': self.getRPY()[2]
        }

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

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################


class TrainingFrame(object):

    def __init__(self, scene=TrainingScene(), internal_index=-1):
        self.scene = scene
        self.internal_index = internal_index
        self.image_path = self.scene.getImagePath(self.internal_index)
        self.image_depth_path = self.scene.getImageDepthPath(
            self.internal_index)
        self.camera_pose = self.scene.getCameraPose(self.internal_index)

    def getId(self):
        return self.getScene().getScenePath(only_base=True) + "!" + self.getImageName()

    def getScene(self):
        return self.scene

    def getImageName(self):
        return os.path.basename(self.getImagePath())

    def getImagePath(self):
        return self.image_path

    def getImageDepthPath(self):
        return self.image_depth_path

    def getCameraPose(self):
        return self.camera_pose

    def getCameraParams(self):
        return self.scene.camera_params

    def getInstances(self):
        return self.scene.getAllInstances()

    def getInstancesBoxes(self):
        instances = self.scene.getAllInstances()
        boxes = []
        for inst in instances:
            vobj = VirtualObject(frame=inst, size=inst.size, label=inst.label)
            img_pts = vobj.getImagePoints(
                camera_frame=self.getCameraPose(),
                camera=self.getCameraParams()
            )
            boxes.append(img_pts)

        return boxes

    def getInstancesBoxesWithLabels(self, sorted=True, filter_labels=[]):
        instances = self.scene.getAllInstances()
        boxes = []
        for inst in instances:
            if len(filter_labels) > 0:
                if inst.label not in filter_labels:
                    continue
            vobj = VirtualObject(frame=inst, size=inst.size, label=inst.label)
            img_pts = vobj.getImagePoints(
                camera_frame=self.getCameraPose(),
                camera=self.getCameraParams()
            )
            min_y = np.max(np.array(img_pts)[:, 1])
            boxes.append((inst.label, img_pts, min_y))

        if sorted:
            boxes.sort(key=lambda tup: tup[2])
        return boxes

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

    @staticmethod
    def drawGT(output, gt_data):
        frame_data = gt_data[1:5]
        print(frame_data)
        VirtualObject.drawFrame(frame_data, output)

    def __str__(self):
        return "TrainingFrame[{},{}]".format(self.scene.getName(), self.internal_index)

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################


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


##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################


class DatasetBuilder(object):

    def __init__(self, training_dataset, dest_folder):
        self.training_dataset = training_dataset
        self.dest_folder = dest_folder

    def build(self, options={}):
        print("Build not implemented for '{}'".format(type(self)))


class RawDatasetBuilder(DatasetBuilder):
    ZERO_PADDING_SIZE = 5

    def __init__(self, training_dataset, dest_folder):
        super(RawDatasetBuilder, self).__init__(training_dataset, dest_folder)

    def build(self, options={}):

        if os.path.exists(self.dest_folder) or len(self.dest_folder) == 0:
            return False

        os.mkdir(self.dest_folder)

        img_folder = os.path.join(self.dest_folder, "images")
        label_folder = os.path.join(self.dest_folder, "labels")
        ids_folder = os.path.join(self.dest_folder, "ids")

        os.mkdir(img_folder)
        os.mkdir(label_folder)
        os.mkdir(ids_folder)

        frames = self.training_dataset.getAllFrames()

        counter = 0
        for frame in frames:

            counter_string = '{}'.format(str(counter).zfill(
                RawDatasetBuilder.ZERO_PADDING_SIZE))

            img_file = os.path.join(img_folder, counter_string + ".jpg")
            label_file = os.path.join(label_folder, counter_string + ".txt")
            id_file = os.path.join(ids_folder, counter_string + ".txt")

            print("IAMGE COPYING", frame.getImagePath(), img_file)
            shutil.copyfile(frame.getImagePath(), img_file)

            gts = np.array(frame.getInstancesGT())
            if gts.size == 0:
                open(label_file, 'a').close()
            else:
                np.savetxt(label_file, gts, fmt='%d %1.4f %1.4f %1.4f %1.4f')

            f = open(id_file, 'w')
            f.write(frame.getId())
            f.close()

            print frame.getId()
            counter = counter + 1

        return True


class PixDatasetBuilder(DatasetBuilder):
    ZERO_PADDING_SIZE = 5

    def __init__(self, training_dataset, dest_folder, jumps=5, val_percentage=0.05, test_percentage=0.1, randomize_frames=False):
        super(PixDatasetBuilder, self).__init__(training_dataset, dest_folder)
        self.jumps = jumps
        self.val_percentage = val_percentage
        self.test_percentage = test_percentage
        self.randomize_frames = randomize_frames

    def build(self, options={}):

        if os.path.exists(self.dest_folder) or len(self.dest_folder) == 0:
            return False

        os.mkdir(self.dest_folder)

        #⬢⬢⬢⬢⬢➤ Generates frames subsplit in TRAIN;TEST;VAL
        if self.val_percentage > 0.00001 or self.test_percentage > 0.00001:
            frames_splits = self.training_dataset.generateRandomFrameSet(
                self.test_percentage, self.val_percentage)
        else:
            frames_splits = (self.training_dataset.getAllFrames(), [], [])
        split_names = ['train', 'test', 'val']

        counter = 0
        for index in range(0, len(frames_splits)):
            frames = frames_splits[index]
            name = split_names[index]

            folder = os.path.join(self.dest_folder, name)
            os.mkdir(folder)

            counter = 0
            for frame in frames:
                if counter % self.jumps == 0:
                    counter_string = '{}'.format(str(int(counter / self.jumps)).zfill(
                        PixDatasetBuilder.ZERO_PADDING_SIZE))

                    # img_file = os.path.join(img_folder, counter_string + ".jpg")
                    # label_file = os.path.join(label_folder, counter_string + ".txt")
                    # id_file = os.path.join(ids_folder, counter_string + ".txt")

                    # print("IAMGE COPYING", frame.getImagePath(), img_file)
                    # shutil.copyfile(frame.getImagePath(), img_file)

                    gts = frame.getInstancesBoxesWithLabels()
                    print name, frame.getId()
                    img = cv2.imread(frame.getImagePath())
                    pair = np.ones(img.shape, dtype=np.uint8) * 255
                    for inst in gts:
                        hull = cv2.convexHull(np.array(inst[1]))
                        cv2.fillConvexPoly(
                            pair, hull, TrainingClass.getColorByLabel(inst[0]))

                    whole = np.hstack((pair, img))

                    img_file = os.path.join(folder, counter_string + ".jpg")
                    print "Writing to", img_file
                    cv2.imwrite(img_file, whole)
                    # full_stack.append(whole)

                    # import sys
                    # sys.exit(0)
                    # if gts.size == 0:
                    #     open(label_file, 'a').close()
                    # else:
                    #     np.savetxt(label_file, gts, fmt='%d %1.4f %1.4f %1.4f %1.4f')

                    # f = open(id_file, 'w')
                    # f.write(frame.getId())
                    # f.close()

                counter = counter + 1

        return True


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
            # filename, 'rgb_{}.{}'.format(str(i).zfill(image_number_padding),
            # image_format))

        train_file.close()
        test_file.close()
        val_file.close()


##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################

class RawDataset(object):

    def __init__(self, folder, create=False):

        self.folder = folder
        self.img_folder = os.path.join(folder, "images")
        self.label_folder = os.path.join(folder, "labels")
        self.ids_folder = os.path.join(folder, "ids")

        if create:
            if os.path.exists(self.folder) or len(self.folder) == 0:
                print("Folder '{}' already exists!".format(self.folder))
            else:
                os.mkdir(self.folder)
                os.mkdir(self.img_folder)
                os.mkdir(self.label_folder)
                os.mkdir(self.ids_folder)

        else:
            image_files = sorted(glob.glob(self.img_folder + "/*.jpg"))
            label_files = sorted(glob.glob(self.label_folder + "/*.txt"))
            id_files = sorted(glob.glob(self.ids_folder + "/*.txt"))

            self.ids = []
            self.data_map = {}
            self.data_list = []
            for index in range(0, len(image_files)):
                id_file = id_files[index]
                image = image_files[index]
                label = label_files[index]

                f = open(id_file, 'r')
                id = f.readline().replace('\n', '')
                self.ids.append(id)
                self.data_map[id] = {
                    'id': id_file,
                    'image': image,
                    'label': label
                }
                self.data_list.append(self.data_map[id])
                f.close()

    def getImageById(self, id):
        return self.image_maps[id]
