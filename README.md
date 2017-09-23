# ROARS: RObot for Augmented Reality Self-labeling

## Training Datasets

As stated in the paper, here are all the training datasets used in our experiments:

* Industrial Dataset:
  * [Industrial_1000_A](http://www.vision.deis.unibo.it/Roars/Industrial_1000_A.tar.gz)
  * [Industrial_1000_M](http://www.vision.deis.unibo.it/Roars/Industrial_1000_M.tar.gz)
  * [Industrial_3000_A](http://www.vision.deis.unibo.it/Roars/Industrial_3000_A.tar.gz)
  * [Industrial_5000_A](http://www.vision.deis.unibo.it/Roars/Industrial_5000_A.tar.gz)
  * [Industrial_15000_A](http://www.vision.deis.unibo.it/Roars/Industrial_15000_A.tar.gz)
  * [Industrial_Test_A](http://www.vision.deis.unibo.it/Roars/Industrial_Test_A.tar.gz)
  * [Industrial_Test_M](http://www.vision.deis.unibo.it/Roars/Industrial_Test_M.tar.gz)
* Fruits Dataset:
  * [Fruits_2500](http://www.vision.deis.unibo.it/Roars/Fruits_2500.tar.gz)
  * [Fruits_5000](http://www.vision.deis.unibo.it/Roars/Fruits_5000.tar.gz)
  * [Fruits_7500](http://www.vision.deis.unibo.it/Roars/Fruits_7500.tar.gz)
  * [Fruits_Test](http://www.vision.deis.unibo.it/Roars/Fruits_Test.tar.gz)

Each Dataset (that we call *Raw Dataset*) is a folder containing:

* *images*: subfolder containing an ordered list of images, in the form "rgb_\<NUMBER>.jpg"
* *labels*: subfolder containing an ordered list of labels, in the form "\<NUMBER>.txt" (this NUMBER match with the one in the corresponding image filename)
  * each row in the *label* file is in the form "\<CLASS> \<X> \<Y> \<W> \<H>" representing a bounding box in relative coordinates (Percentage value 0..1 of the size of the image) and X,Y is the center of the box.
* *ids*: subfolder containing an ordered list of ids, used for debug purposes

Each datasets also contains a simple "verify.py" python script that picks&shows randomly a frame in the dataset (including relative bounding boxes); it's used only for debug.

## Generate and Explore a Scene

### Dependencies

* PyQt4 (sudo apt-get install python-qt4)
* QDarkStyle (pip install qdarkstyle)
* Opencv
* KDL

A *Roars Scene* is a simple folder containing a dataset relative to a robot scan. The folder shall contain:

* ***images*** : a folder containing an ordered list of images
* *robot_poses.txt*: a file containing an ordere list of robot poses.. the i-th robot pose is relative to the i-th image
* *camera_extrinsics.txt*: a file containing the transformation of the matrix relative to the robot wrist. [x y z qx qy qz qw]
* *camera_intrinsics.txt*: a file containing the camera intrinsics parameters [w h fx fy cx cy k1 k2 p1 p2]

You can download a Demo Scene from here: [Example Scene](http://www.vision.deis.unibo.it/Roars/Roars_Demo_Scene.tar.gz). This is a full working scene ready to be used.

From ours *Roars Scene* we need to create a *Manifest* that is just a descriptor of the scene that contains labeling informations (e.g. classes and instances). This manifest is used as input data for each roars labeling tool.
Here an example to generate the manifest from the provided Demo Scene:

* ```wget http://www.vision.deis.unibo.it/Roars/Roars_Demo_Scene.tar.gz```
* ```tar -xvf Roars_Demo_Scene.tar.gz```
* ```rosrun roars generate_roars_dataset_manifest.py _scene_path:=generic_scene _output_manifest_file:=generic_scene.roars _classes:='apple;pear;plum;peach;banana'```

The node *generate_roars_dataset_manifest.py* produce the output file ```_output_manifest_file:=generic_scene.roars``` parsing the subfolder ```_scene_path:=generic_scene```. The manifest will be decorated with a list of labels described with ```_classes:='apple;pear;plum;peach;banana'``` (each label is ';'-separated).
With the manifest file now we can launch our Scene Explorer with:

* ```rosrun roars roars_scene_explorer.py _manifest:=generic_scene.roars _window_size:=1200x700```

The parameter ```_manifest:=generic_scene.roars``` is our manifest file; with ```_window_size:=1200x700``` we can choose a good resolution in accordance with your display.





