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
