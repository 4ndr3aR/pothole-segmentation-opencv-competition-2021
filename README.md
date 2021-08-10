# Pothole Segmentation - OpenCV Competition 2021

We were able to run on OAK-D a semantic segmentation model trained with Fast.ai, the popular Pytorch-based library, and the dataset in [1] enriched with a few dozen other segmented hole images to make the detection of this class more robust. The architecture we have chosen is DeepLabv3+ with MobileNetv2 as the backbone. The videos were made by running the neural network model on the OAK-D.

As can be seen in the test set videos, the detection (red = pothole, blue = crack) is good enough, especially considering the particularly challenging video and the few thousand images available.

## Download the test set video

`wget http://deeplearning.ge.imati.cnr.it/genova-5G/video/opencv-competition-2021/VID_20210505_120526-test-set-walk.mp4`

## Connect the OAK-D, source the virtualenv and run with:

`./road-potholes-cracks-semantic-segmentation.py --videofile VID_20210505_120526-test-set-walk.mp4 --rotatevideo True --outputvideo VID_20210505_120526-test-set-walk-deeplabv3-mobilenetv2-no-data-aug-640x360.mp4`

## Screenshots from OAK-D live feed

![screenshot-01](https://github.com/4ndr3aR/pothole-segmentation-opencv-competition-2021/raw/main/pics/live-screenshots/screenshot-01.png)
![screenshot-02](https://github.com/4ndr3aR/pothole-segmentation-opencv-competition-2021/raw/main/pics/live-screenshots/screenshot-02.png)
![screenshot-03](https://github.com/4ndr3aR/pothole-segmentation-opencv-competition-2021/raw/main/pics/live-screenshots/screenshot-03.png)
![screenshot-04](https://github.com/4ndr3aR/pothole-segmentation-opencv-competition-2021/raw/main/pics/live-screenshots/screenshot-04.png)

## Inference on test set video

![test set video](https://github.com/4ndr3aR/pothole-segmentation-opencv-competition-2021/raw/main/videos/VID_20210505_120526-test-set-walk-deeplabv3-mobilenetv2-no-data-aug-640x360.mp4)
