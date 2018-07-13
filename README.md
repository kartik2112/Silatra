## Dataset

The training images, videos can be found here: [Hand Poses, Gestures Dataset - SiLaTra](https://drive.google.com/file/d/1BkINAqq8-Fknoo4WKkJu733RtTWQ9po4/view?usp=sharing)

This zip file has 
* **Dataset/Hand_Poses_Dataset** which consist of all static signs (Letters, Digits and Gesture Signs (which are intermediate signs of gestures))
* **Dataset/Gesture_Videos_Dataset** which are videos of gestures used for training HMM
* **Dataset/TalkingHands_Original_Videos_Not Used For Training_Just For Reference** consists of videos downloaded from Talking Hands which were used as a reference for generating training videos which are stored in **Dataset/Gesture_Videos_Dataset**.

## Dependency Details

The dependencies used in python and that can be directly installed using ``pip install libName`` are ``hmmlearn``, ``sklearn`` (for kNN), ``pandas``, ``netifaces``, ``argparse``, ``numpy``, ``imutils``, ``dlib`` (For face
detection), ``Flask``. OpenCV library cannot be installed on Ubuntu directly using ``pip install`` command. For the latest build a lot of steps are involved which are specified well by PyImageSearch here: [Ubuntu 16.04: How to install OpenCV - PyImageSearch](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)

## Folder Content Description

### SiLaTra_Server
The folder **SiLaTra_Server** is the entry point into the server module. The Server developed is very modular. All the models, modules, dependency files are stored in this folder. When you download this folder, all these code dependencies will already be present. The entry point into this server module is server.py. The server.py code decides a port no randomly, checks if it is open and invokes Receiver.py to start TCP socket on this port. Receiver.py will manage recognition. The description of these resources used by Receiver.py in SiLaTra_Server folder are:

* silatra_utils.py: Contains functions for feature extraction from segmented hand, managing sign prediction by finding modal values in a stream of predictions and functions dealing with displaying signs on screen through windows.

* Modules/FaceEliminator.py: Contains functions for blackening face and neck area when given the detected face coordinates and segmented mask.

* Modules/PersonStabilizer.py: Used for stabilizing person as object using face as reference.

* Modules/TimingMod.py: Used for keeping track of time required for individual process activities (for performance measurement).

* Gesture_Modules/hmmGestureClassify.py: Contains everything required for handling gesture recognition – from accessing models to predicting probabilities for each one of them and comparing these probabilities for finding the recognized gesture.

* Gesture_Modules/directionTracker.py: Used for tracking hand centroid for determining motion for gesture recognition.

* Models: Contains k-NN, HMM Models for recognition

The server socket is created using this entry point. There are 2 ways of defining the
port:

* Port No specified using terminal interaction: ``python3 Receiver.py``

* Port No specified in commandline itself: ``python3 Receiver.py --portNo 12345``

After this is invoked, the IP Address and the port can be specified on the Android
application and user can see the real-time translation on screen.
Another argument provided in this program is the creation of gesture videos for training
HMM which can be used as: ``python3 Receiver.py --recordVideos True --subDir GN``.
Here, the subdirectory is specified where the developer wants to store his training
videos.


### SilatraPythonModuleBuilder

Open terminal here and install this module using command: ``python3 setup.py install``. This installs the silatra_cpp module used for segmentation.

### Utilities

Detailed description of the contents within this folder is provided in the README.md file inside the [Utilities Folder](/Utilities/).
