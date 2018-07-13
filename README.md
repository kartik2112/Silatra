## Dataset

The training images, videos can be found here: [Hand Poses, Gestures Dataset - SiLaTra](https://drive.google.com/file/d/1BkINAqq8-Fknoo4WKkJu733RtTWQ9po4/view?usp=sharing)

This zip file has 
* **Dataset/Hand_Poses_Dataset** which consist of all static signs (Letters, Digits and Gesture Signs (which are intermediate signs of gestures))
* **Dataset/Gesture_Videos_Dataset** which are videos of gestures used for training HMM
* **Dataset/TalkingHands_Original_Videos_Not Used For Training_Just For Reference** consists of videos downloaded from Talking Hands which were used as a reference for generating training videos which are stored in **Dataset/Gesture_Videos_Dataset**.

## Installation and Usage

### Dependency Details

The dependencies used in python and that can be directly installed using ``pip install libName`` are ``hmmlearn``, ``sklearn`` (for kNN), ``pandas``, ``netifaces``, ``argparse``, ``numpy``, ``imutils``, ``dlib`` (For face detection), ``Flask``, ``atexit``, ``pickle``. OpenCV library cannot be installed on Ubuntu directly using ``pip install`` command. For the latest build a lot of steps are involved which are specified well by PyImageSearch here: [Ubuntu 16.04: How to install OpenCV - PyImageSearch](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)

### Local Usage

You can start the Flask server on the local machine by giving command: ``python3 server.py``. Now you can start using these services by using the latest [Silatra Android App](https://github.com/DevendraVyavaharkar/SiLaTra-UDP). The IP Address of the local machine needs to be specified in Settings. **Keep Direct Connection unchecked.** Now, you can click on Message icon and click on Capture button. This will start the transmission feed. Before clicking on the Capture button, you can select the mode - **SIGN** or **GESTURE**. After the need is over, click on Stop button. You can use the flash button to switch on flash. 

**Note:** Direct connection should be used only when you know what you are doing. Its usage is as follows:
* Instead of starting ``server.py``, here we start ``Receiver.py`` directly by specifying the desired port No. Example:
    ``python3 Receiver.py --portNo 49164 --displayWindows False --recognitionMode SIGN --socketTimeOutEnable True``
* After this, in Settings, you specify the machine's IP Address and this port no. (49164). This port no. can be any open port on which TCP Server Socket can be established.

``server.py`` eases your work by just needing you to invoke using command ``python3 server.py`` and specifying only the IP address in Settings. The socket invocation is internally handled by ``server.py``.

### Receiver.py arguments Usage

    Receiver.py [-h] [--portNo PORTNO] [--displayWindows DISPLAYWINDOWS]
                    [--recognitionMode RECOGNITIONMODE]
                    [--socketTimeOutEnable SOCKETTIMEOUTENABLE]
                    [--stabilize STABILIZE] [--recordVideos RECORDVIDEOS]
                    [--subDir SUBDIR]

    Main Entry Point

    optional arguments:
    -h, --help            show this help message and exit
    --portNo PORTNO       
                            Usage: python3 Receiver.py --portNo 12345
    --displayWindows DISPLAYWINDOWS
                            Usage: python3 Receiver.py --displayWindows True | False
    --recognitionMode RECOGNITIONMODE
                            Usage: python3 Receiver.py --recognitionMode SIGN | GESTURE
    --socketTimeOutEnable SOCKETTIMEOUTENABLE
                            Usage: python3 Receiver.py --socketTimeOutEnable True | False
    --stabilize STABILIZE
                            Usage: python3 Receiver.py --stabilize True | False
    --recordVideos RECORDVIDEOS
                            Usage: python3 Receiver.py --recordVideos True | False --subDir GN
    --subDir SUBDIR       
                            Usage: python3 Receiver.py --recordVideos True | False --subDir GN
    
Example usage:
``python3 Receiver.py --portNo 49164 --displayWindows False --recognitionMode SIGN --socketTimeOutEnable True``



## Installation on AWS Ubuntu flavour

AWS Ubuntu 16.04 server free-tier image was chosen with 30 GB SSD. This has 1 GB RAM. So we extended the swap by 3 GB with reference from: [Linux Add a Swap File – HowTo - nixCraft](https://www.cyberciti.biz/faq/linux-add-a-swap-file-howto/). All the dependencies were installed exactly in the way specified above, installing any intermediate dependencies that have been missed out from above. ``dlib`` cannot be installed (either from source or by using ``pip``) directly. This was the main reason swap file was added.

Now, we needed to run server.py on Flask server that would assign port No and invoke server socket for recognition. And for remote execution, server.py needs to run in background. Using & proved to be of no use since, once we quit the remote shell, the server is terminated as it will be a child of the remote shell. For this purpose we used ``supervisorctl``. For installation and configuring invocation of this server, we referred: [How To Install and Manage Supervisor on Ubuntu and Debian VPS - Digital Ocean](https://www.digitalocean.com/community/tutorials/how-to-install-and-manage-supervisor-on-ubuntu-and-debian-vps). The ``conf`` file that has been stored can be found [here (silatra_server.conf)](/SiLaTra_Server/silatra_server.conf). This ``conf`` file has been stored at **/etc/supervisor/conf.d/silatra_server.conf** on the AWS Linux. After this you can hit the commands specified on [How To Install and Manage Supervisor on Ubuntu and Debian VPS - Digital Ocean](https://www.digitalocean.com/community/tutorials/how-to-install-and-manage-supervisor-on-ubuntu-and-debian-vps). By using ``sudo supervisorctl``, you can see that this server is running.



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

### Archived Codes

Contain the older versions of these codes in the outer directory. These have been deprecated. These codes have the remnants of the older versions such as "Feature Extraction Using Fourier Descriptors - C++ implementation", "Gesture Recognition using Automata" and other such temporary testing codes for experimentation. They have been kept so as to go back in time to revisit the efforts in the hope that someday it could help anyone in some way.

### Gesture Videos By TalkingHands

These contain gesture videos which have been downloaded from Talking Hands website. These were the ones that were referred for creating training videos. Some of them were considered finally for gestures because they were easy. (These gestures that are supported in the final version are 1-handed and do not involve the hand overlapping the face any time).