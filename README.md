# Silatra
**Si**gn **La**nguage **Tra**nslator is an application that will convert Indian Sign Language gestures recorded via camera to English Text. More to follow.

Developed by:
* Tejas Dastane
* Varun Rao
* Kartik Shenoy
* Devendra Vyavaharkar

### Pre-requisites

* Install pytho-dev, python3-dev using following commands for invoking Python scripts from C++ files
  `sudo apt-get install python-dev`
  `sudo apt-get install python3-dev`

### Steps to build the file

Actually here cmake, make commands are used

But for simplicity the commands have been wrapped into a single file builder.sh

* The CMakeLists.txt file has the filename which is to be compiled. If you want to execute another file either change this filename in CMakeLists.txt file or change the filename to skinDetection.cpp
* Make sure build folder is there

### Usage
* From terminal execute command
	`./builder.sh`
  This will create the executable file
* For running the executable file, you can pass different arguments to run it in different modes

  To run the program in normal mode, that is see the execution realtime from webcam feed use:
  
  `./build/SiLaTra`
  
  To run the program on an image, see the execution and store the distances in the corresponding csv file use:
  
  `./build/SiLaTra -img ./training-images/Digits/1/Right_Hand/Normal/1.png`

  To run the program on a set of images in a particular folder with the same label and store the distances in the corresponding csv file use (Here, each time the corresponding csv file is cleaned):
  
  `./build/SiLaTra -AllImgs ./training-images/Digits/1/Right_Hand/Normal/`

  To run the program on multiple sets of images and generate the corresponding csv files use the following. This is similar too the previous mode, but can operate on multiple folders that are specified on commandline:

  `./build/SiLaTra -fullRefresh ./training-images/Digits/1/Right_Hand/Normal/ ./training-images/Digits/2/Right_Hand/Normal/ ./training-images/Digits/3/Right_Hand/Normal/ ./training-images/Digits/4/Right_Hand/Normal/ ./training-images/Digits/5/Right_Hand/Normal/ ./training-images/Digits/6/Right_Hand/Normal/ ./training-images/Digits/7/Right_Hand/Normal/ ./training-images/Digits/8/Right_Hand/Normal/ ./training-images/Digits/9/Right_Hand/Normal/ ./training-images/Digits/0/Right_Hand/Normal/`
  
  To run the program and create the training data set while seeing the execution, use:
  
  `./build/SiLaTra -cap Digits/1/Right_Hand/Normal`
  
  To store the image click key `c` while the program is executing.
  
  To stop the program executuon, click `q`

* You can classify the current image using the model specified in the code. For `-img`, `-AllImgs` modes,  by default you can see the classification. For `-cap` and normal mode, you can see the classification    for a particular image frame by hitting `m` (Currently intended to mean Magic). But this output will be seen amongst a continuous stream of contour data. So you need to scroll up to see this classification.
	
* Using the csv files (containing distances), you can classify them by running `python3 getMeSomeResultsInPythonBro.py`

The *Files found Online* are those files that have been downloaded and might have been used

### Git Related Usage
* This repository has stored datasets. These are of sizes greater than 100 MB (max limit for Normal Github usage)
  For this purpose, Git LFS (Large File Storage) extension has been found. To install this on Ubuntu, execute the following commands:
  1. `sudo apt-get install software-properties-common` to install add-apt-repository (or sudo apt-get install python-software-properties if you are on Ubuntu <= 12.04)
  2. `sudo add-apt-repository ppa:git-core/ppa`
  3. The curl script below calls apt-get update, if you aren't using it, don't forget to call `apt-get update` before installing git-lfs.
  
  `curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash`
  
* After the repository of packages has been downloaded, install the required package by executing the command:

  `sudo apt-get install git-lfs`
  
* To verify the same has been installed, execute

  `git lfs install`
  
* Now clone the repository and then from the root of that repository run

  `git lfs track "*.zip"`
  
For more information, refer [Github LFS Reference](https://help.github.com/articles/working-with-large-files/)

#### If this doesn't work out for you, follow [Installing Git Large File Storage](https://help.github.com/articles/installing-git-large-file-storage/)
