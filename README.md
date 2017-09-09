# Silatra
**Si**gn **La**nguage **Tra**nslator is an application that will convert Indian Sign Language gestures recorded via camera to English Text. More to follow.

Developed by:
* Tejas Dastane
* Varun Rao
* Kartik Shenoy
* Devendra Vyavaharkar

### Steps to build the file:

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
  
  To run the program on an image and see the execution use:
  
  `./build/SiLaTra -img ./training-images/Digits/1/Right_Hand/Normal/1.png`
  
  To run the program and create the training data set while seeing the execution, use:
  
  `./build/SiLaTra -cap Digits/1/Right_Hand/Normal`
  
  To store the image click key `c` while the program is executing.
  
  To stop the program executuon, click `q`
	
The *Files found Online* are those files that have been downloaded and might have been used

### Git Related Usage
* This repository has stored datasets. These are of sizes greater than 100 MB (max limit for Normal Github usage)
  For this purpose, Git LFS (Large File Storage) extension has been found. To install this on Ubuntu, execute the following commands:
  
  `curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash`
  
* After the repository of packages has been downloaded, install the required package by executing the command:

  `sudo apt-get install git-lfs`
  
* To verify the same has been installed, execute

  `git lfs install`
  
* Now clone the repository and then from the root of that repository run

  `git lfs track "*.zip"`
  
For more information, refer [Github LFS Reference](https://help.github.com/articles/working-with-large-files/)
