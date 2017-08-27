# Silatra

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
	
The **Files found Online are those files that have been downloaded and might have been used
