# Reference: https://docs.python.org/3.5/extending/building.html#building
# Reference: https://docs.python.org/3.5/distutils/apiref.html#module-distutils.extension

#Instructions to install moduke in python: 
# python3 setup.py build
# python3 setup.py install --record silatraPythonModuleNewFilesAdded.txt

from distutils.core import setup, Extension
import numpy

module1 = Extension('silatra',
                    extra_compile_args = ['-std=c++14'],
                    include_dirs = [numpy.get_include()],
                    # extra_objects = ['./HaarCascades/haarcascade_frontalface_default.xml'],
                    libraries = ['python3.5m','stdc++fs',
                                'opencv_calib3d','opencv_core','opencv_cudaarithm','opencv_cudabgsegm','opencv_cudacodec','opencv_cudafeatures2d','opencv_cudafilters','opencv_cudaimgproc','opencv_cudalegacy','opencv_cudaobjdetect','opencv_cudaoptflow','opencv_cudastereo','opencv_cudawarping','opencv_cudev','opencv_dnn','opencv_features2d','opencv_flann','opencv_highgui','opencv_imgcodecs','opencv_imgproc','opencv_ml','opencv_objdetect','opencv_photo','opencv_shape','opencv_stitching','opencv_superres','opencv_video','opencv_videoio','opencv_videostab','opencv_aruco','opencv_bgsegm','opencv_bioinspired','opencv_ccalib','opencv_cvv','opencv_datasets','opencv_dpm','opencv_face','opencv_freetype','opencv_fuzzy','opencv_hdf','opencv_img_hash','opencv_line_descriptor','opencv_optflow','opencv_phase_unwrapping','opencv_plot','opencv_reg','opencv_rgbd','opencv_saliency','opencv_stereo','opencv_structured_light','opencv_surface_matching','opencv_text','opencv_tracking','opencv_xfeatures2d','opencv_ximgproc','opencv_xobjdetect','opencv_xphoto'
                                ],
                    library_dirs = ['/usr/local/include','/usr/local/include/opencv',
                                '/usr/local/include/python3.5'
                                # '/home/kartik/.virtualenvs/cvAcc/lib/python3.5/site-packages/numpy/core/include'
                                ],
                    sources = ['SilatraWrapper.cpp',
                                'PythonInterfacingTrial/cv_cpp_py_interface.cpp',
                                './GetMyHand/handDetection.cpp','./GetMyHand/skinColorSegmentation.cpp',
                                './GetMyHand/trackBarHandling.cpp','./GetMyHand/predictionsHandler.cpp',
                                './GetMyHand/Classification/classifyPythonAPI.cpp'])

setup (name = 'SiLaTra',
       version = '1.0',
       description = 'This is a sign language translation package',
       ext_modules = [module1])

       