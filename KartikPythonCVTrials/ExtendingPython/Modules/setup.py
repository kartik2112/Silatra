# Reference: https://docs.python.org/3.5/extending/building.html#building
# Reference: https://docs.python.org/3.5/distutils/apiref.html#module-distutils.extension

from distutils.core import setup, Extension

module1 = Extension('spam',
                    # libraries = ['python3.5m'],
                    # library_dirs = ['/usr/local/include/python3.5'],
                    sources = ['spammodule.c'])

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])