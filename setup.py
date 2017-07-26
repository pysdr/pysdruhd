# USAGE:
# python setup.py install --prefix=~/apps/target-6-23-17

from distutils.core import setup, Extension
import os

# Find include and lib paths based on whether or not they sourced their PYBOMBS install
if os.environ.get('PYBOMBS_PREFIX') is not None:
    print "Found PYBOMBS Install!"
    LIBRARY_PATH = [os.environ['LIBRARY_PATH'].split(':')[0]] # first element should be the right one... kind of messy
    INCLUDE_PATH = [os.environ['PYBOMBS_PREFIX'] + '/include']
else:
    print "Did not find PYBOMBS install (did you forget to source it?), using /usr/local"
    LIBRARY_PATH = ['/usr/local/lib']
    INCLUDE_PATH = ['/usr/local/include']  

pysdruhd = Extension('pysdruhd',
                    include_dirs = INCLUDE_PATH,
                    libraries = ['uhd'],
                    library_dirs = LIBRARY_PATH,
                    runtime_library_dirs = LIBRARY_PATH,
                    extra_compile_args=["-fPIC"], # tells GCC to generate position-independent code, without this I get an undefined symbol error
                    sources = ['pysdruhd.c'])

setup(name = 'pysdruhd',
      version = '1.0',
      description = 'C python extension that wraps UHD in a friendly way',
      author = 'N West',
      author_email = 'xxxxx',
      url = 'https://github.com/pysdr/pysdruhd',
      long_description = '''
This is a C python extension that wraps UHD in a friendly way. 
The goal here is to make a UHD interface that doesn't get in the way and feels like python. 
The purpose of this is not to provide a 1:1 API exposure to UHD, but to make it easy to do the things I'm interested in doing from python.
''',
      ext_modules = [pysdruhd])
