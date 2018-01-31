from distutils.core import setup

setup(name='pysdruhd',
      version='0.0',
      description='A Python Wrapper for UHD',
      author='Nathan West',
      author_email='nate.ewest@gmail.com',
      package_dir={'': '${CMAKE_CURRENT_SOURCE_DIR}' },
      packages=['pysdruhd']
)