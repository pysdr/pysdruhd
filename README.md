
# pysdruhd

This is a C python extension that wraps UHD in a friendly way. The goal here is to make a UHD interface that doesn't get in the way and feels like python. The purpose of this is not to provide a 1:1 API exposure to UHD, but to make it easy to do the things I'm interested in doing from python.

## Building

You need python (currently I don't do much cmake-fu to detect python versions, so only 2 works), numpy, and UHD. You build like any other cmake project:

1. `git clone` `mkdir build` `cd build`
2. `cmake -DCMAKE_INSTALL_PREFIX=~/apps/target-4-5-17 ../` replace ~/apps/target-4-5-17 with your PyBOMBS target directory
3. `make`
4. `sudo make install`

## Status

There's a ton of features missing and it's not so graceful when things fail. You can either wait for that to improve, try to fix and add things, or ignore the project.

## Copying and using

This is licensed with GPLv3. You should follow those rules, but you should also be a nice person and share any changes or improvements. Additionally if you make something cool, at least publish a blog post or paper in an open access journal about it (and ideally share code).
