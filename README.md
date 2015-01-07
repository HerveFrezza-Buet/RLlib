# RLlib

C++ library for reinforcement learning

Published in JMLR

<a href="http://www.jmlr.org/papers/v14/frezza-buet13a.html">A C++ Template-Based Reinforcement Learning Library: Fitting the Code to the Mathematics</a>
Hervé Frezza-Buet, Matthieu Geist; JMLR, 14(Feb):625−628, 2013. 


# Installation

First, get the files.

``` 
git clone https://github.com/HerveFrezza-Buet/RLlib
``` 


The library consists of header files only. So you can put them somewhere your compiler can access them, and it is ok. Nevertheless, a cleaner install can be done as follows with cmake.

``` 
cd <your_path_here>/RLlib
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr
sudo make install
``` 


# Documentation

Read examples in the suggested order. Doxygen pages are accessible from the RLlib/html/index.html file.



