# The Sorted Pulse Data Library (SPDLib) #

SPDLib is a set of open source software tools for processing laser scanning data (i.e., LiDAR), including data captured from airborne and terrestrial platforms. A key feature of SPDLib is the ability to process and store full waveform datasets alongside traditional discrete return data.

## Installing ##

Binaries of SPDLib for Mac OS X and Linux are provided through [conda](http://conda.pydata.org/miniconda.html). After installing minconda, SPDLib and the required pre-requisites can be installed using:

```
conda create -n spdlib_env -c rios spdlib
source activate spdlib_env
```

If you need to use SPDLib under Windows the easiest way to do so is through a virtual machine, for more details see [here](https://spectraldifferences.wordpress.com/2014/09/24/installing-rsgislib-on-windows-through-a-virtual-machine).

## Documentation and Support ##

The documentation for SPDLib is available from: https://bitbucket.org/petebunting/spdlib-documentation

Tutorials are available from: https://sourceforge.net/projects/spdlib/files/Tutorials/

Support for SPDLib is provided through a mailing list: https://sourceforge.net/p/spdlib/mailman/spdlib-develop/
Please check through existing posts to see if there is already an answer to your question before emailing. To help us answer your question provide as much information as possible (SPDLib version and how you installed it, OS, things you have already tried to solve the problem etc.,).

We occasionally post tutorials on SPDLib to our [blog](https://spectraldifferences.wordpress.com/tag/spdlib/).

## Citing ##

If you use the SPD file format or SPDLib in a paper you should cite the following publications:

Bunting, P., Armston, J., Lucas, R. M., & Clewley, D. (2013). Sorted pulse data (SPD) library. Part I: A generic file format for LiDAR data from pulsed laser systems in terrestrial environments. Computers and Geosciences, 56, 197-206. doi:10.1016/j.cageo.2013.01.019

Bunting, P., Armston, J., Clewley, D., & Lucas, R. M. (2013). Sorted pulse data (SPD) library-Part II: A processing framework for LiDAR data from pulsed laser systems in terrestrial environments. Computers and Geosciences, 56, 207-215. doi:10.1016/j.cageo.2013.01.010