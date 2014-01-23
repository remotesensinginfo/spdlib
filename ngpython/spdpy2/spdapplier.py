#
#  spdapplier.py
#  SPDLIB
#
#  Created by Sam Gillingham on 22/01/2014.
#  Copyright 2013 SPDLib. All rights reserved.
#
#  This file is part of SPDLib.
#
#  SPDLib is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  SPDLib is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with SPDLib.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Applier interface for SPD processing
"""

# the C++ code that actually does the apply
from ._spdpy2 import blockProcessor

class ApplierControls(object):
    """
    Controls for the operation of the apply() function
    
    This object starts with default values for all controls, and 
    has methods for setting each of them to something else. 
    
    Attributes are:
        overlap                 The size in bins of the overlap
        blockXSize              The number of bins in the X direction 
        blockYSize              The number of bins in the Y direction
        printProgress           Whether to print progress or not
        keepMinExtent           ???
        processingResolution    If zero, this is the resolution of the dataset
        numImgBands             If writing to an image this is the number of bands
        gdalFormat              If writing to an image this is the GDAL format
    """
    def __init__(self):
        self.overlap = 1
        self.blockXSize = 10
        self.blockYSize = 10
        self.printProgress = True
        self.keepMinExtent = True
        self.processingResolution = 0; # native bin size of the dataset
        self.numImgBands = 1;
        self.gdalFormat = 'KEA'
        
    def setOverlap(self, overlap):
        "sets the overlap"
        self.overlap = overlap
        
    def setBlockXSize(self, size):
        "sets the X block size"
        self.blockXSize = size

    def setBlockYSize(self, size):
        "sets the Y block size"
        self.blockYSize = size
        
    def setPrintProgress(self, printProgress):
        "Whether to print progress or not"
        self.printProgress = printProgress
        
    def setKeepMinExtent(self, keepMinExtent):
        "Dunno"
        self.keepMinExtent = keepMinExtent
        
    def setProcessingResolution(self, res=0):
        """
        Sets the processing resolution
        set to zero for the native resolution of the dataset
        """
        self.processingResolution = res
        
    def setNumImgBands(self, bands):
        "Number of output image bands"
        self.numImgBands = bands
        
    def setGdalFormat(self, fmt):
        "name of GDAL driver for output image"
        self.gdalFormat = fmt
        
class OtherInputs(object):
    """
    Generic object to store any extra inputs and outputs used 
    inside the function being applied. This class was originally
    named for inputs, but in fact works just as well for outputs, 
    too. Any items stored on this will be persistent between 
    iterations of the block loop. 
    """
    pass
    
def applier(applyfn, inputSPDFile, inputImageFile=None, outputSPDFile=None,
        outputImageFile=None, controls=None, otherinputs=None):
    """
    Applies the applyfn over the input blocks and writes the outputs
        inputSPDFile is the path to the input SPD File. This must be supplied.
        inputImageFile is the path to the input Image File. This is optional.
        outputSPDFile is the path to the output SPD File. This is optional.
        outputImageFile is the path to the output Image File. This is optional.
        controls is an instance of ApplierControls. If not passed, default values are used.
        otherinputs is an instance of OtherInputs where additional things can be stored.
        
    The applyfn signature looks like this when otherinputs is None:
        applyfn(pulses, points, imagedata)
    And like this otherwise:
        applyfn(pulses, points, imagedata, otherinputs)
    
    imagedata will be None if both inputImageFile and outputImageFile are None
    """
    if controls is None:
        # use default ones
        controls = ApplierControls()
        
    if inputSPDFile is None:
        raise ValueError("inputSPDFile must be not None")
        
    blockProcessor(applyfn, inputSPDFile, inputImageFile, outputSPDFile, 
            outputImageFile, controls, otherinputs)
    