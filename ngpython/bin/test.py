#!/usr/bin/env python

"""
Test script for new generation SPDLib Python bindings
"""
from __future__ import print_function
import sys
import spdpy2
import numpy

# open the file
spdfile = spdpy2.SPDFile(sys.argv[1])

pulses, points = spdfile.readBlock((20, 20, 120, 120))

print(pulses.dtype, pulses.size)
print(pulses)
print(points.dtype)
print(points, points.size)
print(pulses[pulses['startPtsIdx'] != pulses['endPtsIdx']])





