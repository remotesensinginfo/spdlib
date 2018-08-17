/*
 *  spdpy.cpp
 *
 *  Functions which provide access to SPD/UPD files
 *  from within Python.
 *
 *  Created by Pete Bunting on 26/02/2011.
 *  Copyright 2011 SPDLib. All rights reserved.
 *
 *  This file is part of SPDLib.
 *
 *  SPDLib is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  SPDLib is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with SPDLib.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SPDPy_H
#define SPDPy_H

#include <iostream>
#include <string>
#include <list>
#include <vector>

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/list.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/docstring_options.hpp>

#include <boost/cstdint.hpp>

#include "spd/SPDCommon.h"
#include "spd/SPDFile.h"
#include "spd/SPDFileReader.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDException.h"
#include "spd/SPDFileIncrementalReader.h"

#include "spdpy_common.h"
#include "spdpy_docs.h"
#include "spdpy_spdwriter.h"
#include "spdpy_updwriter.h"

namespace spdlib_py
{

    void printSPDFile(std::string filepath) throw(spdlib::SPDException)
    {
        try
        {
            spdlib::SPDFile *spdFile = new spdlib::SPDFile(filepath);
            spdlib::SPDFileReader reader;
            reader.readHeaderInfo(filepath, spdFile);
            cout << spdFile << endl;
            delete spdFile;
        }
        catch(spdlib::SPDException &e)
        {
            throw e;
        }
    }

    spdlib::SPDFile openSPDFileHeader(std::string filepath) throw(spdlib::SPDException)
    {
        try
        {
            spdlib::SPDFile spdFile(filepath);
            spdlib::SPDFileReader reader;
            reader.readHeaderInfo(filepath, &spdFile);
            return spdFile;
        }
        catch(spdlib::SPDException &e)
        {
            throw e;
        }
    }

    spdlib::SPDFile createSPDFile(std::string filepath) throw(spdlib::SPDException)
    {
       spdlib::SPDFile spdFile(filepath);
       return spdFile;
    }

    SPDPulsePy createSPDPulsePy() throw(spdlib::SPDException)
    {
       SPDPulsePy pulse = SPDPulsePy();
       return pulse;
    }

    SPDPointPy createSPDPointPy() throw(spdlib::SPDException)
    {
       SPDPointPy point = SPDPointPy();
       return point;
    }

    void readSPDHeaderRow(spdlib::SPDFile spdFile, uint_fast32_t row, boost::python::list binOffsets, boost::python::list numPulsesInBins) throw(spdlib::SPDException)
    {
        try
        {
            // Create C++ arrays
            unsigned long long *binOffsetsArr = new unsigned long long[spdFile.getNumberBinsX()];
            unsigned long *numPtsInBinArr = new unsigned long[spdFile.getNumberBinsX()];

            // Create incremental reader
            spdlib::SPDFileIncrementalReader incReader;

            // Open incremental UPD reader
            incReader.open(&spdFile);

            // Read SPD header values from SPD file
            incReader.readRefHeaderRow(row, binOffsetsArr, numPtsInBinArr);

            // close the incremental reader
            incReader.close();

            // Copy values into lists.
            for(boost::uint_fast64_t i = 0; i < spdFile.getNumberBinsX(); ++i)
            {
                binOffsets.append(binOffsetsArr[i]);
                numPulsesInBins.append(numPtsInBinArr[i]);
            }

            // Delete pulses list.
            delete[] binOffsetsArr;
            delete[] numPtsInBinArr;
        }
        catch(spdlib::SPDException &e)
        {
            throw e;
        }
    }

    boost::python::list readSPDPulsesRow(spdlib::SPDFile spdFile, boost::uint_fast32_t row) throw(spdlib::SPDException)
    {
        try
        {
            // Create C++ list
            std::vector<spdlib::SPDPulse*> **pulses = new std::vector<spdlib::SPDPulse*>*[spdFile.getNumberBinsX()];
            for(boost::uint_fast64_t i = 0; i < spdFile.getNumberBinsX(); ++i)
            {
                pulses[i] = new std::vector<spdlib::SPDPulse*>();
            }

            // Create incremental reader
            spdlib::SPDFileIncrementalReader incReader;

            // Open incremental UPD reader
            incReader.open(&spdFile);

            // // Read SPD Pulse data from SPD file
            incReader.readPulseDataRow(row, pulses);

            // close the incremental reader
            incReader.close();

            // Convert to a python list
            boost::python::list outPulses;

            // Copy values into lists.
            for(boost::uint_fast64_t i = 0; i < spdFile.getNumberBinsX(); ++i)
            {
                // Convert to a python list
                boost::python::list tmpPulses;
                convertVector2PyList(pulses[i], &tmpPulses);
                delete pulses[i];
                outPulses.append(tmpPulses);
            }
            delete[] pulses;

            return outPulses;
        }
        catch(spdlib::SPDException &e)
        {
            throw e;
        }
    }

    boost::python::list readSPDPulsesRowCols(spdlib::SPDFile spdFile, boost::uint_fast32_t row, boost::uint_fast32_t startCol, boost::uint_fast32_t endCol) throw(spdlib::SPDException)
    {
        try
        {
            // Create C++ list
            std::vector<spdlib::SPDPulse*> *pulses = new std::vector<spdlib::SPDPulse*>();

            // Create incremental reader
            spdlib::SPDFileIncrementalReader incReader;

            // Open incremental UPD reader
            incReader.open(&spdFile);

            // // Read SPD Pulse data from SPD file
            incReader.readPulseData(pulses, row, startCol, endCol);

            // close the incremental reader
            incReader.close();

            // Convert to a python list
            boost::python::list outPulses;
            convertVector2PyList(pulses, &outPulses);

            // Delete pulses list.
            delete pulses;

            return outPulses;
        }
        catch(spdlib::SPDException &e)
        {
            throw e;
        }
    }

    boost::python::list readSPDPulsesOffset(spdlib::SPDFile spdFile, boost::uint_fast64_t offset, boost::uint_fast64_t numPulses) throw(spdlib::SPDException)
    {
        try
        {
            // Create C++ list
            std::vector<spdlib::SPDPulse*> *pulses = new std::vector<spdlib::SPDPulse*>();

            // Create incremental reader
            spdlib::SPDFileIncrementalReader incReader;

            // Open incremental UPD reader
            incReader.open(&spdFile);

            // // Read SPD Pulse data from SPD file
            incReader.readPulseData(pulses, offset, numPulses);

            // close the incremental reader
            incReader.close();

            // Convert to a python list
            boost::python::list outPulses;
            convertVector2PyList(pulses, &outPulses);

            // Delete pulses list.
            delete pulses;

            return outPulses;
        }
        catch(spdlib::SPDException &e)
        {
            throw e;
        }
    }

    // BBOX = [startX, startY, endX, endY]
    boost::python::list readSPDPulsesIntoBlock(spdlib::SPDFile spdFile, boost::python::list bbox) throw(spdlib::SPDException)
    {
        try
        {
            //
            boost::uint_fast32_t *bboxArr = new boost::uint_fast32_t[4];
            bboxArr[0] = boost::python::extract<boost::uint_fast32_t>(bbox[0])();
            bboxArr[1] = boost::python::extract<boost::uint_fast32_t>(bbox[1])();
            bboxArr[2] = boost::python::extract<boost::uint_fast32_t>(bbox[2])();
            bboxArr[3] = boost::python::extract<boost::uint_fast32_t>(bbox[3])();

            boost::uint_fast32_t xBlockSize = bboxArr[2] - bboxArr[0];
            boost::uint_fast32_t yBlockSize = bboxArr[3] - bboxArr[1];

            // Create C++ list
            std::vector<spdlib::SPDPulse*> ***pulses = new std::vector<spdlib::SPDPulse*>**[yBlockSize];
            for(boost::uint_fast32_t i = 0; i < yBlockSize; ++i)
            {
                pulses[i] = new std::vector<spdlib::SPDPulse*>*[xBlockSize];
                for(boost::uint_fast32_t j = 0; j < xBlockSize; ++j)
                {
                    pulses[i][j] = new std::vector<spdlib::SPDPulse*>();
                }
            }

            // Create incremental reader
            spdlib::SPDFileIncrementalReader incReader;

            // Open incremental UPD reader
            incReader.open(&spdFile);

            // Read SPD Pulse data from SPD file
            incReader.readPulseDataBlock(pulses, bboxArr);

            // close the incremental reader
            incReader.close();

            // Convert to a python list
            boost::python::list outPulses;
            for(boost::uint_fast32_t i = 0; i < yBlockSize; ++i)
            {
                boost::python::list tmpPulsesRow;
                for(boost::uint_fast32_t j = 0; j < xBlockSize; ++j)
                {
                    boost::python::list tmpPulses;
                    convertVector2PyList(pulses[i][j], &tmpPulses);
                    delete pulses[i][j];
                    tmpPulsesRow.append(tmpPulses);
                }
                outPulses.append(tmpPulsesRow);
            }
            delete[] pulses;
            delete[] bboxArr;

            return outPulses;
        }
        catch(spdlib::SPDException &e)
        {
            throw e;
        }
    }

    boost::python::list readSPDPulsesIntoBlockList(spdlib::SPDFile spdFile, boost::python::list bbox) throw(spdlib::SPDException)
    {
        try
        {
            //
            uint_fast32_t *bboxArr = new uint_fast32_t[4];
            bboxArr[0] = boost::python::extract<uint_fast32_t>(bbox[0])();
            bboxArr[1] = boost::python::extract<uint_fast32_t>(bbox[1])();
            bboxArr[2] = boost::python::extract<uint_fast32_t>(bbox[2])();
            bboxArr[3] = boost::python::extract<uint_fast32_t>(bbox[3])();

            // Create C++ list
            std::vector<spdlib::SPDPulse*> *pulses = new std::vector<spdlib::SPDPulse*>();

            // Create incremental reader
            spdlib::SPDFileIncrementalReader incReader;

            // Open incremental UPD reader
            incReader.open(&spdFile);

            // Read SPD Pulse data from SPD file
            incReader.readPulseDataBlock(pulses, bboxArr);

            // close the incremental reader
            incReader.close();

            // Convert to a python list
            boost::python::list outPulses;
            convertVector2PyList(pulses, &outPulses);

            // Delete pulses list.
            delete pulses;
            delete[] bboxArr;

            return outPulses;
        }
        catch(spdlib::SPDException &e)
        {
            throw e;
        }
    }

    SPDFile copySPDFileAttributes(SPDFile spdFile, string newSPDFilePath)
    {
        SPDFile spdFileOut("");
        spdFile.copyAttributesTo(&spdFileOut);
        return spdFileOut;
    }

    boost::python::list getSPDFileWavelengths(SPDFile spdFile)
    {
        boost::python::list wavelengths;
        std::vector<float> *cWavelengths = spdFile.getWavelengths();
        for(std::vector<float>::iterator iterFloats = cWavelengths->begin(); iterFloats != cWavelengths->end(); ++iterFloats)
        {
            wavelengths.append(*iterFloats);
        }
        return wavelengths;
    }

    boost::python::list getSPDFileBandwidths(SPDFile spdFile)
    {
        boost::python::list bandwidths;
        std::vector<float>* cBandwidths = spdFile.getBandwidths();
        for(std::vector<float>::iterator iterFloats = cBandwidths->begin(); iterFloats != cBandwidths->end(); ++iterFloats)
        {
            bandwidths.append(*iterFloats);
        }
        return bandwidths;
    }

    void setSPDFileWavelengthsAndBandwidths(SPDFile spdFile, boost::python::list wavelengths, boost::python::list bandwidths) throw(SPDException)
    {
        std::vector<float> *cWavelengths = spdFile.getWavelengths();
        cWavelengths->clear();
        std::vector<float>* cBandwidths = spdFile.getBandwidths();
        cBandwidths->clear();

        boost::uint_fast64_t numWavelengths = len(wavelengths);
        boost::uint_fast64_t numBandwidths = len(bandwidths);

        if(numWavelengths != numBandwidths)
        {
            throw SPDException("The number of wavelengths and bandwidths needs to be the same.");
        }

        float val = 0;
        for(boost::uint_fast64_t i = 0; i < numWavelengths; ++i)
        {
            val = boost::python::extract<float>(wavelengths[i])();
            cWavelengths->push_back(val);

            val = boost::python::extract<float>(bandwidths[i])();
            cBandwidths->push_back(val);
        }
    }


    BOOST_PYTHON_MODULE(spdpy)
    {
        using namespace boost::python;

        docstring_options(true);

        register_exception_translator<SPDException>(&translate);

        def("printSPDFile", printSPDFile, "Print a selection of the SPDFile header parameters.");
        def("copySPDFileAttributes", copySPDFileAttributes, "Copy the SPDFile header file attributes to another SPDFile.");
        def("openSPDFileHeader", openSPDFileHeader, "Read SPDFile header.");
        def("readSPDHeaderRow", readSPDHeaderRow, "Read the index header arrays (Number of Pulses and pulses offset.");
        def("readSPDPulsesRow", readSPDPulsesRow, "Read a row of pulses from the SPD File.");
        def("readSPDPulsesRowCols", readSPDPulsesRowCols, "Read pulses between start column and end column within row from the SPD File.");
        def("readSPDPulsesOffset", readSPDPulsesOffset, "Read number of pulses after the offset number of pulses.");
        def("readSPDPulsesIntoBlock", readSPDPulsesIntoBlock, "Read a block of pulses from the file defined in bin space using bounding box");
        def("readSPDPulsesIntoBlockList", readSPDPulsesIntoBlockList, "Read a block of pulses from the file defined in bin space using bounding box into a single list.");
        def("createSPDFile", createSPDFile, "Create a new SPDFile.");
        def("createSPDPulsePy", createSPDPulsePy, "Create a new SPDPulse.");
        def("createSPDPointPy", createSPDPointPy, "Create a new SPDPoint.");
        def("getSPDFileWavelengths", getSPDFileWavelengths, "Get wavelengths list from SPDFile header.");
        def("getSPDFileBandwidths", getSPDFileBandwidths, "Get bandwidths list from SPDFile header.");
        def("setSPDFileWavelengthsAndBandwidths", setSPDFileWavelengthsAndBandwidths, "Set wavelengths and bandwidths list in SPDFile header.");


        class_<SPDPointPy>("SPDPointPy")
            .def_readwrite("returnID", &SPDPointPy::returnID)
            .def_readwrite("gpsTime", &SPDPointPy::gpsTime)
            .def_readwrite("x", &SPDPointPy::x)
            .def_readwrite("y", &SPDPointPy::y)
            .def_readwrite("z", &SPDPointPy::z)
            .def_readwrite("height", &SPDPointPy::height)
            .def_readwrite("range", &SPDPointPy::range)
            .def_readwrite("amplitudeReturn", &SPDPointPy::amplitudeReturn)
            .def_readwrite("widthReturn", &SPDPointPy::widthReturn)
            .def_readwrite("red", &SPDPointPy::red)
            .def_readwrite("green", &SPDPointPy::green)
            .def_readwrite("blue", &SPDPointPy::blue)
            .def_readwrite("classification", &SPDPointPy::classification)
            .def_readwrite("user", &SPDPointPy::user)
            .def_readwrite("modelKeyPoint", &SPDPointPy::modelKeyPoint)
            .def_readwrite("lowPoint", &SPDPointPy::lowPoint)
            .def_readwrite("overlap", &SPDPointPy::overlap)
            .def_readwrite("ignore", &SPDPointPy::ignore)
            .def_readwrite("wavePacketDescIdx", &SPDPointPy::wavePacketDescIdx)
            .def_readwrite("waveformOffset", &SPDPointPy::waveformOffset);

        class_<SPDPulsePy>("SPDPulsePy")
            .def_readwrite("pulseID", &SPDPulsePy::pulseID)
            .def_readwrite("gpsTime", &SPDPulsePy::gpsTime)
            .def_readwrite("x0", &SPDPulsePy::x0)
            .def_readwrite("y0", &SPDPulsePy::y0)
            .def_readwrite("z0", &SPDPulsePy::z0)
            .def_readwrite("h0", &SPDPulsePy::h0)
            .def_readwrite("xIdx", &SPDPulsePy::xIdx)
            .def_readwrite("yIdx", &SPDPulsePy::yIdx)
            .def_readwrite("numberOfReturns", &SPDPulsePy::numberOfReturns)
            .def_readwrite("azimuth", &SPDPulsePy::azimuth)
            .def_readwrite("zenith", &SPDPulsePy::zenith)
            .def_readwrite("pts", &SPDPulsePy::pts)
            .def_readwrite("transmitted", &SPDPulsePy::transmitted)
            .def_readwrite("received", &SPDPulsePy::received)
            .def_readwrite("numOfTransmittedBins", &SPDPulsePy::numOfTransmittedBins)
            .def_readwrite("numOfReceivedBins", &SPDPulsePy::numOfReceivedBins)
            .def_readwrite("rangeToWaveformStart", &SPDPulsePy::rangeToWaveformStart)
            .def_readwrite("amplitudePulse", &SPDPulsePy::amplitudePulse)
            .def_readwrite("widthPulse", &SPDPulsePy::widthPulse)
            .def_readwrite("user", &SPDPulsePy::user)
            .def_readwrite("sourceID", &SPDPulsePy::sourceID)
            .def_readwrite("scanline", &SPDPulsePy::scanline)
            .def_readwrite("scanlineIdx", &SPDPulsePy::scanlineIdx)
            .def_readwrite("edgeFlightLineFlag", &SPDPulsePy::edgeFlightLineFlag)
            .def_readwrite("scanDirectionFlag", &SPDPulsePy::scanDirectionFlag)
            .def_readwrite("scanAngleRank", &SPDPulsePy::scanAngleRank)
            .def_readwrite("receiveWaveNoiseThreshold", &SPDPulsePy::receiveWaveNoiseThreshold)
            .def_readwrite("transWaveNoiseThres", &SPDPulsePy::transWaveNoiseThres)
            .def_readwrite("receiveWaveGain", &SPDPulsePy::receiveWaveGain)
            .def_readwrite("receiveWaveOffset", &SPDPulsePy::receiveWaveOffset)
            .def_readwrite("transWaveGain", &SPDPulsePy::transWaveGain)
            .def_readwrite("transWaveOffset", &SPDPulsePy::transWaveOffset)
            .def_readwrite("wavelength", &SPDPulsePy::wavelength);

        class_<spdlib::SPDFile>("SPDFile", SPDFILE_CLASS_DOC.c_str(), init<std::string>())
            .def_readonly("filePath", &spdlib::SPDFile::getFilePath, "The file name and location of the system.")
            .def_readonly("filetype", &spdlib::SPDFile::getFileType)
            .def_readonly("spatialReference", &spdlib::SPDFile::getSpatialReference)
            .def_readonly("indexType", &spdlib::SPDFile::getIndexType)
            .def_readonly("discretePtDefined", &spdlib::SPDFile::getDiscretePtDefined)
            .def_readonly("decomposedPtDefined", &spdlib::SPDFile::getDecomposedPtDefined)
            .def_readonly("transWaveformDefined", &spdlib::SPDFile::getTransWaveformDefined)
            .def_readonly("receiveWaveformDefined", &spdlib::SPDFile::getReceiveWaveformDefined)
            .def_readonly("majorSPDVersion", &spdlib::SPDFile::getMajorSPDVersion)
            .def_readonly("minorSPDVersion", &spdlib::SPDFile::getMinorSPDVersion)
            .def_readonly("pointVersion", &spdlib::SPDFile::getPointVersion)
            .def_readonly("pulseVersion", &spdlib::SPDFile::getPulseVersion)
            .def_readonly("generatingSoftware", &spdlib::SPDFile::getGeneratingSoftware)
            .def_readonly("systemIdentifier", &spdlib::SPDFile::getSystemIdentifier)
            .def_readonly("fileSignature", &spdlib::SPDFile::getFileSignature)
            .def_readonly("yearOfCreation", &spdlib::SPDFile::getYearOfCreation)
            .def_readonly("monthOfCreation", &spdlib::SPDFile::getMonthOfCreation)
            .def_readonly("dayOfCreation", &spdlib::SPDFile::getDayOfCreation)
            .def_readonly("hourOfCreation", &spdlib::SPDFile::getHourOfCreation)
            .def_readonly("minuteOfCreation", &spdlib::SPDFile::getMinuteOfCreation)
            .def_readonly("secondOfCreation", &spdlib::SPDFile::getSecondOfCreation)
            .def_readonly("yearOfCapture", &spdlib::SPDFile::getYearOfCapture)
            .def_readonly("monthOfCapture", &spdlib::SPDFile::getMonthOfCapture)
            .def_readonly("dayOfCapture", &spdlib::SPDFile::getDayOfCapture)
            .def_readonly("hourOfCapture", &spdlib::SPDFile::getHourOfCapture)
            .def_readonly("minuteOfCapture", &spdlib::SPDFile::getMinuteOfCapture)
            .def_readonly("secondOfCapture", &spdlib::SPDFile::getSecondOfCapture)
            .def_readonly("numPts", &spdlib::SPDFile::getNumberOfPoints)
            .def_readonly("numPulses", &spdlib::SPDFile::getNumberOfPulses)
            .def_readonly("userMetaField", &spdlib::SPDFile::getUserMetaField)
            .def_readonly("xMin", &spdlib::SPDFile::getXMin)
            .def_readonly("xMax", &spdlib::SPDFile::getXMax)
            .def_readonly("yMin", &spdlib::SPDFile::getYMin)
            .def_readonly("yMax", &spdlib::SPDFile::getYMax)
            .def_readonly("zMin", &spdlib::SPDFile::getZMin)
            .def_readonly("zMax", &spdlib::SPDFile::getZMax)
            .def_readonly("zenithMin", &spdlib::SPDFile::getZenithMin)
            .def_readonly("zenithMax", &spdlib::SPDFile::getZenithMax)
            .def_readonly("azimuthMin", &spdlib::SPDFile::getAzimuthMin)
            .def_readonly("azimuthMax", &spdlib::SPDFile::getAzimuthMax)
            .def_readonly("rangeMin", &spdlib::SPDFile::getRangeMin)
            .def_readonly("rangeMax", &spdlib::SPDFile::getRangeMax)
            .def_readonly("scanlineMin", &spdlib::SPDFile::getScanlineMin)
            .def_readonly("scanlineMax", &spdlib::SPDFile::getScanlineMax)
            .def_readonly("scanlineIdxMin", &spdlib::SPDFile::getScanlineIdxMin)
            .def_readonly("scanlineIdxMax", &spdlib::SPDFile::getScanlineIdxMax)
            .def_readonly("binSize", &spdlib::SPDFile::getBinSize)
            .def_readonly("numBinsX", &spdlib::SPDFile::getNumberBinsX)
            .def_readonly("numBinsY", &spdlib::SPDFile::getNumberBinsY)
            .def_readonly("numOfwavelengths", &spdlib::SPDFile::getNumOfWavelengths)
            .def_readonly("pulseRepetitionFreq", &spdlib::SPDFile::getPulseRepetitionFreq)
            .def_readonly("beamDivergence", &spdlib::SPDFile::getBeamDivergence)
            .def_readonly("sensorHeight", &spdlib::SPDFile::getSensorHeight)
            .def_readonly("footprint", &spdlib::SPDFile::getFootprint)
            .def_readonly("maxScanAngle", &spdlib::SPDFile::getMaxScanAngle)
            .def_readonly("rgbDefined", &spdlib::SPDFile::getRGBDefined)
            .def_readonly("pulseBlockSize", &spdlib::SPDFile::getPulseBlockSize)
            .def_readonly("pointBlockSize", &spdlib::SPDFile::getPointBlockSize)
            .def_readonly("receivedBlockSize", &spdlib::SPDFile::getReceivedBlockSize)
            .def_readonly("transmittedBlockSize", &spdlib::SPDFile::getTransmittedBlockSize)
            .def_readonly("waveformBitRes", &spdlib::SPDFile::getWaveformBitRes)
            .def_readonly("temporalBinSpacing", &spdlib::SPDFile::getTemporalBinSpacing)
            .def_readonly("returnNumsSynGen", &spdlib::SPDFile::getReturnNumsSynGen)
            .def_readonly("heightDefined", &spdlib::SPDFile::getHeightDefined)
            .def_readonly("sensorSpeed", &spdlib::SPDFile::getSensorSpeed)
            .def_readonly("sensorScanRate", &spdlib::SPDFile::getSensorScanRate)
            .def_readonly("pointDensity", &spdlib::SPDFile::getPointDensity)
            .def_readonly("pulseDensity", &spdlib::SPDFile::getPulseDensity)
            .def_readonly("pulseCrossTrackSpacing", &spdlib::SPDFile::getPulseCrossTrackSpacing)
            .def_readonly("pulseAlongTrackSpacing", &spdlib::SPDFile::getPulseAlongTrackSpacing)
            .def_readonly("originDefined", &spdlib::SPDFile::getOriginDefined)
            .def_readonly("pulseAngularSpacingAzimuth", &spdlib::SPDFile::getPulseAngularSpacingAzimuth)
            .def_readonly("pulseAngularSpacingZenith", &spdlib::SPDFile::getPulseAngularSpacingZenith)
            .def_readonly("pulseIdxMethod", &spdlib::SPDFile::getPulseIdxMethod)
            .def_readonly("sensorApertureSize", &spdlib::SPDFile::getSensorApertureSize)
            .def_readonly("pulseEnergy", &spdlib::SPDFile::getPulseEnergy)
            .def_readonly("fieldOfView", &spdlib::SPDFile::getFieldOfView)
            .def("setFilePath", &spdlib::SPDFile::setFilePath)
            .def("setFiletype", &spdlib::SPDFile::setFileType)
            .def("setSpatialReference", &spdlib::SPDFile::setSpatialReference)
            .def("setIndexType", &spdlib::SPDFile::setIndexType)
            .def("setDiscretePtDefined", &spdlib::SPDFile::setDiscretePtDefined)
            .def("setDecomposedPtDefined", &spdlib::SPDFile::setDecomposedPtDefined)
            .def("setTransWaveformDefined", &spdlib::SPDFile::setTransWaveformDefined)
            .def("setReceiveWaveformDefined", &spdlib::SPDFile::setReceiveWaveformDefined)
            .def("setMajorSPDVersion", &spdlib::SPDFile::setMajorSPDVersion)
            .def("setMinorSPDVersion", &spdlib::SPDFile::setMinorSPDVersion)
            .def("setPointVersion", &spdlib::SPDFile::setPointVersion)
            .def("setPulseVersion", &spdlib::SPDFile::setPulseVersion)
            .def("setGeneratingSoftware", &spdlib::SPDFile::setGeneratingSoftware)
            .def("setSystemIdentifier", &spdlib::SPDFile::setSystemIdentifier)
            .def("setFileSignature", &spdlib::SPDFile::setFileSignature)
            .def("setYearOfCreation", &spdlib::SPDFile::setYearOfCreation)
            .def("setMonthOfCreation", &spdlib::SPDFile::setMonthOfCreation)
            .def("setDayOfCreation", &spdlib::SPDFile::setDayOfCreation)
            .def("setHourOfCreation", &spdlib::SPDFile::setHourOfCreation)
            .def("setMinuteOfCreation", &spdlib::SPDFile::setMinuteOfCreation)
            .def("setSecondOfCreation", &spdlib::SPDFile::setSecondOfCreation)
            .def("setYearOfCapture", &spdlib::SPDFile::setYearOfCapture)
            .def("setMonthOfCapture", &spdlib::SPDFile::setMonthOfCapture)
            .def("setDayOfCapture", &spdlib::SPDFile::setDayOfCapture)
            .def("setHourOfCapture", &spdlib::SPDFile::setHourOfCapture)
            .def("setMinuteOfCapture", &spdlib::SPDFile::setMinuteOfCapture)
            .def("setSecondOfCapture", &spdlib::SPDFile::setSecondOfCapture)
            .def("setNumPts", &spdlib::SPDFile::setNumberOfPoints)
            .def("setNumPulses", &spdlib::SPDFile::setNumberOfPulses)
            .def("setUserMetaField", &spdlib::SPDFile::setUserMetaField)
            .def("setXMin", &spdlib::SPDFile::setXMin)
            .def("setXMax", &spdlib::SPDFile::setXMax)
            .def("setYMin", &spdlib::SPDFile::setYMin)
            .def("setYMax", &spdlib::SPDFile::setYMax)
            .def("setZMin", &spdlib::SPDFile::setZMin)
            .def("setZMax", &spdlib::SPDFile::setZMax)
            .def("setZenithMin", &spdlib::SPDFile::setZenithMin)
            .def("setZenithMax", &spdlib::SPDFile::setZenithMax)
            .def("setAzimuthMin", &spdlib::SPDFile::setAzimuthMin)
            .def("setAzimuthMax", &spdlib::SPDFile::setAzimuthMax)
            .def("setRangeMin", &spdlib::SPDFile::setRangeMin)
            .def("setRangeMax", &spdlib::SPDFile::setRangeMax)
            .def("setScanlineMin", &spdlib::SPDFile::setScanlineMin)
            .def("setScanlineMax", &spdlib::SPDFile::setScanlineMax)
            .def("setScanlineIdxMin", &spdlib::SPDFile::setScanlineIdxMin)
            .def("setScanlineIdxMax", &spdlib::SPDFile::setScanlineIdxMax)
            .def("setBinSize", &spdlib::SPDFile::setBinSize)
            .def("setNumBinsX", &spdlib::SPDFile::setNumberBinsX)
            .def("setNumBinsY", &spdlib::SPDFile::setNumberBinsY)
            .def("setNumOfWavelengths", &spdlib::SPDFile::setNumOfWavelengths)
            .def("setPulseRepetitionFreq", &spdlib::SPDFile::setPulseRepetitionFreq)
            .def("setBeamDivergence", &spdlib::SPDFile::setBeamDivergence)
            .def("setSensorHeight", &spdlib::SPDFile::setSensorHeight)
            .def("setFootprint", &spdlib::SPDFile::setFootprint)
            .def("setMaxScanAngle", &spdlib::SPDFile::setMaxScanAngle)
            .def("setRgbDefined", &spdlib::SPDFile::setRGBDefined)
            .def("setPulseBlockSize", &spdlib::SPDFile::setPulseBlockSize)
            .def("setPointBlockSize", &spdlib::SPDFile::setPointBlockSize)
            .def("setReceivedBlockSize", &spdlib::SPDFile::setReceivedBlockSize)
            .def("setTransmittedBlockSize", &spdlib::SPDFile::setTransmittedBlockSize)
            .def("setWaveformBitRes", &spdlib::SPDFile::setWaveformBitRes)
            .def("setTemporalBinSpacing", &spdlib::SPDFile::setTemporalBinSpacing)
            .def("setReturnNumsSynGen", &spdlib::SPDFile::setReturnNumsSynGen)
            .def("setHeightDefined", &spdlib::SPDFile::setHeightDefined)
            .def("setSensorSpeed", &spdlib::SPDFile::setSensorSpeed)
            .def("setSensorScanRate", &spdlib::SPDFile::setSensorScanRate)
            .def("setPointDensity", &spdlib::SPDFile::setPointDensity)
            .def("setPulseDensity", &spdlib::SPDFile::setPulseDensity)
            .def("setPulseCrossTrackSpacing", &spdlib::SPDFile::setPulseCrossTrackSpacing)
            .def("setPulseAlongTrackSpacing", &spdlib::SPDFile::setPulseAlongTrackSpacing)
            .def("setOriginDefined", &spdlib::SPDFile::setOriginDefined)
            .def("setPulseAngularSpacingAzimuth", &spdlib::SPDFile::setPulseAngularSpacingAzimuth)
            .def("setPulseAngularSpacingZenith", &spdlib::SPDFile::setPulseAngularSpacingZenith)
            .def("setPulseIdxMethod", &spdlib::SPDFile::setPulseIdxMethod)
            .def("setSensorApertureSize", &spdlib::SPDFile::setSensorApertureSize)
            .def("setPulseEnergy", &spdlib::SPDFile::setPulseEnergy)
            .def("setFieldOfView", &spdlib::SPDFile::setFieldOfView);

        class_<SPDPyNoIdxWriter>("SPDPyNoIdxWriter", SPDNONIDXWRITER_CLASS_DOC.c_str())
            .def("open", &SPDPyNoIdxWriter::open, "Open writer with SPDFile.")
            .def("writeData", &SPDPyNoIdxWriter::writeData, "Write SPDPulses to SPDFile.")
            .def("close", &SPDPyNoIdxWriter::close, "Close SPDFile.");

        class_<SPDPySeqWriter>("SPDPySeqWriter", SPDSEQIDXWRITER_CLASS_DOC.c_str())
            .def("open", &SPDPySeqWriter::open, "Open writer with SPDFile.")
            .def("writeDataColumn", &SPDPySeqWriter::writeDataColumn, "Write a column of SPDPulses to SPDFile.")
            .def("writeDataRow", &SPDPySeqWriter::writeDataRow, "Write a row of SPDPulses to SPDFile.")
            .def("close", &SPDPySeqWriter::close, "Close SPDFile.");

        class_<SPDPyNonSeqWriter>("SPDPyNonSeqWriter", SPDNONSEQIDXWRITER_CLASS_DOC.c_str())
            .def("open", &SPDPyNonSeqWriter::open, "Open writer with SPDFile.")
            .def("writeDataColumn", &SPDPyNonSeqWriter::writeDataColumn, "Write a column of SPDPulses to SPDFile.")
            .def("writeDataRow", &SPDPyNonSeqWriter::writeDataRow, "Write a row of SPDPulses to SPDFile.")
            .def("close", &SPDPyNonSeqWriter::close, "Close SPDFile.");
    }
}

#endif

