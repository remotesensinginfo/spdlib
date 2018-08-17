/*
 *  spdpy_common.h
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

#ifndef SPDPy_Common_H
#define SPDPy_Common_H

#include <iostream>
#include <string>
#include <list>
#include <vector>

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/python/exception_translator.hpp>

#include "spd/SPDCommon.h"
#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDException.h"

using namespace spdlib;
using namespace std;

namespace spdlib_py
{
    struct SPDPointPy
    {
        SPDPointPy():returnID(0), gpsTime(0), x(0), y(0), z(0), height(0), range(0), amplitudeReturn(0), widthReturn(0), red(0), green(0), blue(0), classification(SPD_UNCLASSIFIED), user(0), modelKeyPoint(SPD_FALSE), lowPoint(SPD_FALSE), overlap(SPD_FALSE), ignore(SPD_FALSE), wavePacketDescIdx(0), waveformOffset(0){};
        boost::uint_fast16_t returnID;
        boost::uint_fast64_t gpsTime;
        double x;
        double y;
        float z;
        float height;
        float range;
        float amplitudeReturn;
        float widthReturn;
        boost::uint_fast16_t red;
        boost::uint_fast16_t green;
        boost::uint_fast16_t blue;
        boost::uint_fast16_t classification;
        boost::uint_fast32_t user;
        boost::int_fast16_t modelKeyPoint;
        boost::int_fast16_t lowPoint;
        boost::int_fast16_t overlap;
        boost::int_fast16_t ignore;
        boost::int_fast16_t wavePacketDescIdx;
        boost::uint_fast32_t waveformOffset;
    };

    struct SPDPulsePy
    {
        SPDPulsePy():pulseID(0), gpsTime(0), x0(0), y0(0), z0(0), h0(0), xIdx(0), yIdx(0), azimuth(0), zenith(0), numberOfReturns(0), pts(), transmitted(), received(), numOfTransmittedBins(0), numOfReceivedBins(0), rangeToWaveformStart(0), amplitudePulse(0), widthPulse(0),  user(0), sourceID(0), edgeFlightLineFlag(0), scanDirectionFlag(0), scanAngleRank(0), receiveWaveNoiseThreshold(0), transWaveNoiseThres(0), receiveWaveGain(1), receiveWaveOffset(0), transWaveGain(1), transWaveOffset(0), wavelength(0){};
        boost::uint_fast64_t pulseID;
        boost::uint_fast64_t gpsTime;
        double x0;
        double y0;
        float z0;
        float h0;
        double xIdx;
        double yIdx;
        float azimuth;
        float zenith;
        boost::uint_fast16_t numberOfReturns;
        boost::python::list pts;
        boost::python::list transmitted;
        boost::python::list received;
        boost::uint_fast16_t numOfTransmittedBins;
        boost::uint_fast16_t numOfReceivedBins;
        float rangeToWaveformStart;
        float amplitudePulse;
        float widthPulse;
        boost::uint_fast32_t user;
        boost::uint_fast16_t sourceID;
        boost::uint_fast32_t scanline;
        boost::uint_fast16_t scanlineIdx;
        boost::uint_fast16_t edgeFlightLineFlag;
        boost::uint_fast16_t scanDirectionFlag;
        float scanAngleRank;
        float receiveWaveNoiseThreshold;
        float transWaveNoiseThres;
        float receiveWaveGain;
        float receiveWaveOffset;
        float transWaveGain;
        float transWaveOffset;
        float wavelength;
    };

    inline void convertList2PyList(std::list<SPDPulse*> *pulses, boost::python::list *outPulses) throw(SPDException)
    {
        // Create iterators
        std::list<SPDPulse*>::iterator iterPulses;
        std::vector<SPDPoint*>::iterator iterPoints;

        // Iterate through pulses and copy to python list
        for(iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
        {
            // Create empty pulse
            SPDPulsePy pulsePy;

            // Copy pulse
            pulsePy.pulseID = (*iterPulses)->pulseID;
            pulsePy.gpsTime = (*iterPulses)->gpsTime;
            pulsePy.x0 = (*iterPulses)->x0;
            pulsePy.y0 = (*iterPulses)->y0;
            pulsePy.z0 = (*iterPulses)->z0;
            pulsePy.h0 = (*iterPulses)->h0;
            pulsePy.xIdx = (*iterPulses)->xIdx;
            pulsePy.yIdx = (*iterPulses)->yIdx;
            pulsePy.azimuth = (*iterPulses)->azimuth;
            pulsePy.zenith = (*iterPulses)->zenith;
            pulsePy.numberOfReturns = (*iterPulses)->numberOfReturns;
            pulsePy.numOfTransmittedBins = (*iterPulses)->numOfTransmittedBins;
            pulsePy.numOfReceivedBins = (*iterPulses)->numOfReceivedBins;
            pulsePy.rangeToWaveformStart = (*iterPulses)->rangeToWaveformStart;
            pulsePy.amplitudePulse = (*iterPulses)->amplitudePulse;
            pulsePy.widthPulse = (*iterPulses)->widthPulse;
            pulsePy.user = (*iterPulses)->user;
            pulsePy.sourceID = (*iterPulses)->sourceID;
            pulsePy.scanline = (*iterPulses)->scanline;
            pulsePy.scanlineIdx = (*iterPulses)->scanlineIdx;
            pulsePy.edgeFlightLineFlag = (*iterPulses)->edgeFlightLineFlag;
            pulsePy.scanDirectionFlag = (*iterPulses)->scanDirectionFlag;
            pulsePy.scanAngleRank = (*iterPulses)->scanAngleRank;
            pulsePy.receiveWaveNoiseThreshold = (*iterPulses)->receiveWaveNoiseThreshold;
            pulsePy.transWaveNoiseThres = (*iterPulses)->transWaveNoiseThres;
            pulsePy.receiveWaveGain = (*iterPulses)->receiveWaveGain;
            pulsePy.receiveWaveOffset = (*iterPulses)->receiveWaveOffset;
            pulsePy.transWaveGain = (*iterPulses)->transWaveGain;
            pulsePy.transWaveOffset = (*iterPulses)->transWaveOffset;
            pulsePy.wavelength = (*iterPulses)->wavelength;

            if((*iterPulses)->numberOfReturns > 0)
            {
                // Copy Points to new pulse
                for(iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
                {
                    // Create empty point
                    SPDPointPy ptPy;

                    // Copy point info to new point
                    ptPy.returnID = (*iterPoints)->returnID;
                    ptPy.gpsTime = (*iterPoints)->gpsTime;
                    ptPy.x = (*iterPoints)->x;
                    ptPy.y = (*iterPoints)->y;
                    ptPy.z = (*iterPoints)->z;
                    ptPy.height = (*iterPoints)->height;
                    ptPy.range = (*iterPoints)->range;
                    ptPy.amplitudeReturn = (*iterPoints)->amplitudeReturn;
                    ptPy.widthReturn = (*iterPoints)->widthReturn;
                    ptPy.red = (*iterPoints)->red;
                    ptPy.green = (*iterPoints)->green;
                    ptPy.blue = (*iterPoints)->blue;
                    ptPy.classification = (*iterPoints)->classification;
                    ptPy.user = (*iterPoints)->user;
                    ptPy.modelKeyPoint = (*iterPoints)->modelKeyPoint;
                    ptPy.lowPoint = (*iterPoints)->lowPoint;
                    ptPy.overlap = (*iterPoints)->overlap;
                    ptPy.ignore = (*iterPoints)->ignore;
                    ptPy.wavePacketDescIdx = (*iterPoints)->wavePacketDescIdx;
                    ptPy.waveformOffset = (*iterPoints)->waveformOffset;

                    pulsePy.pts.append(ptPy);
                }
            }

            if((*iterPulses)->numOfTransmittedBins > 0)
            {
                // Copy transmitted bin values to new pulse
                for(boost::uint_fast16_t i = 0; i < (*iterPulses)->numOfTransmittedBins; ++i)
                {
                    pulsePy.transmitted.append((*iterPulses)->transmitted[i]);
                }
            }

            if((*iterPulses)->numOfReceivedBins > 0)
            {
                // Copy received bin values to new pulse
                for(boost::uint_fast16_t i = 0; i < (*iterPulses)->numOfReceivedBins; ++i)
                {
                    pulsePy.received.append((*iterPulses)->received[i]);
                }
            }

            // Append pulse to output list.
            outPulses->append(pulsePy);

            // Delete SPD pulse from memory (i.e., the C++ version!)
            SPDPulseUtils::deleteSPDPulse(*iterPulses);
        }
        pulses->clear();
    };

    inline void convertVector2PyList(std::vector<SPDPulse*> *pulses, boost::python::list *outPulses) throw(SPDException)
    {
        // Create iterators
        std::vector<SPDPulse*>::iterator iterPulses;
        std::vector<SPDPoint*>::iterator iterPoints;

        // Iterate through pulses and copy to python list
        for(iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
        {
            // Create empty pulse
            SPDPulsePy pulsePy;

            // Copy pulse
            pulsePy.pulseID = (*iterPulses)->pulseID;
            pulsePy.gpsTime = (*iterPulses)->gpsTime;
            pulsePy.x0 = (*iterPulses)->x0;
            pulsePy.y0 = (*iterPulses)->y0;
            pulsePy.z0 = (*iterPulses)->z0;
            pulsePy.h0 = (*iterPulses)->h0;
            pulsePy.xIdx = (*iterPulses)->xIdx;
            pulsePy.yIdx = (*iterPulses)->yIdx;
            pulsePy.azimuth = (*iterPulses)->azimuth;
            pulsePy.zenith = (*iterPulses)->zenith;
            pulsePy.numberOfReturns = (*iterPulses)->numberOfReturns;
            pulsePy.numOfTransmittedBins = (*iterPulses)->numOfTransmittedBins;
            pulsePy.numOfReceivedBins = (*iterPulses)->numOfReceivedBins;
            pulsePy.rangeToWaveformStart = (*iterPulses)->rangeToWaveformStart;
            pulsePy.amplitudePulse = (*iterPulses)->amplitudePulse;
            pulsePy.widthPulse = (*iterPulses)->widthPulse;
            pulsePy.user = (*iterPulses)->user;
            pulsePy.sourceID = (*iterPulses)->sourceID;
            pulsePy.scanline = (*iterPulses)->scanline;
            pulsePy.scanlineIdx = (*iterPulses)->scanlineIdx;
            pulsePy.edgeFlightLineFlag = (*iterPulses)->edgeFlightLineFlag;
            pulsePy.scanDirectionFlag = (*iterPulses)->scanDirectionFlag;
            pulsePy.scanAngleRank = (*iterPulses)->scanAngleRank;
            pulsePy.receiveWaveNoiseThreshold = (*iterPulses)->receiveWaveNoiseThreshold;
            pulsePy.transWaveNoiseThres = (*iterPulses)->transWaveNoiseThres;
            pulsePy.receiveWaveGain = (*iterPulses)->receiveWaveGain;
            pulsePy.receiveWaveOffset = (*iterPulses)->receiveWaveOffset;
            pulsePy.transWaveGain = (*iterPulses)->transWaveGain;
            pulsePy.transWaveOffset = (*iterPulses)->transWaveOffset;
            pulsePy.wavelength = (*iterPulses)->wavelength;

            if((*iterPulses)->numberOfReturns > 0)
            {
                // Copy Points to new pulse
                for(iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
                {
                    // Create empty point
                    SPDPointPy ptPy;

                    // Copy point info to new point
                    ptPy.returnID = (*iterPoints)->returnID;
                    ptPy.gpsTime = (*iterPoints)->gpsTime;
                    ptPy.x = (*iterPoints)->x;
                    ptPy.y = (*iterPoints)->y;
                    ptPy.z = (*iterPoints)->z;
                    ptPy.height = (*iterPoints)->height;
                    ptPy.range = (*iterPoints)->range;
                    ptPy.amplitudeReturn = (*iterPoints)->amplitudeReturn;
                    ptPy.widthReturn = (*iterPoints)->widthReturn;
                    ptPy.red = (*iterPoints)->red;
                    ptPy.green = (*iterPoints)->green;
                    ptPy.blue = (*iterPoints)->blue;
                    ptPy.classification = (*iterPoints)->classification;
                    ptPy.user = (*iterPoints)->user;
                    ptPy.modelKeyPoint = (*iterPoints)->modelKeyPoint;
                    ptPy.lowPoint = (*iterPoints)->lowPoint;
                    ptPy.overlap = (*iterPoints)->overlap;
                    ptPy.ignore = (*iterPoints)->ignore;
                    ptPy.wavePacketDescIdx = (*iterPoints)->wavePacketDescIdx;
                    ptPy.waveformOffset = (*iterPoints)->waveformOffset;

                    pulsePy.pts.append(ptPy);
                }
            }

            if((*iterPulses)->numOfTransmittedBins > 0)
            {
                // Copy transmitted bin values to new pulse
                for(boost::uint_fast16_t i = 0; i < (*iterPulses)->numOfTransmittedBins; ++i)
                {
                    pulsePy.transmitted.append((*iterPulses)->transmitted[i]);

                }
            }

            if((*iterPulses)->numOfReceivedBins > 0)
            {
                // Copy received bin values to new pulse
                for(boost::uint_fast16_t i = 0; i < (*iterPulses)->numOfReceivedBins; ++i)
                {
                    pulsePy.received.append((*iterPulses)->received[i]);

                }
            }

            // Append pulse to output list.
            outPulses->append(pulsePy);

            // Delete SPD pulse from memory (i.e., the C++ version!)
            SPDPulseUtils::deleteSPDPulse(*iterPulses);
        }

        pulses->clear();
    };

    inline std::vector<SPDPulse*>* convertPyList2Vector(boost::python::list *pulses) throw(SPDException)
    {
        std::vector<SPDPulse*> *outPulses = new std::vector<SPDPulse*>();
        try
        {
            outPulses->reserve(len(*pulses));

            SPDPulsePy pulsePy;
            SPDPointPy ptPy;

            boost::uint_fast64_t numPulses = len(*pulses);

            for(boost::uint_fast64_t i = 0; i < numPulses; ++i)
            {
                pulsePy = boost::python::extract<SPDPulsePy>((*pulses)[i])();

                SPDPulse *spdPulse = new SPDPulse();

                // Copy pulse
                spdPulse->pulseID = pulsePy.pulseID;
                spdPulse->gpsTime = pulsePy.gpsTime;
                spdPulse->x0 = pulsePy.x0;
                spdPulse->y0 = pulsePy.y0;
                spdPulse->z0 = pulsePy.z0;
                spdPulse->h0 = pulsePy.h0;
                spdPulse->xIdx = pulsePy.xIdx;
                spdPulse->yIdx = pulsePy.yIdx;
                spdPulse->azimuth = pulsePy.azimuth;
                spdPulse->zenith = pulsePy.zenith;
                spdPulse->numberOfReturns = pulsePy.numberOfReturns;
                spdPulse->numOfTransmittedBins = pulsePy.numOfTransmittedBins;
                spdPulse->numOfReceivedBins = pulsePy.numOfReceivedBins;
                spdPulse->rangeToWaveformStart = pulsePy.rangeToWaveformStart;
                spdPulse->amplitudePulse = pulsePy.amplitudePulse;
                spdPulse->widthPulse = pulsePy.widthPulse;
                spdPulse->user = pulsePy.user;
                spdPulse->sourceID = pulsePy.sourceID;
                spdPulse->scanline = pulsePy.scanline;
                spdPulse->scanlineIdx = pulsePy.scanlineIdx;
                spdPulse->edgeFlightLineFlag = pulsePy.edgeFlightLineFlag;
                spdPulse->scanDirectionFlag = pulsePy.scanDirectionFlag;
                spdPulse->scanAngleRank = pulsePy.scanAngleRank;
                spdPulse->receiveWaveNoiseThreshold = pulsePy.receiveWaveNoiseThreshold;
                spdPulse->transWaveNoiseThres = pulsePy.transWaveNoiseThres;
                spdPulse->receiveWaveGain = pulsePy.receiveWaveGain;
                spdPulse->receiveWaveOffset = pulsePy.receiveWaveOffset;
                spdPulse->transWaveGain = pulsePy.transWaveGain;
                spdPulse->transWaveOffset = pulsePy.transWaveOffset;
                spdPulse->wavelength = pulsePy.wavelength;

                if(len(pulsePy.pts) > 0)
                {
                    spdPulse->pts = new vector<SPDPoint*>();
                    spdPulse->pts->reserve(len(pulsePy.pts));
                    for(uint_fast32_t j = 0; j < len(pulsePy.pts); ++j)
                    {
                        // Get Python Point
                        ptPy = boost::python::extract<SPDPointPy>(pulsePy.pts[j])();

                        // Create empty point
                        SPDPoint *pt = new SPDPoint();

                        // Copy point info to new point
                        pt->returnID = ptPy.returnID;
                        pt->gpsTime = ptPy.gpsTime;
                        pt->x = ptPy.x;
                        pt->y = ptPy.y;
                        pt->z = ptPy.z;
                        pt->height = ptPy.height;
                        pt->range = ptPy.range;
                        pt->amplitudeReturn = ptPy.amplitudeReturn;
                        pt->widthReturn = ptPy.widthReturn;
                        pt->red = ptPy.red;
                        pt->green = ptPy.green;
                        pt->blue = ptPy.blue;
                        pt->classification = ptPy.classification;
                        pt->user = ptPy.user;
                        pt->modelKeyPoint = ptPy.modelKeyPoint;
                        pt->lowPoint = ptPy.lowPoint;
                        pt->overlap = ptPy.overlap;
                        pt->ignore = ptPy.ignore;
                        pt->wavePacketDescIdx = ptPy.wavePacketDescIdx;
                        pt->waveformOffset = ptPy.waveformOffset;

                        spdPulse->pts->push_back(pt);
                    }
                }

                if(len(pulsePy.transmitted) > 0)
                {
                    spdPulse->transmitted = new uint_fast32_t[len(pulsePy.transmitted)];

                    // Copy transmitted bin values to new pulse
                    for(boost::uint_fast16_t j = 0; j < len(pulsePy.transmitted); ++j)
                    {
                        spdPulse->transmitted[j] = boost::python::extract<uint_fast32_t>(pulsePy.transmitted[j])();
                    }
                }

                if(len(pulsePy.received) > 0)
                {
                    spdPulse->received = new uint_fast32_t[len(pulsePy.received)];

                    // Copy received bin values to new pulse
                    for(boost::uint_fast16_t j = 0; j < len(pulsePy.received); ++j)
                    {
                        spdPulse->received[j] = boost::python::extract<uint_fast32_t>(pulsePy.received[j])();
                    }
                }

                outPulses->push_back(spdPulse);
            }
        }
        catch(std::exception &e)
        {
            throw SPDException(e.what());
        }

        return outPulses;
    };

    inline std::list<SPDPulse*>* convertPyList2List(boost::python::list *pulses) throw(SPDException)
    {
        std::list<SPDPulse*> *outPulses = new std::list<SPDPulse*>();

        SPDPulsePy pulsePy;
        SPDPointPy ptPy;

        boost::uint_fast64_t numPulses = len(*pulses);

        for(boost::uint_fast64_t i = 0; i < numPulses; ++i)
        {
            pulsePy = boost::python::extract<SPDPulsePy>((*pulses)[i])();

            SPDPulse *spdPulse = new SPDPulse();

            // Copy pulse
            spdPulse->pulseID = pulsePy.pulseID;
            spdPulse->gpsTime = pulsePy.gpsTime;
            spdPulse->x0 = pulsePy.x0;
            spdPulse->y0 = pulsePy.y0;
            spdPulse->z0 = pulsePy.z0;
            spdPulse->h0 = pulsePy.h0;
            spdPulse->xIdx = pulsePy.xIdx;
            spdPulse->yIdx = pulsePy.yIdx;
            spdPulse->azimuth = pulsePy.azimuth;
            spdPulse->zenith = pulsePy.zenith;
            spdPulse->numberOfReturns = pulsePy.numberOfReturns;
            spdPulse->numOfTransmittedBins = pulsePy.numOfTransmittedBins;
            spdPulse->numOfReceivedBins = pulsePy.numOfReceivedBins;
            spdPulse->rangeToWaveformStart = pulsePy.rangeToWaveformStart;
            spdPulse->amplitudePulse = pulsePy.amplitudePulse;
            spdPulse->widthPulse = pulsePy.widthPulse;
            spdPulse->user = pulsePy.user;
            spdPulse->sourceID = pulsePy.sourceID;
            spdPulse->scanline = pulsePy.scanline;
            spdPulse->scanlineIdx = pulsePy.scanlineIdx;
            spdPulse->edgeFlightLineFlag = pulsePy.edgeFlightLineFlag;
            spdPulse->scanDirectionFlag = pulsePy.scanDirectionFlag;
            spdPulse->scanAngleRank = pulsePy.scanAngleRank;
            spdPulse->receiveWaveNoiseThreshold = pulsePy.receiveWaveNoiseThreshold;
            spdPulse->transWaveNoiseThres = pulsePy.transWaveNoiseThres;
            spdPulse->receiveWaveGain = pulsePy.receiveWaveGain;
            spdPulse->receiveWaveOffset = pulsePy.receiveWaveOffset;
            spdPulse->transWaveGain = pulsePy.transWaveGain;
            spdPulse->transWaveOffset = pulsePy.transWaveOffset;
            spdPulse->wavelength = pulsePy.wavelength;

            if(len(pulsePy.pts) > 0)
            {
                spdPulse->pts = new vector<SPDPoint*>();
                spdPulse->pts->reserve(len(pulsePy.pts));
                for(boost::uint_fast32_t j = 0; j < len(pulsePy.pts); ++j)
                {
                    // Get Python Point
                    ptPy = boost::python::extract<SPDPointPy>(pulsePy.pts[j])();

                    // Create empty point
                    SPDPoint *pt = new SPDPoint();

                    // Copy point info to new point
                    pt->returnID = ptPy.returnID;
                    pt->gpsTime = ptPy.gpsTime;
                    pt->x = ptPy.x;
                    pt->y = ptPy.y;
                    pt->z = ptPy.z;
                    pt->height = ptPy.height;
                    pt->range = ptPy.range;
                    pt->amplitudeReturn = ptPy.amplitudeReturn;
                    pt->widthReturn = ptPy.widthReturn;
                    pt->red = ptPy.red;
                    pt->green = ptPy.green;
                    pt->blue = ptPy.blue;
                    pt->classification = ptPy.classification;
                    pt->user = ptPy.user;
                    pt->modelKeyPoint = ptPy.modelKeyPoint;
                    pt->lowPoint = ptPy.lowPoint;
                    pt->overlap = ptPy.overlap;
                    pt->ignore = ptPy.ignore;
                    pt->wavePacketDescIdx = ptPy.wavePacketDescIdx;
                    pt->waveformOffset = ptPy.waveformOffset;

                    spdPulse->pts->push_back(pt);
                }
            }

            if(len(pulsePy.transmitted) > 0)
            {
                spdPulse->transmitted = new boost::uint_fast32_t[len(pulsePy.transmitted)];

                // Copy transmitted bin values to new pulse
                for(boost::uint_fast16_t j = 0; j < len(pulsePy.transmitted); ++j)
                {
                    spdPulse->transmitted[j] = boost::python::extract<boost::uint_fast32_t>(pulsePy.transmitted[j])();
                }
            }

            if(len(pulsePy.received) > 0)
            {
                spdPulse->received = new boost::uint_fast32_t[len(pulsePy.received)];

                // Copy received bin values to new pulse
                for(boost::uint_fast16_t j = 0; j < len(pulsePy.received); ++j)
                {
                    spdPulse->received[j] = boost::python::extract<boost::uint_fast32_t>(pulsePy.received[j])();
                }
            }

            outPulses->push_back(spdPulse);
        }

        return outPulses;
    };


    void translate(SPDException const& e)
    {
        // Use the Python 'C' API to set up an exception object
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

#endif
