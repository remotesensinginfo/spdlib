/*
 *  SPDRasterInterpolation.h
 *
 *  Created by Pete Bunting on 05/03/2012.
 *  Copyright 2012 SPDLib. All rights reserved.
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
 *
 */

#ifndef SPDRasterInterpolation_H
#define SPDRasterInterpolation_H

#include <iostream>
#include <string>
#include <vector>

#include <boost/cstdint.hpp>

#include "spd/SPDCommon.h"
#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDDataBlockProcessor.h"
#include "spd/SPDPointInterpolation.h"
#include "spd/SPDGridData.h"

#include "ogr_api.h"

#include "boost/math/special_functions/fpclassify.hpp"

// mark all exported classes/functions with DllExport to have
// them exported by Visual Studio
#undef DllExport
#ifdef _MSC_VER
    #ifdef libspd_EXPORTS
        #define DllExport   __declspec( dllexport )
    #else
        #define DllExport   __declspec( dllimport )
    #endif
#else
    #define DllExport
#endif

namespace spdlib
{

    class DllExport SPDDTMInterpolation : public SPDDataBlockProcessor
	{
	public:
        SPDDTMInterpolation(SPDPointInterpolator *interpolator);

        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException);
		
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException)
        {throw SPDProcessingException("SPDDTMInterpolation only produces an image output.");};

        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) throw(SPDProcessingException);

        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses) throw(SPDProcessingException)
        {throw SPDProcessingException("SPDDTMInterpolation requires processing with a grid.");};

        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {
            std::vector<std::string> bandNames;
            bandNames.push_back("DTM");
            return bandNames;
        }
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
        {
            // Nothing to do...
        }

        ~SPDDTMInterpolation();

    protected:
        SPDPointInterpolator *interpolator;
	};

    class DllExport SPDDSMInterpolation : public SPDDataBlockProcessor
	{
	public:
        SPDDSMInterpolation(SPDPointInterpolator *interpolator);

        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException);
		
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException)
        {throw SPDProcessingException("SPDDSMInterpolation only produces an image output.");};

        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) throw(SPDProcessingException);

        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses) throw(SPDProcessingException)
        {throw SPDProcessingException("SPDDSMInterpolation only produces an image output.");};

        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {
            std::vector<std::string> bandNames;
            bandNames.push_back("DSM");
            return bandNames;
        }
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
        {
            // Nothing to do...
        }

        ~SPDDSMInterpolation();

    protected:
        SPDPointInterpolator *interpolator;
	};

    class DllExport SPDCHMInterpolation : public SPDDataBlockProcessor
	{
	public:
        SPDCHMInterpolation(SPDPointInterpolator *interpolator, bool useVegClassifiedPts, bool useMinThres, double minThresVal);

        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException);
		
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException)
        {throw SPDProcessingException("SPDCHMInterpolation only produces an image output.");};

        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) throw(SPDProcessingException);

        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses) throw(SPDProcessingException)
        {throw SPDProcessingException("SPDCHMInterpolation only produces an image output.");};

        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {
            std::vector<std::string> bandNames;
            bandNames.push_back("CHM");
            return bandNames;
        }
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
        {
            // Nothing to do...
        }

        ~SPDCHMInterpolation();

    protected:
        SPDPointInterpolator *interpolator;
        bool useVegClassifiedPts;
        bool useMinThres;
        double minThresVal;
	};

    class DllExport SPDAmplitudeInterpolation : public SPDDataBlockProcessor
	{
	public:
        SPDAmplitudeInterpolation(SPDPointInterpolator *interpolator, bool useGroundClassifiedPts);

        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException);
		
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException)
        {throw SPDProcessingException("SPDAmplitudeInterpolation only produces an image output.");};

        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) throw(SPDProcessingException)
		{throw SPDProcessingException("SPDAmplitudeInterpolation requires processing with a grid.");};

        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses) throw(SPDProcessingException)
        {throw SPDProcessingException("SPDAmplitudeInterpolation requires processing with a grid.");};

        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {
            std::vector<std::string> bandNames;
            bandNames.push_back("Amplitude");
            return bandNames;
        }
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
        {
            // Nothing to do...
        }

        ~SPDAmplitudeInterpolation();

    protected:
        SPDPointInterpolator *interpolator;
		bool useGroundClassifiedPts;
	};

    class DllExport SPDRangeInterpolation : public SPDDataBlockProcessor
	{
	public:
        SPDRangeInterpolation(SPDPointInterpolator *interpolator);

        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException);
		
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException)
        {throw SPDProcessingException("SPDRangeInterpolation only produces an image output.");};

        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) throw(SPDProcessingException)
		{throw SPDProcessingException("SPDRangeInterpolation requires processing with a grid.");};

        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses) throw(SPDProcessingException)
        {throw SPDProcessingException("SPDRangeInterpolation requires processing with a grid.");};

        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {
            std::vector<std::string> bandNames;
            bandNames.push_back("Range");
            return bandNames;
        }
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
        {
            // Nothing to do...
        }

        ~SPDRangeInterpolation();

    protected:
        SPDPointInterpolator *interpolator;
	};

}



#endif
