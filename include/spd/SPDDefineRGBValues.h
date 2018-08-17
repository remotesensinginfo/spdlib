/*
 *  SPDDefineRGBValues.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 10/03/2011.
 *  Copyright 2010 SPDLib. All rights reserved.
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


#ifndef SPDDefineRGBValues_H
#define SPDDefineRGBValues_H

#include <iostream>
#include <string>
#include <vector>

#include <boost/math/special_functions/fpclassify.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDProcessPulses.h"
#include "spd/SPDPulseProcessor.h"
#include "spd/SPDProcessingException.h"

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
	
    class DllExport SPDDefineRGBValues : public SPDPulseProcessor
	{
	public:
        SPDDefineRGBValues(boost::uint_fast16_t redBand, boost::uint_fast16_t greenBand, boost::uint_fast16_t blueBand);

        void processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException);
		void processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing is not implemented for processDataColumn().");};
        void processDataWindowImage(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, float ***imageData, SPDXYPoint ***cenPts, boost::uint_fast32_t numImgBands, float binSize, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
		void processDataWindow(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};

        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {return std::vector<std::string>();};
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
        {
            spdFile->setRGBDefined(SPD_TRUE);
        };

        ~SPDDefineRGBValues();
    protected:
        boost::uint_fast16_t redBand;
        boost::uint_fast16_t greenBand;
        boost::uint_fast16_t blueBand;
	};

    class DllExport SPDFindRGBValuesStats : public SPDPulseProcessor
	{
	public:
        SPDFindRGBValuesStats();

        void processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing is not implemented for processDataColumnImage().");};
		void processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException);
        void processDataWindowImage(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, float ***imageData, SPDXYPoint ***cenPts, boost::uint_fast32_t numImgBands, float binSize, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
		void processDataWindow(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};

        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {return std::vector<std::string>();};
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
        {};

        void setCalcStdDev(bool calcStdDev, float redMean, float greenMean, float blueMean)
        {
            this->calcStdDev = calcStdDev;

            first = true;
            countPts = 0;

            this->redMean = 0;
            this->redStdDev = 0;
            this->redMin = 0;
            this->redMax = 0;

            this->greenMean = 0;
            this->greenStdDev = 0;
            this->greenMin = 0;
            this->greenMax = 0;

            this->blueMean = 0;
            this->blueStdDev = 0;
            this->blueMin = 0;
            this->blueMax = 0;

            this->redMean = redMean;
            this->greenMean = greenMean;
            this->blueMean = blueMean;
        };

        void reset()
        {
            this->calcStdDev = false;
            first = true;
            countPts = 0;

            this->redMean = 0;
            this->redStdDev = 0;
            this->redMin = 0;
            this->redMax = 0;

            this->greenMean = 0;
            this->greenStdDev = 0;
            this->greenMin = 0;
            this->greenMax = 0;

            this->blueMean = 0;
            this->blueStdDev = 0;
            this->blueMin = 0;
            this->blueMax = 0;
        }

        float getRedMean(){return this->redMean/this->countPts;};
        float getRedStdDev(){return sqrt(this->redStdDev/this->countPts);};
        boost::uint_fast16_t getRedMin(){return this->redMin;};
        boost::uint_fast16_t getRedMax(){return this->redMax;};

        float getGreenMean(){return this->greenMean/this->countPts;};
        float getGreenStdDev(){return sqrt(this->greenStdDev/this->countPts);};
        boost::uint_fast16_t getGreenMin(){return this->greenMin;};
        boost::uint_fast16_t getGreenMax(){return this->greenMax;};

        float getBlueMean(){return this->blueMean/this->countPts;};
        float getBlueStdDev(){return sqrt(this->blueStdDev/this->countPts);};
        boost::uint_fast16_t getBlueMin(){return this->blueMin;};
        boost::uint_fast16_t getBlueMax(){return this->blueMax;};

        ~SPDFindRGBValuesStats();
    protected:
        bool calcStdDev;

        double redMean;
        double redStdDev;
        boost::uint_fast16_t redMin;
        boost::uint_fast16_t redMax;

        double greenMean;
        double greenStdDev;
        boost::uint_fast16_t greenMin;
        boost::uint_fast16_t greenMax;

        double blueMean;
        double blueStdDev;
        boost::uint_fast16_t blueMin;
        boost::uint_fast16_t blueMax;

        bool first;
        boost::uint_fast64_t countPts;
	};


    class DllExport SPDLinearStretchRGBValues : public SPDPulseProcessor
	{
	public:
        SPDLinearStretchRGBValues(float redMin, float redMax, float greenMin, float greenMax, float blueMin, float blueMax, bool stretchIndepend);

        void processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing is not implemented for processDataColumnImage().");};
		void processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException);
        void processDataWindowImage(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, float ***imageData, SPDXYPoint ***cenPts, boost::uint_fast32_t numImgBands, float binSize, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
		void processDataWindow(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};

        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {return std::vector<std::string>();};
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
        {};
        uint_fast16_t scalePixelValue(uint_fast16_t value);

        ~SPDLinearStretchRGBValues();
    protected:
        bool stretchIndepend;

        float redMin;
        float redMax;
        float greenMin;
        float greenMax;
        float blueMin;
        float blueMax;
        float redRange;
        float greenRange;
        float blueRange;

        float maxRange;
        float totalMin;
        float totalMax;
	};

}

#endif




