/*
 *  SPDCalcFileStats.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 28/03/2012.
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


#ifndef SPDCalcFileStats_H
#define SPDCalcFileStats_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"

#include "spd/SPDProcessPulses.h"
#include "spd/SPDPulseProcessor.h"
#include "spd/SPDProcessingException.h"

namespace spdlib
{	
	class DllExport SPDCalcFileStats
	{
	public:
		SPDCalcFileStats();
        void calcImagePulsePointDensity(std::string inputSPDFile, std::string outputImageFile, boost::uint_fast32_t blockXSize=250, boost::uint_fast32_t blockYSize=250, float processingResolution=0, std::string gdalFormat="ENVI") throw(SPDProcessingException);
        void calcOverallPulsePointDensityStats(std::string inputSPDFile, std::string outputTextFile, boost::uint_fast32_t blockXSize=250, boost::uint_fast32_t blockYSize=250, float processingResolution=0) throw(SPDProcessingException);
		~SPDCalcFileStats();
	};
    
    
    class DllExport SPDPulseProcessorCalcStats : public SPDPulseProcessor
	{
	public:
        SPDPulseProcessorCalcStats();
        
        void processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException);
		void processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException);
        
        void processDataWindowImage(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, float ***imageData, SPDXYPoint ***cenPts, boost::uint_fast32_t numImgBands, float binSize, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
		void processDataWindow(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
        
        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException);
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException);
        
        void setCalcStdDev(float meanPulses, float meanPoints)
        {
            first = true;
            calcStdDevVals = true;
            this->meanPulses = meanPulses;
            this->meanPoints = meanPoints;
        };
        
        boost::uint_fast64_t getBinCount(){return countBins;};
        boost::uint_fast64_t getMinPulses(){return minPulses;};
        boost::uint_fast64_t getMaxPulses(){return maxPulses;};
        float getMeanPulses(){return ((double)sumPulses)/(double(countBins));};
        float getStdDevPulses(){return sqrt(sqDiffPulses/(double(countBins)));};
        boost::uint_fast64_t getMinPoints(){return minPoints;};
        boost::uint_fast64_t getMaxPoints(){return maxPoints;};
        float getMeanPoints(){return ((double)sumPoints)/(double(countBins));};
        float getStdDevPoints(){return sqrt(sqDiffPoints/(double(countBins)));};
        
        ~SPDPulseProcessorCalcStats();
    protected:
        bool calcStdDevVals;
        bool first;
        boost::uint_fast32_t countBins;
        
        boost::uint_fast64_t sumPulses;
        boost::uint_fast32_t minPulses;
        boost::uint_fast32_t maxPulses;
        float meanPulses;
        double sqDiffPulses;
        
        boost::uint_fast64_t sumPoints;
        boost::uint_fast32_t minPoints;
        boost::uint_fast32_t maxPoints;
        float meanPoints;
        double sqDiffPoints;
	};
    
}

#endif
