/*
 *  SPDGenBinaryMask.h
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


#ifndef SPDGenBinaryMask_H
#define SPDGenBinaryMask_H

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

using namespace std;

namespace spdlib
{	
	class SPDGenBinaryMask
	{
	public:
		SPDGenBinaryMask();
        void generateBinaryMask(boost::uint_fast32_t numPulses, string inputSPDFile, string outputImageFile, boost::uint_fast32_t blockXSize=250, boost::uint_fast32_t blockYSize=250, float processingResolution=0, string gdalFormat="ENVI") throw(SPDProcessingException);
		~SPDGenBinaryMask();
	};
    
    
    class SPDPulseProcessorCalcMask : public SPDPulseProcessor
	{
	public:
        SPDPulseProcessorCalcMask(boost::uint_fast32_t numPulses);
        
        void processDataColumnImage(SPDFile *inSPDFile, vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException);
		void processDataColumn(SPDFile *inSPDFile, vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing is not implemented for processDataColumn().");};
        void processDataWindowImage(SPDFile *inSPDFile, vector<SPDPulse*> ***pulses, float ***imageData, SPDXYPoint ***cenPts, boost::uint_fast32_t numImgBands, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
		void processDataWindow(SPDFile *inSPDFile, vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
        
        vector<string> getImageBandDescriptions() throw(SPDProcessingException);
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException);
        
        ~SPDPulseProcessorCalcMask();
    protected:
        boost::uint_fast32_t numPulses;
	};
    
}

#endif
