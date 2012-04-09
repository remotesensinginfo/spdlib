/*
 *  SPDRemoveVerticalNoise.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 07/06/2011.
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

#ifndef SPDRemoveVerticalNoise_H
#define SPDRemoveVerticalNoise_H

#include <iostream>
#include <string>
#include <list>
#include "math.h"

#include "gsl/gsl_statistics_double.h"

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDPoint.h"
#include "spd/SPDProcessPulses.h"
#include "spd/SPDPulseProcessor.h"
#include "spd/SPDProcessingException.h"

using namespace std;

namespace spdlib
{
    class SPDRemoveVerticalNoise : public SPDPulseProcessor
	{
	public:
        SPDRemoveVerticalNoise(bool absUpSet, bool absLowSet, bool relUpSet, bool relLowSet, float absUpThres, float absLowThres, float relUpThres, float relLowThres);
        
        void processDataColumnImage(SPDFile *inSPDFile, vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing is not implemented for processDataColumn().");};
        
		void processDataColumn(SPDFile *inSPDFile, vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException);
        
        void processDataWindowImage(SPDFile *inSPDFile, vector<SPDPulse*> ***pulses, float ***imageData, SPDXYPoint ***cenPts, boost::uint_fast32_t numImgBands, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
		void processDataWindow(SPDFile *inSPDFile, vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
        
        vector<string> getImageBandDescriptions() throw(SPDProcessingException){return vector<string>();};
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException){};
        
        ~SPDRemoveVerticalNoise();
    protected:
        bool absUpSet;
        bool absLowSet;
        bool relUpSet;
        bool relLowSet;
        float absUpThres;
        float absLowThres;
        float relUpThres;
        float relLowThres;
	};
}

#endif