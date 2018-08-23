/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 22/01/2012.
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

#include <string>
#include <iostream>
#include <algorithm>
#include <math.h>


#include "spd/SPDException.h"
#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDPoint.h"
#include "spd/SPDFileReader.h"
#include "spd/SPDFileWriter.h"
#include "spd/SPDIOUtils.h"

int main (int argc, char * const argv[]) 
{
    std::cout.precision(12);
    
    try
    {
        srand(20);
        unsigned int numBlocks = 100;
        unsigned int numPulsesInBlock = 1000;
        unsigned int waveformLength = 45;
        std::string outputFile = "/Users/pete/Desktop/testspd.spd";
        spdlib::SPDFile *spdFile = new spdlib::SPDFile(outputFile);
        spdlib::SPDNoIdxFileWriter *writer = new spdlib::SPDNoIdxFileWriter();
        spdlib::SPDPulse *pulse = NULL;
        spdlib::SPDPoint *point = NULL;
        spdlib::SPDPulseUtils plsUtils;
        spdlib::SPDPointUtils ptUtils;
        std::vector<spdlib::SPDPulse*> *pulses = new std::vector<spdlib::SPDPulse*>();
        unsigned int plsID = 0;
        if(writer->open(spdFile, outputFile))
        {
            for(unsigned int i = 0; i < numBlocks; ++i)
            {
                for(unsigned int n = 0; n < numPulsesInBlock; ++n)
                {
                    pulse = new spdlib::SPDPulse();
                    plsUtils.initSPDPulse(pulse);
                    pulse->pulseID = plsID++;
                    pulse->numberOfReturns = 1 + (rand() % (int)(5 + 1));
                    for(unsigned j = 0; j < pulse->numberOfReturns; ++j)
                    {
                        point = new spdlib::SPDPoint();
                        ptUtils.initSPDPoint(point);
                        point->returnID = j+1;
                        point->x = 1000 + (rand() % (int)(1000 + 1));
                        point->y = 20000 + (rand() % (int)(1000 + 1));
                        if(j == 0)
                        {
                            pulse->xIdx = point->x;
                            pulse->yIdx = point->y;
                        }
                        point->z = 10 + (rand() % (int)(100 + 1));
                        point->amplitudeReturn = ((float)rand()/(float)RAND_MAX)*100;
                        pulse->pts->push_back(point);
                    }
                    
                    pulse->receiveWaveGain = 1;
                    pulse->receiveWaveOffset = 0;
                    pulse->numOfReceivedBins = waveformLength;
                    pulse->received = new boost::uint_fast32_t[pulse->numOfReceivedBins];
                    for(unsigned int j = 0; j < waveformLength; ++j)
                    {
                        pulse->received[j] = 3 + (rand() % (boost::uint_fast32_t)(100 + 1));
                    }
                    pulses->push_back(pulse);
                }
                writer->writeDataColumn(pulses, 0, 0);
            }
            spdFile->setDiscretePtDefined(spdlib::SPD_TRUE);
            spdFile->setReceiveWaveformDefined(spdlib::SPD_TRUE);
            writer->finaliseClose();
        }
        
        
    }
    catch(spdlib::SPDException &e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
    }
    std::cout << "spdtest - end\n";
}