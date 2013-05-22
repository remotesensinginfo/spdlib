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


#include "spd/SPDException.h"
#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDPoint.h"
#include "spd/SPDFileReader.h"
#include "spd/SPDFileWriter.h"
#include "spd/SPDFileIncrementalReader.h"
#include "spd/SPDIOUtils.h"
#include "spd/SPDLASFileImporter.h"

#include "spd/SPDProcessDataBlocks.h"
#include "spd/SPDDataBlockProcessor.h"

int main (int argc, char * const argv[]) 
{
    std::cout.precision(12);
    
    try
    {
        spdlib::SPDFile *spdInFile = new spdlib::SPDFile("/Users/pete/Temp/MfE_LiDAR/LI080228_RAW4_1m.spd");
        
        spdlib::SPDDataBlockProcessorBlank *blockProcessor = new spdlib::SPDDataBlockProcessorBlank();
        spdlib::SPDProcessDataBlocks processBlocks = spdlib::SPDProcessDataBlocks(blockProcessor);
        processBlocks.processDataBlocksGridPulsesOutputSPD(spdInFile, "/Users/pete/Temp/MfE_LiDAR/LI080228_RAW4_1m_out.spd", 2);
        
        delete blockProcessor;
    }
    catch(spdlib::SPDException &e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
    }
    std::cout << "spdtest - end\n";
}