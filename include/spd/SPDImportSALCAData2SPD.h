/*
 *  SPDImportSALCAData2SPD.h
 *  spdlib
 *
 *  Created by Pete Bunting on 04/12/2013.
 *  Copyright 2013 SPDLib. All rights reserved.
 *
 *  Code within this file has been provided by
 *  Steven Hancock for reading the SALCA data
 *  and sorting out the geometry. This has been
 *  adapted and brought across into the SPD
 *  importer interface.
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

#ifndef SPDImportSALCAData2SPD_H
#define SPDImportSALCAData2SPD_H

#include <list>
#include <vector>

#include <boost/cstdint.hpp>
#include "boost/filesystem.hpp"
#include <boost/algorithm/string/trim.hpp>

#include "ogrsf_frmts.h"

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDDataExporter.h"
#include "spd/SPDDataImporter.h"
#include "spd/SPDTextFileLineReader.h"
#include "spd/SPDTextFileUtilities.h"

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
	
    typedef struct{ /*to hold SALCA header file options*/
        float maxR;
        int nAz;
        int nZen;
        float azStep;
        float zStep;
        float maxZen;
        float azStart;
        float azSquint;
        float zenSquint;
        float *zen;       /*true zenith*/
        float *azOff;     /*azimuth offset*/
        float omega;      /*mirror slope angle*/
    } SalcaHDRParams;
    
    
    class DllExport SPDSALCADataBinaryImporter : public SPDDataImporter
	{
	public:
		SPDSALCADataBinaryImporter(bool convertCoords=false, std::string outputProjWKT="", std::string schema="", boost::uint_fast16_t indexCoords=SPD_FIRST_RETURN, bool defineOrigin=false, double originX=0, double originY=0, float originZ=0, float waveNoiseThreshold=0);
		SPDDataImporter* getInstance(bool convertCoords, std::string outputProjWKT,std::string schema,boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold);
        std::list<SPDPulse*>* readAllDataToList(std::string, SPDFile *spdFile);
		std::vector<SPDPulse*>* readAllDataToVector(std::string inputFile, SPDFile *spdFile);
		void readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) ;
		bool isFileType(std::string fileType);
        void readHeaderInfo(std::string inputFile, SPDFile *spdFile) ;
		~SPDSALCADataBinaryImporter();
	private:
		SalcaHDRParams* readHeaderParameters(std::string headerFilePath, std::vector<std::pair<float,std::string> > *fileList);
        /** read data into array */
        int* readData(std::string inFilePath, int i, unsigned int numb, unsigned int nBins, unsigned int *length) ;
        /** Find outgoing pulse and check saturation */
        void findWaveformsBinIdxes(int *data, unsigned int dataLen, unsigned int maxRNBins, unsigned int prevWl2End, unsigned int *wl1StartIdxTrans, unsigned int *wl2StartIdxTrans, unsigned int *wl1EndIdxTrans, unsigned int *wl2EndIdxTrans, unsigned int *wl1StartIdxRec, unsigned int *wl2StartIdxRec, unsigned int *wl1EndIdxRec, unsigned int *wl2EndIdxRec) ;
        bool zeroCrossing(int *data, unsigned int startIdx, unsigned int endIdx, unsigned int idx) ;
	};
    
    
}

#endif


