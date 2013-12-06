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
    
    
    class SPDSALCADataBinaryImporter : public SPDDataImporter
	{
	public:
		SPDSALCADataBinaryImporter(bool convertCoords=false, std::string outputProjWKT="", std::string schema="", boost::uint_fast16_t indexCoords=SPD_FIRST_RETURN, bool defineOrigin=false, double originX=0, double originY=0, float originZ=0, float waveNoiseThreshold=0);
		SPDDataImporter* getInstance(bool convertCoords, std::string outputProjWKT,std::string schema,boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold);
        std::list<SPDPulse*>* readAllDataToList(std::string, SPDFile *spdFile)throw(SPDIOException);
		std::vector<SPDPulse*>* readAllDataToVector(std::string inputFile, SPDFile *spdFile)throw(SPDIOException);
		void readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) throw(SPDIOException);
		bool isFileType(std::string fileType);
        void readHeaderInfo(std::string inputFile, SPDFile *spdFile) throw(SPDIOException);
		~SPDSALCADataBinaryImporter();
	private:
		SalcaHDRParams* readHeaderParameters(std::string headerFilePath, std::vector<std::pair<float,std::string> > *fileList)throw(SPDIOException);
        /** Translate from nice squint angles to those used in equations */
        void translateSquint(SalcaHDRParams *options);
        /** Precalculate squint angles */
        void setSquint(SalcaHDRParams *options, int numb);
        /** Caluclate squint angle */
        void squint(float *cZen,float *cAz,float zM,float aM,float zE,float aE,float omega);
        /** Rotate about x axis */
        void rotateX(float *vect,float theta);
        /** Rotate about y axis */
        void rotateY(float *vect,float theta);
        /** Rotate about z axis */
        void rotateZ(float *vect,float theta);
        /** read data into array */
        char* readData(std::string inFilePath, int i, int *numb, int *nBins, int *length, SalcaHDRParams *options) throw(SPDIOException);
        /** Find outgoing pulse and check saturation */
        int findStart(int start,int end,char *satTest, char *data,int offset);
	};
    
    
}

#endif


