/*
 *  SPDLASFileImporter.h
 *  spdlib
 *
 *  Created by Pete Bunting on 02/12/2010.
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

#ifndef SPDLASFileImporter_H
#define SPDLASFileImporter_H

#include <list>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include <lasreader.hpp>
#include <laswaveform13reader.hpp>

#include <ogr_spatialref.h>

#include <boost/cstdint.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDTextFileLineReader.h"
#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDDataImporter.h"

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
    
    /** Function to get WKT string from LAS header */
    std::string getWKTfromLAS(LASheader &header);

    /**
     Standard importer for LAS files (LAS)
    */
    class DllExport SPDLASFileImporter : public SPDDataImporter
    {
    public:
        SPDLASFileImporter(bool convertCoords=false, std::string outputProjWKT="", std::string schema="", boost::uint_fast16_t indexCoords=SPD_FIRST_RETURN, bool defineOrigin=false, double originX=0, double originY=0, float originZ=0, float waveNoiseThreshold=0);
        SPDDataImporter* getInstance(bool convertCoords, std::string outputProjWKT,std::string schema,boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold);
      std::list<SPDPulse*>* readAllDataToList(std::string inputFile, SPDFile *spdFile)throw(SPDIOException);
        std::vector<SPDPulse*>* readAllDataToVector(std::string inputFile, SPDFile *spdFile)throw(SPDIOException);
        void readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) throw(SPDIOException);
        bool isFileType(std::string fileType);
      void readHeaderInfo(std::string inputFile, SPDFile *spdFile) throw(SPDIOException);
      void setStrict(bool strictPulses)throw(SPDIOException){this->strictPulses = strictPulses;};
        ~SPDLASFileImporter();
    private:
        SPDPoint* createSPDPoint(LASpoint const& pt)throw(SPDIOException);
        bool classWarningGiven;
      bool strictPulses;
    };
    
    /**
     Strict importer for LAS files (LASSTRICT).
     
     Throws SPDIOException if pulses can't be created from available points (i.e., not all expected returns are found).
     */
    class DllExport SPDLASFileImporterStrictPulses : public SPDDataImporter
    {
    public:
        SPDLASFileImporterStrictPulses(bool convertCoords=false, std::string outputProjWKT="", std::string schema="", boost::uint_fast16_t indexCoords=SPD_FIRST_RETURN, bool defineOrigin=false, double originX=0, double originY=0, float originZ=0, float waveNoiseThreshold=0);
        SPDDataImporter* getInstance(bool convertCoords, std::string outputProjWKT,std::string schema,boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold);
      std::list<SPDPulse*>* readAllDataToList(std::string inputFile, SPDFile *spdFile)throw(SPDIOException);
        std::vector<SPDPulse*>* readAllDataToVector(std::string inputFile, SPDFile *spdFile)throw(SPDIOException);
        void readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) throw(SPDIOException);
        bool isFileType(std::string fileType);
      void readHeaderInfo(std::string inputFile, SPDFile *spdFile) throw(SPDIOException);
        ~SPDLASFileImporterStrictPulses();
    private:
        SPDPoint* createSPDPoint(LASpoint const& pt)throw(SPDIOException);
      SPDLASFileImporter* lasDataImporter;
        bool classWarningGiven;
    };
    
    /**
     No pulse importer for LAS files (LASNP)
     
     */
    class DllExport SPDLASFileNoPulsesImporter : public SPDDataImporter
    {
    public:
        SPDLASFileNoPulsesImporter(bool convertCoords=false, std::string outputProjWKT="", std::string schema="", boost::uint_fast16_t indexCoords=SPD_FIRST_RETURN, bool defineOrigin=false, double originX=0, double originY=0, float originZ=0, float waveNoiseThreshold=0);
        SPDDataImporter* getInstance(bool convertCoords, std::string outputProjWKT,std::string schema,boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold);
        std::list<SPDPulse*>* readAllDataToList(std::string inputFile, SPDFile *spdFile)throw(SPDIOException);
        std::vector<SPDPulse*>* readAllDataToVector(std::string inputFile, SPDFile *spdFile)throw(SPDIOException);
        void readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) throw(SPDIOException);
        bool isFileType(std::string fileType);
        void readHeaderInfo(std::string inputFile, SPDFile *spdFile) throw(SPDIOException);
        ~SPDLASFileNoPulsesImporter();
    private:
        SPDPoint* createSPDPoint(LASpoint const& pt)throw(SPDIOException);
        bool classWarningGiven;
    };
}

#endif


