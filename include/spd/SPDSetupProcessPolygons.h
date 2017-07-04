/*
 *  SPDSetupProcessPolygons.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 05/04/2012.
 *  Copyright 20121 SPDLib. All rights reserved.
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
 */


#ifndef SPDSetupProcessPolygons_H
#define SPDSetupProcessPolygons_H

#include <fstream>
#include <iostream>
#include <string>
#include <list>

#include "gdal_priv.h"
#include "ogrsf_frmts.h"
#include "ogr_api.h"

#include <boost/cstdint.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDPolygonProcessor.h"
#include "spd/SPDProcessPolygons.h"
#include "spd/SPDFileIncrementalReader.h"
#include "spd/SPDFileUtilities.h"

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
	
	class DllExport SPDSetupProcessPolygonsAbstract
	{
	public:
		SPDSetupProcessPolygonsAbstract(){};
		virtual void processPolygons(std::string spdInputFile, std::string inputLayer, std::string outputLayer, bool deleteOutShpIfExists, bool copyAttributes, SPDPolygonProcessor *processor)throw(SPDProcessingException) = 0;
		virtual void processPolygons(std::string spdInputFile, std::string inputLayer, std::string outputASCII, SPDPolygonProcessor *processor)throw(SPDProcessingException) = 0;
		virtual ~SPDSetupProcessPolygonsAbstract(){};
	protected:
		std::string getLayerName(std::string filepath);
	};
	
	class DllExport SPDSetupProcessShapefilePolygons : public SPDSetupProcessPolygonsAbstract
	{
	public:
		SPDSetupProcessShapefilePolygons();
		void processPolygons(std::string spdInputFile, std::string inputLayer, std::string outputLayer, bool deleteOutShpIfExists,  bool copyAttributes, SPDPolygonProcessor *processor)throw(SPDProcessingException);
        void processPolygons(std::string spdInputFile, std::string inputLayer, std::string outputASCII, SPDPolygonProcessor *processor)throw(SPDProcessingException);
        ~SPDSetupProcessShapefilePolygons();
	};
}

#endif



