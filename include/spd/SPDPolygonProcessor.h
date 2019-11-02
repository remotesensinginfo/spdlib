/*
 *  SPDPolygonProcessor.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 09/03/2011.
 *  Copyright 2011 SPDLib. All rights reserved.
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


#ifndef SPDPolygonProcessor_H
#define SPDPolygonProcessor_H

#include <iostream>
#include <fstream>
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
	
	class DllExport SPDPolygonProcessor
	{
	public:
		SPDPolygonProcessor(){};
		virtual void processFeature(OGRFeature *inFeature, OGRFeature *outFeature, boost::uint_fast64_t fid, std::vector<SPDPulse*> *pulses, SPDFile *spdFile) = 0;
        virtual void processFeature(OGRFeature *inFeature, std::ofstream *outASCIIFile, boost::uint_fast64_t fid, std::vector<SPDPulse*> *pulses, SPDFile *spdFile) = 0;
		virtual void createOutputLayerDefinition(OGRLayer *outputLayer, OGRFeatureDefn *inFeatureDefn)  = 0;
		virtual void writeASCIIHeader(std::ofstream *outASCIIFile)  = 0;
        virtual ~SPDPolygonProcessor(){};
	};
}
							 
#endif
							 
							 
							 

