/*
 *  SPDProcessPolygons.h
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


#ifndef SPDProcessPolygons_H
#define SPDProcessPolygons_H

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
#include "spd/SPDFileIncrementalReader.h"

namespace spdlib
{	
	class SPDProcessPolygons
	{
	public:
		SPDProcessPolygons(SPDPolygonProcessor *processor);
		void processPolygons(SPDFile *spdFile, SPDFileIncrementalReader *spdReader, OGRLayer *inputLayer, OGRLayer *outputLayer, bool copyAttributes)throw(SPDProcessingException);
        void processPolygons(SPDFile *spdFile, SPDFileIncrementalReader *spdReader, OGRLayer *inputLayer, std::ofstream *outASCIIFile)throw(SPDProcessingException);
		~SPDProcessPolygons();
	private:
		SPDPolygonProcessor *processor;
		void copyFeatureDefn(OGRLayer *outputSHPLayer, OGRFeatureDefn *inFeatureDefn) throw(SPDProcessingException);
		void copyFeatureData(OGRFeature *inFeature, OGRFeature *outFeature, OGRFeatureDefn *inFeatureDefn, OGRFeatureDefn *outFeatureDefn) throw(SPDProcessingException);
	};
}

#endif

