/*
 *  SPDVectorUtils.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 12/03/2011.
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
 *
 */


#ifndef SPDVectorUtils_H
#define SPDVectorUtils_H

#include <iostream>
#include <fstream>
#include <string>
#include <list>

#include "ogrsf_frmts.h"
#include "ogr_api.h"

#include <boost/cstdint.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"

using namespace std;

namespace spdlib
{	
	class SPDVectorUtils
	{
	public:
		SPDVectorUtils();
		OGRGeometryCollection* getGeometryCollection(string inputVector) throw(SPDIOException);
        string getLayerName(string filepath);
		~SPDVectorUtils();
	};
}

#endif
