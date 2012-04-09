 /*
  *  SPDGridData.h
  *  spdlib
  *
  *  Created by Pete Bunting on 28/11/2010.
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


#ifndef SPDGridData_H
#define SPDGridData_H

#include <string>
#include <iostream>
#include <list>

#include <boost/cstdint.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include "ogrsf_frmts.h"

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDCommon.h"

using namespace std;
using boost::numeric_cast;
using boost::numeric::bad_numeric_cast;
using boost::numeric::positive_overflow;
using boost::numeric::negative_overflow;

namespace spdlib
{
	class SPDGridData
	{
	public:
		SPDGridData();
		list<SPDPulse*>*** gridData(list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
		list<SPDPulse*>*** gridData(vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
        void gridData(list<SPDPulse*>* pls, SPDFile *spdFile, list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridData(vector<SPDPulse*>* pls, SPDFile *spdFile, list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
		void reGridData(boost::uint_fast16_t indexType, vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double originX, double originY, float outBinSize) throw(SPDProcessingException);
        ~SPDGridData();
	private:
		list<SPDPulse*>*** gridDataCartesian(list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
		list<SPDPulse*>*** gridDataSpherical(list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
        list<SPDPulse*>*** gridDataCylindrical(list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
        list<SPDPulse*>*** gridDataPolar(list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
		list<SPDPulse*>*** gridDataScan(list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
        list<SPDPulse*>*** gridDataCartesian(vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
		list<SPDPulse*>*** gridDataSpherical(vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
        list<SPDPulse*>*** gridDataCylindrical(vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
        list<SPDPulse*>*** gridDataPolar(vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
        list<SPDPulse*>*** gridDataScan(vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
        
        void gridDataCartesian(list<SPDPulse*>* pls, list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataSpherical(list<SPDPulse*>* pls, list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataCylindrical(list<SPDPulse*>* pls, list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataPolar(list<SPDPulse*>* pls, list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataScan(list<SPDPulse*>* pls, list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataCartesian(vector<SPDPulse*>* pls, list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataSpherical(vector<SPDPulse*>* pls, list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataCylindrical(vector<SPDPulse*>* pls, list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataPolar(vector<SPDPulse*>* pls, list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataScan(vector<SPDPulse*>* pls, list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        
        void reGridDataCartesian(vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double tlX, double tlY, float outBinSize) throw(SPDProcessingException);
        void reGridDataSpherical(vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double tlX, double tlY, float outBinSize) throw(SPDProcessingException);
        void reGridDataCylindrical(vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double tlX, double tlY, float outBinSize) throw(SPDProcessingException);
        void reGridDataPolar(vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double tlX, double tlY, float outBinSize) throw(SPDProcessingException);
        void reGridDataScan(vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double tlX, double tlY, float outBinSize) throw(SPDProcessingException);
        
	};
}

#endif



