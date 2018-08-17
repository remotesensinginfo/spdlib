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

#include <iostream>
#include <list>
#include <vector>

#include <boost/cstdint.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include "ogrsf_frmts.h"

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDCommon.h"

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
	class DllExport SPDGridData
	{
	public:
		SPDGridData();
		std::list<SPDPulse*>*** gridData(std::list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
		std::list<SPDPulse*>*** gridData(std::vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
        void gridData(std::list<SPDPulse*>* pls, SPDFile *spdFile, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridData(std::vector<SPDPulse*>* pls, SPDFile *spdFile, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
		void reGridData(boost::uint_fast16_t indexType, std::vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, std::vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double originX, double originY, float outBinSize) throw(SPDProcessingException);

        void cartGridDataIgnoringOutGrid(std::vector<SPDPulse*>* pls, std::vector<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);

        ~SPDGridData();
	private:
		std::list<SPDPulse*>*** gridDataCartesian(std::list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
		std::list<SPDPulse*>*** gridDataSpherical(std::list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
        std::list<SPDPulse*>*** gridDataCylindrical(std::list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
        std::list<SPDPulse*>*** gridDataPolar(std::list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
		std::list<SPDPulse*>*** gridDataScan(std::list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
        std::list<SPDPulse*>*** gridDataCartesian(std::vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
		std::list<SPDPulse*>*** gridDataSpherical(std::vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
        std::list<SPDPulse*>*** gridDataCylindrical(std::vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
        std::list<SPDPulse*>*** gridDataPolar(std::vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);
        std::list<SPDPulse*>*** gridDataScan(std::vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException);

        void gridDataCartesian(std::list<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataSpherical(std::list<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataCylindrical(std::list<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataPolar(std::list<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataScan(std::list<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataCartesian(std::vector<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataSpherical(std::vector<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataCylindrical(std::vector<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataPolar(std::vector<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);
        void gridDataScan(std::vector<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException);

        void reGridDataCartesian(std::vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, std::vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double tlX, double tlY, float outBinSize) throw(SPDProcessingException);
        void reGridDataSpherical(std::vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, std::vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double tlX, double tlY, float outBinSize) throw(SPDProcessingException);
        void reGridDataCylindrical(std::vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, std::vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double tlX, double tlY, float outBinSize) throw(SPDProcessingException);
        void reGridDataPolar(std::vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, std::vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double tlX, double tlY, float outBinSize) throw(SPDProcessingException);
        void reGridDataScan(std::vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, std::vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double tlX, double tlY, float outBinSize) throw(SPDProcessingException);

	};
}

#endif



