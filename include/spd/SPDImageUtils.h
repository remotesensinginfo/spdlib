/*
 *  SPDImageUtils.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 11/02/2011.
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


#ifndef SPDImageUtils_H
#define SPDImageUtils_H

#include <iostream>
#include <string>
#include <list>

#include "gdal_priv.h"
#include "ogrsf_frmts.h"
#include "gdal_rat.h"

#include <boost/cstdint.hpp>

#include "spd/SPDImageException.h"
#include "spd/SPDTextFileUtilities.h"

#include "boost/math/special_functions/fpclassify.hpp"

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
	
	class DllExport SPDImageUtils
	{
	public:
		SPDImageUtils();
		void getImagePixelValues(GDALDataset *dataset,boost::uint_fast32_t imgX,boost::uint_fast32_t imgY, float **pxlVals,boost::uint_fast32_t winHSize,boost::uint_fast16_t band) ;
		void getImagePixelPtValues(GDALDataset *dataset,boost::int_fast32_t *imgX,boost::int_fast32_t *imgY, float **pxlVals,boost::uint_fast32_t winHSize,boost::uint_fast16_t band) ;
		void getPixelLocation(GDALDataset *dataset, double x, double y, std::string wktStrBBox,boost::uint_fast32_t *imgX,boost::uint_fast32_t *imgY, float *xOff, float *yOff) ;
		void getPixelPointLocations(GDALDataset *dataset, double x, double y, std::string wktStrBBox,boost::int_fast32_t *imgX,boost::int_fast32_t *imgY, float *xOff, float *yOff) ;
		/**
		 * For a cubic interpolation 4 data values are required. Therefore, winSize must equal 4.
		 */ 
		float cubicInterpValue(float xShift, float yShift, float **pixels,boost::uint_fast32_t winSize) ;
        
        void getImageOverlapCut2Env(GDALDataset **datasets, int numDS,  int **dsOffsets, int *width, int *height, double *gdalTransform, OGREnvelope *env, int *maxBlockX, int *maxBlockY) ;
        bool closeResTest(double baseRes, double targetRes, double resDiffThresh=0.0001);
        void copyInDatasetIntoOutDataset(GDALDataset *dataset, GDALDataset *outputImageDS, OGREnvelope *env) ;
		boost::uint_fast32_t findColumnIndex(const GDALRasterAttributeTable *gdalATT, std::string colName) ;
        boost::uint_fast32_t findColumnIndexOrCreate(GDALRasterAttributeTable *gdalATT, std::string colName, GDALRATFieldType dType) ;
        ~SPDImageUtils();
	private:
		/**
		 * For a cubic interpolation 4 data values are required. Therefore, pixels must have length 4.
		 */ 
		float cubicEstValueFromCurve(float *pixels, float shift);
	};
}

#endif



