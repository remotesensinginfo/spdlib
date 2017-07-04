/*
 *  SPDMatrixUtils.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 31/01/2011.
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

#ifndef SPDMatrixUtils_H
#define SPDMatrixUtils_H

#include <string>
#include <iostream>
#include <fstream>
#include <math.h>

#include <boost/cstdint.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include "spd/SPDException.h"
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

namespace spdlib{
	/// Utilities for SPD matrices
	/**
	 * m - x axis
	 * n - y axis
	 */
	struct DllExport Matrix
	{
		double *matrix;
		int m;
		int n;
	};
	
	class DllExport SPDMatrixUtils
	{
	public:
		SPDMatrixUtils();
		Matrix* createMatrix(int n, int m) throw(SPDException);
		Matrix* createMatrix(Matrix *matrix) throw(SPDException);
		Matrix* copyMatrix(Matrix *matrix) throw(SPDException);
		void freeMatrix(Matrix *matrix);
		double determinant(Matrix *matrix) throw(SPDException);
		Matrix* cofactors(Matrix *matrix) throw(SPDException);
		Matrix* transpose(Matrix *matrix) throw(SPDException);
		void multipleSingle(Matrix *matrix, double multiple) throw(SPDException);
		Matrix* multiplication(Matrix *matrixA, Matrix *matrixB) throw(SPDException);
		void printMatrix(Matrix *matrix);
		void saveMatrix2GridTxt(Matrix *matrix, std::string filepath) throw(SPDException);
		void saveMatrix2CSV(Matrix *matrix, std::string filepath) throw(SPDException);
		void saveMatrix2txt(Matrix *matrix, std::string filepath) throw(SPDException);
		void saveMatrix2Binary(Matrix *matrix, std::string filepath) throw(SPDException);
		Matrix* readMatrixFromTxt(std::string filepath) throw(SPDException);
		Matrix* readMatrixFromGridTxt(std::string filepath) throw(SPDException);
		Matrix* readMatrixFromBinary(std::string filepath) throw(SPDException);
		Matrix* normalisedMatrix(Matrix *matrix, double min, double max) throw(SPDException);
		Matrix* duplicateMatrix(Matrix *matrix, int xDuplications, int yDuplications) throw(SPDException);
		~SPDMatrixUtils();
	};
}

#endif
