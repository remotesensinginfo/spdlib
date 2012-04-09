/*
 *  SPDImageUtils.cpp
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

#include "spd/SPDImageUtils.h"

namespace spdlib
{

	SPDImageUtils::SPDImageUtils()
	{
		
	}
	
	void SPDImageUtils::getImagePixelValues(GDALDataset *dataset, boost::uint_fast32_t imgX, boost::uint_fast32_t imgY, float **pxlVals, boost::uint_fast32_t winHSize, boost::uint_fast16_t band) throw(SPDImageException)
	{
		const char *val = "NaN";
		
		boost::uint_fast32_t xSize = dataset->GetRasterXSize();
		boost::uint_fast32_t ySize = dataset->GetRasterYSize();
		boost::uint_fast32_t winSize = (winHSize * 2) + 1;
		
		boost::uint_fast32_t tlX = imgX - winHSize;
		boost::uint_fast32_t tlY = imgY + winHSize;
		boost::uint_fast32_t brX = imgX + winHSize;
		boost::uint_fast32_t brY = imgY - winHSize;
		
		GDALRasterBand *rasterBand = dataset->GetRasterBand(band);
		float *data = (float *) CPLMalloc(sizeof(float)*xSize);
		
		for(boost::uint_fast32_t row = tlY, i = 0; row >= brY; --row, ++i)
		{
			if(row < 0)
			{
				for(boost::uint_fast32_t j = 0; j < winSize; ++j)
				{
					pxlVals[i][j] = nan(val);
				}
			}
			else if(row >= ySize)
			{
				for(boost::uint_fast32_t j = 0; j < winSize; ++j)
				{
					pxlVals[i][j] = nan(val);
				}
			}
			else 
			{
				rasterBand->RasterIO(GF_Read, 0, row, xSize, 1, data, xSize, 1, GDT_Float32, 0, 0);
				for(boost::uint_fast32_t col = tlX, j = 0; col <= brX; ++col, ++j)
				{
					if(col < 0)
					{
						pxlVals[i][j] = nan(val);
					}
					else if(col >= xSize)
					{
						pxlVals[i][j] = nan(val);
					}
					else 
					{
						pxlVals[i][j] = data[col];
					}
				}
			}			
		}
		
		delete[] data;
	}
	
	void SPDImageUtils::getImagePixelPtValues(GDALDataset *dataset,boost::int_fast32_t *imgX,boost::int_fast32_t *imgY, float **pxlVals, boost::uint_fast32_t winHSize, boost::uint_fast16_t band) throw(SPDImageException)
	{
		boost::uint_fast32_t xSize = dataset->GetRasterXSize();
		boost::uint_fast32_t ySize = dataset->GetRasterYSize();
		
	boost::int_fast32_t tlX = imgX[0] - (winHSize - 1);
	boost::int_fast32_t tlY = imgY[0] + (winHSize - 1);
	boost::int_fast32_t brX = imgX[3] + (winHSize - 1);
	boost::int_fast32_t brY = imgY[3] - (winHSize - 1);
		
		GDALRasterBand *rasterBand = dataset->GetRasterBand(band);
		float *data = (float *) CPLMalloc(sizeof(float)*xSize);
		
		for(int_fast32_t row = tlY, i = 0; row >= brY; --row, ++i)
		{
			if(row < 0)
			{
				rasterBand->RasterIO(GF_Read, 0, 0, xSize, 1, data, xSize, 1, GDT_Float32, 0, 0);
				for(int_fast32_t col = tlX, j = 0; col <= brX; ++col, ++j)
				{
					if(col < 0)
					{
						pxlVals[i][j] = data[0];
					}
					else if(((boost::uint_fast32_t)col) >= xSize)
					{
						pxlVals[i][j] = data[xSize-1];
					}
					else 
					{
						pxlVals[i][j] = data[col];
					}
				}
			}
			else if(((boost::uint_fast32_t)row) >= ySize)
			{
				rasterBand->RasterIO(GF_Read, 0, ySize-1, xSize, 1, data, xSize, 1, GDT_Float32, 0, 0);
				for(int_fast32_t col = tlX, j = 0; col <= brX; ++col, ++j)
				{
					if(col < 0)
					{
						pxlVals[i][j] = data[0];
					}
					else if(((boost::uint_fast32_t)col) >= xSize)
					{
						pxlVals[i][j] = data[xSize-1];
					}
					else 
					{
						pxlVals[i][j] = data[col];
					}
				}
			}
			else 
			{
				rasterBand->RasterIO(GF_Read, 0, row, xSize, 1, data, xSize, 1, GDT_Float32, 0, 0);
				for(int_fast32_t col = tlX, j = 0; col <= brX; ++col, ++j)
				{
					if(col < 0)
					{
						pxlVals[i][j] = data[0];
					}
					else if(((boost::uint_fast32_t)col) >= xSize)
					{
						pxlVals[i][j] = data[xSize-1];
					}
					else 
					{
						pxlVals[i][j] = data[col];
					}
				}
			}			
		}
		
		delete[] data;
	}
	
	void SPDImageUtils::getPixelLocation(GDALDataset *dataset, double x, double y, string wktStrBBox, boost::uint_fast32_t *imgX, boost::uint_fast32_t *imgY, float *xOff, float *yOff) throw(SPDImageException)
	{
		try 
		{
			double projX = 0;
			double projY = 0;
			
			// Check whether the projections are the same..
			// If different convert.
			const char *dataProjRef = wktStrBBox.c_str();
			const char *imgProjRef = dataset->GetProjectionRef();
			OGRSpatialReference *imgSpatialRef = new OGRSpatialReference(imgProjRef);
			OGRSpatialReference *dataSpatialRef = new OGRSpatialReference(dataProjRef);
			
			if(imgSpatialRef == dataSpatialRef)
			{
				projX = x;
				projY = y;
			}
			else
			{
				OGRCoordinateTransformation *ogrTransform = OGRCreateCoordinateTransformation(dataSpatialRef, imgSpatialRef);
				if(ogrTransform == NULL)
				{
					cout << "Image Projection: " << imgProjRef << endl;
					cout << "Data Projection: " << dataProjRef << endl;
					throw SPDImageException("A transformation between the projections could not be created.");
				}
				projX = x;
				projY = y;
				if(!ogrTransform->Transform(1, &projX, &projY))
				{
					SPDTextFileUtilities textUtils;
					string message = string("The transformation failed from [") + textUtils.doubletostring(x) + string(", ") + textUtils.doubletostring(y) + "] to [" + textUtils.doubletostring(projX) + string(", ") + textUtils.doubletostring(projY) + string("]");
					throw SPDImageException(message);
				}
				OGRCoordinateTransformation::DestroyCT(ogrTransform);
			}
			
			//cout << "Pulse Transformed: [" << projX << "," << projY << "]\n";
			
			// Find point location
			double *transformation = new double[6];
			dataset->GetGeoTransform(transformation);
			*imgX = floor((projX - transformation[0]) / transformation[1]);
			*imgY = floor((transformation[3] - projY) / transformation[1]);
			
			double pxlCentreX = transformation[0] + (((*imgX) * transformation[1]) + transformation[1]/2);
			double pxlCentreY = transformation[3] - (((*imgY) * transformation[1]) + transformation[1]/2);
			
			*xOff = (projX - pxlCentreX) / transformation[1];
			*yOff = (projY - pxlCentreY) / transformation[1];
			
			OGRSpatialReference::DestroySpatialReference(imgSpatialRef);
			OGRSpatialReference::DestroySpatialReference(dataSpatialRef);
		}
		catch (SPDImageException &e) 
		{
			throw e;
		}
		
	}
	
	void SPDImageUtils::getPixelPointLocations(GDALDataset *dataset, double x, double y, string wktStrBBox,boost::int_fast32_t *imgX,boost::int_fast32_t *imgY, float *xOff, float *yOff) throw(SPDImageException)
	{
		try 
		{
			double projX = 0;
			double projY = 0;
			
			// Check whether the projections are the same..
			// If different convert.
			const char *dataProjRef = wktStrBBox.c_str();
			const char *imgProjRef = dataset->GetProjectionRef();
			OGRSpatialReference *imgSpatialRef = new OGRSpatialReference(imgProjRef);
			OGRSpatialReference *dataSpatialRef = new OGRSpatialReference(dataProjRef);
			
			if(imgSpatialRef == dataSpatialRef)
			{
				projX = x;
				projY = y;
			}
			else
			{
				OGRCoordinateTransformation *ogrTransform = OGRCreateCoordinateTransformation(dataSpatialRef, imgSpatialRef);
				if(ogrTransform == NULL)
				{
					cout << "Image Projection: " << imgProjRef << endl;
					cout << "Data Projection: " << dataProjRef << endl;
					throw SPDImageException("A transformation between the projections could not be created.");
				}
				projX = x;
				projY = y;
				if(!ogrTransform->Transform(1, &projX, &projY))
				{
					SPDTextFileUtilities textUtils;
					string message = string("The transformation failed from [") + textUtils.doubletostring(x) + string(", ") + textUtils.doubletostring(y) + "] to [" + textUtils.doubletostring(projX) + string(", ") + textUtils.doubletostring(projY) + string("]");
					throw SPDImageException(message);
				}
				OGRCoordinateTransformation::DestroyCT(ogrTransform);
			}
			
			//cout << "Pulse Transformed: [" << projX << "," << projY << "]\n";

			// Find point location
			double *transformation = new double[6];
			dataset->GetGeoTransform(transformation);
			
			//cout << "Image Transformation (TL): [" << transformation[0] << "," << transformation[3] << "]\n";
			//cout << "Image Resolution: " << transformation[1] << endl;
			
			imgX[0] = floor((projX - transformation[0]) / transformation[1]); // Left X
			imgY[0] = ceil((transformation[3] - projY) / transformation[1]);  // Top Y
			imgX[1] = ceil((projX - transformation[0]) / transformation[1]);  // Right X
			imgY[1] = ceil((transformation[3] - projY) / transformation[1]);  // Top Y
			imgX[2] = floor((projX - transformation[0]) / transformation[1]); // Left X
			imgY[2] = floor((transformation[3] - projY) / transformation[1]); // Bottom Y
			imgX[3] = ceil((projX - transformation[0]) / transformation[1]);  // Right X
			imgY[3] = floor((transformation[3] - projY) / transformation[1]); // Bottom Y
			
			//cout << "Image Pxl [0]: " << imgX[0] << ", " << imgY[0] << endl;
			//cout << "Image Pxl [1]: " << imgX[1] << ", " << imgY[1] << endl;
			//cout << "Image Pxl [2]: " << imgX[2] << ", " << imgY[2] << endl;
			//cout << "Image Pxl [3]: " << imgX[3] << ", " << imgY[3] << endl;
			
			double pxlCentreX = transformation[0] + (imgX[0] * transformation[1]);
			double pxlCentreY = transformation[3] - (imgY[0] * transformation[1]);
			
			*xOff = (projX - pxlCentreX) / transformation[1];
			*yOff = (projY - pxlCentreY) / transformation[1];
			
			OGRSpatialReference::DestroySpatialReference(imgSpatialRef);
			OGRSpatialReference::DestroySpatialReference(dataSpatialRef);
		}
		catch (SPDImageException &e) 
		{
			throw e;
		}
	}
	
	float SPDImageUtils::cubicInterpValue(float xShift, float yShift, float **pixels, boost::uint_fast32_t winSize) throw(SPDImageException)
	{
		if(winSize != 4)
		{
			throw SPDImageException("Window Size must equal 4 for a cubic interpolation.");
		}
		
		float newValue = 0;
		
		float *newPixels = new float[4];
		float *tmpPixels = new float[4];
		tmpPixels[0] = pixels[0][0];
		tmpPixels[1] = pixels[0][1];
		tmpPixels[2] = pixels[0][2];
		tmpPixels[3] = pixels[0][3];
		newPixels[0] = this->cubicEstValueFromCurve(tmpPixels, yShift);
		
		tmpPixels[0] = pixels[1][0];
		tmpPixels[1] = pixels[1][1];
		tmpPixels[2] = pixels[1][2];
		tmpPixels[3] = pixels[1][3];
		newPixels[1] = this->cubicEstValueFromCurve(tmpPixels, yShift);
		
		tmpPixels[0] = pixels[2][0];
		tmpPixels[1] = pixels[2][1];
		tmpPixels[2] = pixels[2][2];
		tmpPixels[3] = pixels[2][3];
		newPixels[2] = this->cubicEstValueFromCurve(tmpPixels, yShift);
		
		tmpPixels[0] = pixels[3][0];
		tmpPixels[1] = pixels[3][1];
		tmpPixels[2] = pixels[3][2];
		tmpPixels[3] = pixels[3][3];
		newPixels[3] = this->cubicEstValueFromCurve(tmpPixels, yShift);
		
		newValue = this->cubicEstValueFromCurve(newPixels, xShift);	
		
		delete[] newPixels;
		delete[] tmpPixels;
		
		return newValue;
	}
	
	float SPDImageUtils::cubicEstValueFromCurve(float *pixels, float shift)
	{
		float newValue = 0;
		////////////// Fit line /////////////////////
		double a0 = 0;
		double a1 = 0;
		double a2 = 0;
		double a3 = 0;
		double shiftSq = 0;
		
		shiftSq = shift * shift;
		
		a0 = pixels[3] - pixels[2] - pixels[0] + pixels[1];
		a1 = pixels[0] - pixels[1] - a0;
		a2 = pixels[2] - pixels[0];
		a3 = pixels[1];
		///////////////////////////////////////////////
		
		/////////////// Find new value /////////////
		newValue = ((a0 * shift * shiftSq) + (a1 * shiftSq) + (a2 * shift) + a3);
		///////////////////////////////////////////
		
		return newValue;
		
	}
	
	SPDImageUtils::~SPDImageUtils()
	{
		
	}
}

