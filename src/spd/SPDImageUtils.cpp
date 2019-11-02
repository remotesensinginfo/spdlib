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
	
	void SPDImageUtils::getImagePixelValues(GDALDataset *dataset, boost::uint_fast32_t imgX, boost::uint_fast32_t imgY, float **pxlVals, boost::uint_fast32_t winHSize, boost::uint_fast16_t band) 
	{		
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
			if(row >= ySize)
			{
				for(boost::uint_fast32_t j = 0; j < winSize; ++j)
				{
					pxlVals[i][j] = std::numeric_limits<float>::signaling_NaN();
				}
			}
			else 
			{
				rasterBand->RasterIO(GF_Read, 0, row, xSize, 1, data, xSize, 1, GDT_Float32, 0, 0);
				for(boost::uint_fast32_t col = tlX, j = 0; col <= brX; ++col, ++j)
				{
					if(col >= xSize)
					{
						pxlVals[i][j] = std::numeric_limits<float>::signaling_NaN();
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
	
	void SPDImageUtils::getImagePixelPtValues(GDALDataset *dataset,boost::int_fast32_t *imgX,boost::int_fast32_t *imgY, float **pxlVals, boost::uint_fast32_t winHSize, boost::uint_fast16_t band) 
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
	
	void SPDImageUtils::getPixelLocation(GDALDataset *dataset, double x, double y, std::string wktStrBBox, boost::uint_fast32_t *imgX, boost::uint_fast32_t *imgY, float *xOff, float *yOff) 
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
					std::cout << "Image Projection: " << imgProjRef << std::endl;
					std::cout << "Data Projection: " << dataProjRef << std::endl;
					throw SPDImageException("A transformation between the projections could not be created.");
				}
				projX = x;
				projY = y;
				if(!ogrTransform->Transform(1, &projX, &projY))
				{
					SPDTextFileUtilities textUtils;
					std::string message = std::string("The transformation failed from [") + textUtils.doubletostring(x) + std::string(", ") + textUtils.doubletostring(y) + "] to [" + textUtils.doubletostring(projX) + std::string(", ") + textUtils.doubletostring(projY) + std::string("]");
					throw SPDImageException(message);
				}
				OGRCoordinateTransformation::DestroyCT(ogrTransform);
			}
			
			//std::cout << "Pulse Transformed: [" << projX << "," << projY << "]\n";
			
			// Find point location
			double *transformation = new double[6];
			dataset->GetGeoTransform(transformation);
			*imgX = floor(((double)(projX - transformation[0]) / transformation[1]));
			*imgY = floor(((double)(transformation[3] - projY) / transformation[1]));
			
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
	
	void SPDImageUtils::getPixelPointLocations(GDALDataset *dataset, double x, double y, std::string wktStrBBox,boost::int_fast32_t *imgX,boost::int_fast32_t *imgY, float *xOff, float *yOff) 
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
					std::cout << "Image Projection: " << imgProjRef << std::endl;
					std::cout << "Data Projection: " << dataProjRef << std::endl;
					throw SPDImageException("A transformation between the projections could not be created.");
				}
				projX = x;
				projY = y;
				if(!ogrTransform->Transform(1, &projX, &projY))
				{
					SPDTextFileUtilities textUtils;
					std::string message = std::string("The transformation failed from [") + textUtils.doubletostring(x) + std::string(", ") + textUtils.doubletostring(y) + "] to [" + textUtils.doubletostring(projX) + std::string(", ") + textUtils.doubletostring(projY) + std::string("]");
					throw SPDImageException(message);
				}
				OGRCoordinateTransformation::DestroyCT(ogrTransform);
			}
			
			//std::cout << "Pulse Transformed: [" << projX << "," << projY << "]\n";

			// Find point location
			double *transformation = new double[6];
			dataset->GetGeoTransform(transformation);
			
			//std::cout << "Image Transformation (TL): [" << transformation[0] << "," << transformation[3] << "]\n";
			//std::cout << "Image Resolution: " << transformation[1] << std::endl;
			
			imgX[0] = floor(((double)(projX - transformation[0]) / transformation[1])); // Left X
			imgY[0] = ceil(((double)(transformation[3] - projY) / transformation[1]));  // Top Y
			imgX[1] = ceil(((double)(projX - transformation[0]) / transformation[1]));  // Right X
			imgY[1] = ceil(((double)(transformation[3] - projY) / transformation[1]));  // Top Y
			imgX[2] = floor(((double)(projX - transformation[0]) / transformation[1])); // Left X
			imgY[2] = floor(((double)(transformation[3] - projY) / transformation[1])); // Bottom Y
			imgX[3] = ceil(((double)(projX - transformation[0]) / transformation[1]));  // Right X
			imgY[3] = floor(((double)(transformation[3] - projY) / transformation[1])); // Bottom Y
			
			//std::cout << "Image Pxl [0]: " << imgX[0] << ", " << imgY[0] << std::endl;
			//std::cout << "Image Pxl [1]: " << imgX[1] << ", " << imgY[1] << std::endl;
			//std::cout << "Image Pxl [2]: " << imgX[2] << ", " << imgY[2] << std::endl;
			//std::cout << "Image Pxl [3]: " << imgX[3] << ", " << imgY[3] << std::endl;
			
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
	
	float SPDImageUtils::cubicInterpValue(float xShift, float yShift, float **pixels, boost::uint_fast32_t winSize) 
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
    
    void SPDImageUtils::getImageOverlapCut2Env(GDALDataset **datasets, int numDS,  int **dsOffsets, int *width, int *height, double *gdalTransform, OGREnvelope *env, int *maxBlockX, int *maxBlockY) 
	{
		double **transformations = new double*[numDS];
		int *xSize = new int[numDS];
		int *ySize = new int[numDS];
        int *xBlockSize = new int[numDS];
		int *yBlockSize = new int[numDS];
		for(int i = 0; i < numDS; i++)
		{
			transformations[i] = new double[6];
			datasets[i]->GetGeoTransform(transformations[i]);
			xSize[i] = datasets[i]->GetRasterXSize();
			ySize[i] = datasets[i]->GetRasterYSize();
			datasets[i]->GetRasterBand(1)->GetBlockSize(&xBlockSize[i], &yBlockSize[i]);
			//std::cout << "TL [" << transformations[i][0] << "," << transformations[i][3] << "]\n";
		}
		double rotateX = 0;
		double rotateY = 0;
		double pixelXRes = 0;
		double pixelYRes = 0;
		double pixelYResPos = 0;
		double minX = 0;
		double maxX = 0;
		double tmpMaxX = 0;
		double minY = 0;
		double tmpMinY = 0;
		double maxY = 0;
		const char *proj = NULL;
		bool first = true;
		
		
		try
		{
			// Calculate Image Overlap.
			for(int i = 0; i < numDS; ++i)
			{
				if(first)
				{
                    *maxBlockX = xBlockSize[i];
                    *maxBlockY = yBlockSize[i];
                    
					pixelXRes = transformations[i][1];
					pixelYRes = transformations[i][5];
					
					rotateX = transformations[i][2];
					rotateY = transformations[i][4];
					
					if(pixelYRes < 0)
					{
						pixelYResPos = pixelYRes * (-1);
					}
					else
					{
						pixelYResPos = pixelYRes;
					}
					
					minX = transformations[i][0];
					maxY = transformations[i][3];
					
					maxX = minX + (xSize[i] * pixelXRes);
					minY = maxY - (ySize[i] * pixelYResPos);
					
					proj = datasets[i]->GetProjectionRef(); // Get projection of first band in image
					
					first = false;
				}
				else
				{
					if((this->closeResTest(pixelXRes, transformations[i][1]) == false) | (this->closeResTest(pixelYRes, transformations[i][5]) == false))
					{
						throw SPDImageException("Not all image bands have the same resolution..");
					}
					
					if(std::string(datasets[i]->GetProjectionRef()) != std::string(proj))
					{
						//std::cout << "Band 1 Projection = " << proj << std::endl;
						//std::cout << "Band " << i <<  " Projection = " << datasets[i]->GetProjectionRef()<< std::endl;
						throw SPDImageException("Not all image bands have the same projection..");
					}
					
					if(transformations[i][2] != rotateX & transformations[i][4] != rotateY)
					{
						throw SPDImageException("Not all image bands have the same rotation..");
					}
					
					if(transformations[i][0] > minX)
					{
						minX = transformations[i][0];
					}
					
					if(transformations[i][3] < maxY)
					{
						maxY = transformations[i][3];
					}
					
					tmpMaxX = transformations[i][0] + (xSize[i] * pixelXRes);
					tmpMinY = transformations[i][3] - (ySize[i] * pixelYResPos);
					
					if(tmpMaxX < maxX)
					{
						maxX = tmpMaxX;
					}
					
					if(tmpMinY > minY)
					{
						minY = tmpMinY;
					}
                    
                    if(xBlockSize[i] > (*maxBlockX))
                    {
                        *maxBlockX = xBlockSize[i];
                    }
                    
                    if(yBlockSize[i] > (*maxBlockY))
                    {
                        *maxBlockY = yBlockSize[i];
                    }
				}
			}
            
			if(maxX - minX <= 0)
			{
				std::cout << "MinX = " << minX << std::endl;
				std::cout << "MaxX = " << maxX << std::endl;
				throw SPDImageException("Images do not overlap in the X axis");
			}
			
			if(maxY - minY <= 0)
			{
				std::cout << "MinY = " << minY << std::endl;
				std::cout << "MaxY = " << maxY << std::endl;
				throw SPDImageException("Images do not overlap in the Y axis");
			}
			
			// Cut to env extent
			if(env->MinX > minX)
			{
                minX = env->MinX;
			}
			
			if(env->MinY > minY)
			{
				minY = env->MinY;
			}
			
			if(env->MaxX < maxX)
			{
				maxX = env->MaxX;
			}
			
			if(env->MaxY < maxY)
			{
				maxY = env->MaxY;
			}
			
            if(maxX - minX <= 0)
			{
				std::cout << "MinX = " << minX << std::endl;
				std::cout << "MaxX = " << maxX << std::endl;
				throw SPDImageException("Images and Envelope do not overlap in the X axis");
			}
			
			if(maxY - minY <= 0)
			{
				std::cout << "MinY = " << minY << std::endl;
				std::cout << "MaxY = " << maxY << std::endl;
				throw SPDImageException("Images and Envelope do not overlap in the Y axis");
			}
            
            gdalTransform[0] = minX;
			gdalTransform[1] = pixelXRes;
			gdalTransform[2] = rotateX;
			gdalTransform[3] = maxY;
			gdalTransform[4] = rotateY;
			gdalTransform[5] = pixelYRes;
			
            //std::cout << "(maxX - minX)/pixelXRes = " << (maxX - minX)/pixelXRes << std::endl;
            //std::cout << "(maxY - minY)/pixelYResPos = " << (maxY - minY)/pixelYResPos << std::endl;
            
			*width = floor(((maxX - minX)/pixelXRes)+0.5);
			*height = floor(((maxY - minY)/pixelYResPos)+0.5);
			
			double diffX = 0;
			double diffY = 0;
			
			for(int i = 0; i < numDS; i++)
			{
				diffX = minX - transformations[i][0];
				diffY = transformations[i][3] - maxY;
				
				if(!((diffX > -0.0001) & (diffX < 0.0001)))
				{
					dsOffsets[i][0] = floor((diffX/pixelXRes));
				}
				else
				{
					dsOffsets[i][0] = 0;
				}
				
				if(!((diffY > -0.0001) & (diffY < 0.0001)))
				{
					dsOffsets[i][1] = floor((diffY/pixelYResPos));
				}
				else
				{
					dsOffsets[i][1] = 0;
				}
			}
			
		}
		catch(SPDImageException& e)
		{
			if(transformations != NULL)
			{
				for(int i = 0; i < numDS; i++)
				{
					delete[] transformations[i];
				}
				delete[] transformations;
			}
			if(xSize != NULL)
			{
				delete[] xSize;
			}
			if(ySize != NULL)
			{
				delete[] ySize;
			}
			throw e;
		}
		
		if(transformations != NULL)
		{
			for(int i = 0; i < numDS; i++)
			{
				delete[] transformations[i];
			}
			delete[] transformations;
		}
		if(xSize != NULL)
		{
			delete[] xSize;
		}
		if(ySize != NULL)
		{
			delete[] ySize;
		}
        
        delete[] xBlockSize;
        delete[] yBlockSize;
	}
    
    bool SPDImageUtils::closeResTest(double baseRes, double targetRes, double resDiffThresh)
    {
    	/** Calculates if two doubles are close to each other with the threshold
    	 * defined in the class.
    	 * - A two sided test is used rather than the absolute value to prevent
    	 * 	 overflows.
    	 */
        
    	bool closeRes = true;
    	double resDiff = baseRes - targetRes;
    	double resDiffVal = resDiffThresh * baseRes;
        
    	if((resDiff > 0) && (resDiff > resDiffVal)){closeRes = false;}
    	else if((resDiff < 0) && (resDiff > -1.*resDiffVal)){closeRes = false;}
        
    	return closeRes;
    }
    
    
    void SPDImageUtils::copyInDatasetIntoOutDataset(GDALDataset *dataset, GDALDataset *outputImageDS, OGREnvelope *env) 
    {
        GDALAllRegister();
		double *gdalTranslation = new double[6];
		int **dsOffsets = new int*[2];
		for(int i = 0; i < 2; i++)
		{
			dsOffsets[i] = new int[2];
		}
		int *inImgOffset = NULL;
        int *outImgOffset = NULL;
		int height = 0;
		int width = 0;
		int numOfBands = 0;
		
		float **inputData = NULL;
        int xBlockSize = 0;
        int yBlockSize = 0;
		
		GDALRasterBand **inputRasterBands = NULL;
		GDALRasterBand **outputRasterBands = NULL;
		
		try
		{            
            if(outputImageDS->GetRasterCount() != dataset->GetRasterCount())
            {
                throw SPDImageException("The input and output datasets do not have the same number of image bands\n");
            }
            
            
            GDALDataset **tmpDatasets = new GDALDataset*[2];
            tmpDatasets[0] = dataset;
            tmpDatasets[1] = outputImageDS;
            
            
			// Find image overlap
			this->getImageOverlapCut2Env(tmpDatasets, 2,  dsOffsets, &width, &height, gdalTranslation, env, &xBlockSize, &yBlockSize);
            
            //std::cout << "Overlap Image Width = " << width << std::endl;
            //std::cout << "Overlap Image Height = " << height << std::endl;
            
            delete[] tmpDatasets;
            
			if(width < 1)
            {
                throw SPDImageException("The output dataset does not have the correct width\n");
            }
            
            if(height < 1)
            {
                throw SPDImageException("The output dataset does not have the correct height\n");
            }
			
            numOfBands = dataset->GetRasterCount();
            
            
			// Get Image Input Bands
			inImgOffset = new int[2];
            inImgOffset[0] = dsOffsets[0][0];
            inImgOffset[1] = dsOffsets[0][1];
            //std::cout << "In Image Off: [" << inImgOffset[0] << ", " << inImgOffset[1] << "]\n";
			inputRasterBands = new GDALRasterBand*[numOfBands];
			for(int j = 0; j < numOfBands; j++)
            {
                inputRasterBands[j] = dataset->GetRasterBand(j+1);
            }
            
			//Get Image Output Bands
            outImgOffset = new int[2];
            outImgOffset[0] = dsOffsets[1][0];
            outImgOffset[1] = dsOffsets[1][1];
            //std::cout << "Out Image Off: [" << outImgOffset[0] << ", " << outImgOffset[1] << "]\n";
			outputRasterBands = new GDALRasterBand*[numOfBands];
			for(int i = 0; i < numOfBands; i++)
			{
				outputRasterBands[i] = outputImageDS->GetRasterBand(i+1);
			}
            
            //std::cout << "Max. block size: " << yBlockSize << std::endl;
            
			// Allocate memory
			inputData = new float*[numOfBands];
			for(int i = 0; i < numOfBands; i++)
			{
				inputData[i] = (float *) CPLMalloc(sizeof(float)*width*yBlockSize);
			}
            
			int nYBlocks = height / yBlockSize;
            int remainRows = height - (nYBlocks * yBlockSize);
            int rowOffset = 0;
            
			// Loop images to process data
			for(int i = 0; i < nYBlocks; i++)
			{                
				for(int n = 0; n < numOfBands; n++)
				{
                    rowOffset = inImgOffset[1] + (yBlockSize * i);
					inputRasterBands[n]->RasterIO(GF_Read, inImgOffset[0], rowOffset, width, yBlockSize, inputData[n], width, yBlockSize, GDT_Float32, 0, 0);
				}
				
				for(int n = 0; n < numOfBands; n++)
				{
                    rowOffset = outImgOffset[1] + (yBlockSize * i);
					outputRasterBands[n]->RasterIO(GF_Write, outImgOffset[0], rowOffset, width, yBlockSize, inputData[n], width, yBlockSize, GDT_Float32, 0, 0);
				}
			}
            
            if(remainRows > 0)
            {
                for(int n = 0; n < numOfBands; n++)
				{
                    rowOffset = inImgOffset[1] + (yBlockSize * nYBlocks);
					inputRasterBands[n]->RasterIO(GF_Read, inImgOffset[0], rowOffset, width, remainRows, inputData[n], width, remainRows, GDT_Float32, 0, 0);
				}
				
				for(int n = 0; n < numOfBands; n++)
				{
                    rowOffset = outImgOffset[1] + (yBlockSize * nYBlocks);
					outputRasterBands[n]->RasterIO(GF_Write, outImgOffset[0], rowOffset, width, remainRows, inputData[n], width, remainRows, GDT_Float32, 0, 0);
				}
            }
		}
		catch(SPDImageException& e)
		{
			if(gdalTranslation != NULL)
			{
				delete[] gdalTranslation;
			}
			
			if(dsOffsets != NULL)
			{
				for(int i = 0; i < 2; i++)
				{
					if(dsOffsets[i] != NULL)
					{
						delete[] dsOffsets[i];
					}
				}
				delete[] dsOffsets;
			}
			
			if(inImgOffset != NULL)
			{
				delete[] inImgOffset;
			}
            
            if(outImgOffset != NULL)
			{
				delete[] outImgOffset;
			}
			
			if(inputData != NULL)
			{
				for(int i = 0; i < numOfBands; i++)
				{
					if(inputData[i] != NULL)
					{
						delete[] inputData[i];
					}
				}
				delete[] inputData;
			}

			if(inputRasterBands != NULL)
			{
				delete[] inputRasterBands;
			}
			
			if(outputRasterBands != NULL)
			{
				delete[] outputRasterBands;
			}
			throw e;
		}
        
		if(gdalTranslation != NULL)
		{
			delete[] gdalTranslation;
		}
		
		if(dsOffsets != NULL)
		{
			for(int i = 0; i < 2; i++)
			{
				if(dsOffsets[i] != NULL)
				{
					delete[] dsOffsets[i];
				}
			}
			delete[] dsOffsets;
		}
		
		if(inImgOffset != NULL)
        {
            delete[] inImgOffset;
        }
        
        if(outImgOffset != NULL)
        {
            delete[] outImgOffset;
        }
		
		if(inputData != NULL)
		{
			for(int i = 0; i < numOfBands; i++)
			{
				if(inputData[i] != NULL)
				{
					CPLFree(inputData[i]);
				}
			}
			delete[] inputData;
		}
		
		if(inputRasterBands != NULL)
		{
			delete[] inputRasterBands;
		}
		
		if(outputRasterBands != NULL)
		{
			delete[] outputRasterBands;
		}
    }
    
    
    boost::uint_fast32_t SPDImageUtils::findColumnIndex(const GDALRasterAttributeTable *gdalATT, std::string colName) 
    {
        int numColumns = gdalATT->GetColumnCount();
        bool foundCol = false;
        boost::uint_fast32_t colIdx = 0;
        for(int i = 0; i < numColumns; ++i)
        {
            if(std::string(gdalATT->GetNameOfCol(i)) == colName)
            {
                foundCol = true;
                colIdx = i;
                break;
            }
        }
        
        if(!foundCol)
        {
            std::string message = std::string("The column ") + colName + std::string(" could not be found.");
            throw SPDImageException(message);
        }
        
        return colIdx;
    }
    
    boost::uint_fast32_t SPDImageUtils::findColumnIndexOrCreate(GDALRasterAttributeTable *gdalATT, std::string colName, GDALRATFieldType dType) 
    {
        int numColumns = gdalATT->GetColumnCount();
        bool foundCol = false;
        boost::uint_fast32_t colIdx = 0;
        for(int i = 0; i < numColumns; ++i)
        {
            if(std::string(gdalATT->GetNameOfCol(i)) == colName)
            {
                foundCol = true;
                colIdx = i;
                break;
            }
        }
        
        if(!foundCol)
        {
            gdalATT->CreateColumn(colName.c_str(), dType, GFU_Generic);
            colIdx = numColumns;
        }
        
        return colIdx;
    }
    
	
	SPDImageUtils::~SPDImageUtils()
	{
		
	}
}

