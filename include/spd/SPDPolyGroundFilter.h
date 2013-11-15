/*
 *  SPDPolyGroundFilter.h
 *
 *  Created by Pete Bunting on 04/03/2012.
 *  Copyright 2012 SPDLib. All rights reserved.
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

#ifndef SPDPolyGroundFilter_H
#define SPDPolyGroundFilter_H

#include <iostream>
#include <string>
#include <list>
#include "math.h"

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDProcessPulses.h"
#include "spd/SPDPulseProcessor.h"
#include "spd/SPDProcessingException.h"

#include <boost/math/special_functions/fpclassify.hpp>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_blas.h>

namespace spdlib
{
	
	class SPDFindMinReturnsProcessor : public SPDPulseProcessor
	{
	public:
        SPDFindMinReturnsProcessor(std::vector<SPDPoint*> *minPts, boost::uint_fast16_t ptSelectClass)
        {
            this->minPts = minPts;
            this->ptSelectClass = ptSelectClass;
        };
        
        void processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
		{throw SPDProcessingException("Processing with an output image is not implemented.");};
        
        void processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException)
        {
            SPDPoint *minPt = NULL;
            float minZ = 0;
            bool first = true;
            
            for(std::vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
            {
                if((*iterPulses)->numberOfReturns > 0)
                {
                    for(std::vector<SPDPoint*>::iterator iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
                    {
                        if(ptSelectClass == SPD_ALL_CLASSES)
                        {
                            if(first)
                            {
                                minPt = *iterPoints;
                                minZ = (*iterPoints)->z;
                                first = false;
                            }
                            else if((*iterPoints)->z < minZ)
                            {
                                minPt = *iterPoints;
                                minZ = (*iterPoints)->z;
                            }
                        }
                        else if((*iterPoints)->classification == ptSelectClass)
                        {
                            if(first)
                            {
                                minPt = *iterPoints;
                                minZ = (*iterPoints)->z;
                                first = false;
                            }
                            else if((*iterPoints)->z < minZ)
                            {
                                minPt = *iterPoints;
                                minZ = (*iterPoints)->z;
                            }
                        }
                    }
                }
            }
            
            if(!first)
            {
                SPDPoint *pt = new SPDPoint();
                SPDPointUtils ptUtils;
                ptUtils.copySPDPointTo(minPt, pt);
                minPts->push_back(pt);
            }
        };
        
        void processDataWindowImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageData, SPDXYPoint ***cenPts, boost::uint_fast32_t numImgBands, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
        
		void processDataWindow(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
        
        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {
            return std::vector<std::string>();
        };
        
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
        {};
        
        ~SPDFindMinReturnsProcessor(){};
    protected:
        std::vector<SPDPoint*> *minPts;
        boost::uint_fast16_t ptSelectClass;
	};

    
    class SPDClassifyGrdReturnsFromSurfaceCoefficientsProcessor : public SPDPulseProcessor
	{
	public:
        SPDClassifyGrdReturnsFromSurfaceCoefficientsProcessor(float grdThres, boost::uint_fast16_t degree, boost::uint_fast16_t iters, gsl_vector *coefficients, boost::uint_fast16_t ptSelectClass)
        {
            this->grdThres = grdThres;
            this->degree = degree;
            this->iters = iters;
            this->coefficients = coefficients;
            this->ptSelectClass = ptSelectClass;
        };
        
        void processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
		{throw SPDProcessingException("Processing with an output image is not implemented.");};
        
        void processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException)
        {
            for(std::vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
            {
                if((*iterPulses)->numberOfReturns > 0)
                {
                    for(std::vector<SPDPoint*>::iterator iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
                    {
                        if(ptSelectClass == SPD_ALL_CLASSES)
                        {
                            // Remove any existing ground return classification.
                            if((*iterPoints)->classification == SPD_GROUND)
                            {
                                (*iterPoints)->classification = SPD_UNCLASSIFIED;
                            }
                            
                            // Calc surface height for return
                            double xcoord= (*iterPoints)->x;
                            double ycoord= (*iterPoints)->y;
                            double zcoord= (*iterPoints)->z;
                            double surfaceValue=0; // reset z value from surface coefficients
                            boost::uint_fast32_t l=0;
                            
                            for (boost::uint_fast32_t m = 0; m < coefficients->size ; m++)
                            {
                                for (boost::uint_fast32_t n=0; n < coefficients->size ; n++)
                                {
                                    if (n+m <= degree)
                                    {
                                        double xelementtPow = pow(xcoord, ((int)(m)));
                                        double yelementtPow = pow(ycoord, ((int)(n)));
                                        double outm = gsl_vector_get(coefficients, l);
                                        
                                        surfaceValue=surfaceValue+(outm*xelementtPow*yelementtPow);
                                        ++l;
                                    }
                                }
                            }
                            
                            // Is return height less than surface height + grdThres
                            // sqrt((zcoord-surfaceValue)*(zcoord-surfaceValue)) <= grdThres
                            
                            if ((zcoord-surfaceValue) <= grdThres) {
                                (*iterPoints)->classification = SPD_GROUND;
                            }
                        }
                        else if(ptSelectClass == (*iterPoints)->classification)
                        {
                            // Remove any existing ground return classification.
                            if((*iterPoints)->classification == ptSelectClass)
                            {
                                (*iterPoints)->classification = SPD_UNCLASSIFIED;
                            }
                            
                            // Calc surface height for return
                            double xcoord= (*iterPoints)->x;
                            double ycoord= (*iterPoints)->y;
                            double zcoord= (*iterPoints)->z;
                            double surfaceValue=0; // reset z value from surface coefficients
                            boost::uint_fast32_t l=0;
                            
                            for (boost::uint_fast32_t m = 0; m < coefficients->size ; m++)
                            {
                                for (boost::uint_fast32_t n=0; n < coefficients->size ; n++)
                                {
                                    if (n+m <= degree)
                                    {
                                        double xelementtPow = pow(xcoord, ((int)(m)));
                                        double yelementtPow = pow(ycoord, ((int)(n)));
                                        double outm = gsl_vector_get(coefficients, l);
                                        
                                        surfaceValue=surfaceValue+(outm*xelementtPow*yelementtPow);
                                        ++l;
                                    }
                                }
                            }
                            
                            // Is return height less than surface height + grdThres
                            // sqrt((zcoord-surfaceValue)*(zcoord-surfaceValue)) <= grdThres
                            
                            if ((zcoord-surfaceValue) <= grdThres) {
                                (*iterPoints)->classification = SPD_GROUND;
                            }
                        }
                        
                    }
                }
            }
        };
        
        void processDataWindowImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageData, SPDXYPoint ***cenPts, boost::uint_fast32_t numImgBands, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
        
		void processDataWindow(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
        
        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {
            return std::vector<std::string>();
        };
        
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
        {};
        
        ~SPDClassifyGrdReturnsFromSurfaceCoefficientsProcessor(){};
    protected:
        float grdThres;
        boost::uint_fast16_t degree;
        boost::uint_fast16_t iters;
        gsl_vector *coefficients;
        boost::uint_fast16_t ptSelectClass;
	};
    
    
    class SPDPolyFitGroundLocalFilter : public SPDDataBlockProcessor
	{
	public:
        // constructor
        SPDPolyFitGroundLocalFilter(float grdThres, boost::uint_fast16_t degree, boost::uint_fast16_t iters, boost::uint_fast16_t ptSelectClass, float binWidth);
        
        // public functions
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
		{throw SPDProcessingException("SPDPolyFitGroundLocalFilter processDataBlockImage not implemented.");};
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException);
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) throw(SPDProcessingException)
		{throw SPDProcessingException("SPDPolyFitGroundLocalFilter requires processing with a grid.");};
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses) throw(SPDProcessingException)
        {throw SPDProcessingException("SPDPolyFitGroundLocalFilter requires processing with a grid.");};
        
        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {
            std::vector<std::string> bandNames;
            return bandNames;
        }
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
        {
            // Nothing to do...
        }
        
        ~SPDPolyFitGroundLocalFilter();
    protected:
        float grdThres;
        boost::uint_fast16_t degree;
        boost::uint_fast16_t iters;
        boost::uint_fast16_t ptSelectClass;
        float binWidth;
	};


    
    class SPDPolyFitGroundFilter
	{
	public:
		SPDPolyFitGroundFilter();
        void applyGlobalPolyFitGroundFilter(std::string inputFile, std::string outputFile, float grdThres, boost::uint_fast16_t degree, boost::uint_fast16_t iters, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize, float processingResolution, boost::uint_fast16_t ptSelectClass)throw(SPDProcessingException);
        void applyLocalPolyFitGroundFilter(std::string inputFile, std::string outputFile, float grdThres, boost::uint_fast16_t degree, boost::uint_fast16_t iters, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize, boost::uint_fast32_t overlap, float processingResolution, boost::uint_fast16_t ptSelectClass)throw(SPDProcessingException);
		~SPDPolyFitGroundFilter();
    private:
        void buildMinGrid(SPDFile *spdFile, std::vector<SPDPoint*> *minPts, std::vector<SPDPoint*> ***minPtGrid)throw(SPDProcessingException);
	};
    
    
}


#endif


