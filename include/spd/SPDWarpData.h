/*
 *  SPDWarpData.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 21/01/2013.
 *  Copyright 2013 SPDLib. All rights reserved.
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

#ifndef SPDWarpData_H
#define SPDWarpData_H

#include <iostream>
#include <string>
#include <fstream>
#include <list>
#include "math.h"

#include <boost/math/special_functions/fpclassify.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDPoint.h"
#include "spd/SPDProcessPulses.h"
#include "spd/SPDPulseProcessor.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDTextFileLineReader.h"
#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDMathsUtils.h"
#include "spd/SPDMatrixUtils.h"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Interpolation_traits_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/algorithm.h>
#include <CGAL/Origin.h>
#include <CGAL/squared_distance_2.h>

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
    class DllExport SPDShiftData : public SPDDataBlockProcessor
	{
	public:
        SPDShiftData(float xShift, float yShift);
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) 
		{throw SPDProcessingException("SPDMultiscaleCurvatureGrdClassification cannot output an image layer.");};
        
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) ;
        
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) 
		{throw SPDProcessingException("SPDMultiscaleCurvatureGrdClassification requires processing with a grid.");};
        
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses) 
        {throw SPDProcessingException("SPDMultiscaleCurvatureGrdClassification requires processing with a grid.");};
        
        std::vector<std::string> getImageBandDescriptions() 
        {
            std::vector<std::string> bandNames;
            return bandNames;
        }
        void setHeaderValues(SPDFile *spdFile) 
        {
            spdFile->setXMin(spdFile->getXMin()+xShift);
            spdFile->setXMax(spdFile->getXMax()+xShift);
            spdFile->setYMin(spdFile->getYMin()+yShift);
            spdFile->setYMax(spdFile->getYMax()+yShift);
        }
        
        ~SPDShiftData();
        
    protected:
        float xShift;
        float yShift;
	};
    
    class DllExport SPDGCPImg2MapNode
	{
	public:
		SPDGCPImg2MapNode(double eastings, double northings, float xOff, float yOff);
		double eastings() const;
		double northings() const;
		float xOff() const;
		float yOff() const;
		double distanceGeo(SPDGCPImg2MapNode *pt);
		~SPDGCPImg2MapNode();
	protected:
        double eastings_;
        double northings_;
		float xOff_;
		float yOff_;
	};
    
    
    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef K::FT                                         CGALCoordType;
    typedef K::Vector_2                                   CGALVector;
    typedef K::Point_2                                    CGALPoint;
    
    typedef CGAL::Delaunay_triangulation_2<K>             DelaunayTriangulation;
    typedef CGAL::Interpolation_traits_2<K>               InterpTraits;
    typedef CGAL::Delaunay_triangulation_2<K>::Vertex_handle    Vertex_handle;
    typedef CGAL::Delaunay_triangulation_2<K>::Face_handle    Face_handle;
    
    typedef std::vector< std::pair<CGALPoint, CGALCoordType> >   CoordinateVector;
    typedef std::map<CGALPoint, SPDGCPImg2MapNode*, K::Less_xy_2>     PointValueMap;
    
    
    class DllExport SPDWarpException : public SPDException
	{
	public:
		SPDWarpException(){msgs = "A SPDImageException has been created..";};
		SPDWarpException(const char *message): SPDException(message){};
		SPDWarpException(std::string message): SPDException(message){};
	};
    
    
    class DllExport SPDWarpPointData
    {
    public:
        SPDWarpPointData();
        virtual bool initWarp(std::string gcpFile)=0;
        virtual float calcXOffset(float eastings, float northings)=0;
        virtual float calcYOffset(float eastings, float northings)=0;
        virtual void calcOffset(float eastings, float northings, float *xOff, float *yOff);
        virtual ~SPDWarpPointData();
    protected:
        virtual void readGCPs(std::string gcpFile) ;
        std::vector<SPDGCPImg2MapNode*> *gcps;
    };
    
    class DllExport SPDNearestNeighbourWarp : public SPDWarpPointData
    {
    public:
        SPDNearestNeighbourWarp();
        virtual bool initWarp(std::string gcpFile);
        virtual float calcXOffset(float eastings, float northings);
        virtual float calcYOffset(float eastings, float northings);
        virtual void calcOffset(float eastings, float northings, float *xOff, float *yOff);
        virtual ~SPDNearestNeighbourWarp();
    protected:
        DelaunayTriangulation *dt;
        PointValueMap *values;
    };
    
    class DllExport SPDTriangulationPlaneFittingWarp : public SPDWarpPointData
    {
    public:
        SPDTriangulationPlaneFittingWarp();
        virtual bool initWarp(std::string gcpFile);
        virtual float calcXOffset(float eastings, float northings);
        virtual float calcYOffset(float eastings, float northings);
        virtual void calcOffset(float eastings, float northings, float *xOff, float *yOff);
        virtual ~SPDTriangulationPlaneFittingWarp();
    protected:
        DelaunayTriangulation *dt;
        PointValueMap *values;
        std::list<SPDGCPImg2MapNode*>* normGCPs(std::list<const SPDGCPImg2MapNode*> *gcps, double eastings, double northings);
		void fitPlane2XPoints(std::list<SPDGCPImg2MapNode*> *normPts, double *a, double *b, double *c) ;
		void fitPlane2YPoints(std::list<SPDGCPImg2MapNode*> *normPts, double *a, double *b, double *c) ;
    };
    
    
    class DllExport SPDPolynomialWarp : public SPDWarpPointData
    {
    public:
        SPDPolynomialWarp(int order);
        virtual bool initWarp(std::string gcpFile);
        virtual float calcXOffset(float eastings, float northings);
        virtual float calcYOffset(float eastings, float northings);
        virtual void calcOffset(float eastings, float northings, float *xOff, float *yOff);
        virtual ~SPDPolynomialWarp();
    protected:
        int polyOrder; // Polynominal order
        gsl_vector *aX;
        gsl_vector *aY;
        gsl_vector *aE;
        gsl_vector *aN;
    };
    
    
    enum SPDWarpLocation
    {
        spdwarppulseidx=1,
        spdwarpfromall=2,
        spdwarppulseorigin=3
    };
    
    class DllExport SPDNonLinearWarp : public SPDImporterProcessor
	{
	public:
		SPDNonLinearWarp(SPDDataExporter *exporter, SPDFile *spdFileOut, SPDWarpPointData *calcOffsets, SPDWarpLocation warpLoc) ;
		void processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) ;
		void completeFileAndClose(SPDFile *spdFile);
		~SPDNonLinearWarp();
	private:
		SPDDataExporter *exporter;
		SPDFile *spdFileOut;
        std::list<SPDPulse*> *pulses;
        SPDWarpPointData *calcOffsets;
        SPDWarpLocation warpLoc;
	};
    
     
}

#endif
