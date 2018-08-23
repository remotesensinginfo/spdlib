/*
 *  SPDPointInterpolation.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 02/03/2011.
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


#ifndef SPDPointInterpolation_H
#define SPDPointInterpolation_H

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <algorithm>

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDMatrixUtils.h"
#include "spd/SPDException.h"
#include "spd/SPDPointGridIndex.h"
#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDMathsUtils.h"

#include "spd/tps/spline.h"
#include "spd/tps/linalg3d.h"

#include "boost/math/special_functions/fpclassify.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Interpolation_traits_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/algorithm.h>
#include <CGAL/Origin.h>
#include <CGAL/squared_distance_2.h>

#include "spd/alglibspd/interpolation.h"
#include "spd/alglibspd/stdafx.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::FT                                         CGALCoordType;
typedef K::Vector_2                                   CGALVector;
typedef K::Point_2                                    CGALPoint;

typedef CGAL::Delaunay_triangulation_2<K>             DelaunayTriangulation;
typedef CGAL::Interpolation_traits_2<K>               InterpTraits;
typedef CGAL::Delaunay_triangulation_2<K>::Vertex_handle    Vertex_handle;
typedef CGAL::Delaunay_triangulation_2<K>::Face_handle    Face_handle;

typedef std::vector< std::pair<CGALPoint, CGALCoordType> >   CoordinateVector;
typedef std::map<CGALPoint, CGALCoordType, K::Less_xy_2>     PointValueMap;

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
	/**
	 * SPDPointInterpolation is an abstract interface for the interpolation of 
	 * SPDPoint/SPDPulse data.
	 *
	 */
	class DllExport SPDPointInterpolator
	{
	public:
		SPDPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		virtual void initInterpolator(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException) = 0;
		virtual void initInterpolator(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException) = 0;
		virtual void initInterpolator(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException) = 0;
		virtual void initInterpolator(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException) = 0;
		virtual float getValue(double eastings, double northings) throw(SPDProcessingException) = 0;
        virtual void getValues(double *eastings, double *northings, float *zVals, unsigned long nPts) throw(SPDProcessingException)
        {
            for(unsigned long i = 0; i < nPts; ++i)
            {
                zVals[i] = this->getValue(eastings[i], northings[i]);
            }
        };
        virtual bool useGetValues(){return false;};
		virtual void resetInterpolator() throw(SPDProcessingException) = 0;
		virtual ~SPDPointInterpolator(){};
	protected:
        std::vector<SPDPoint*>* findPoints(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		std::vector<SPDPoint*>* findPoints(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		std::vector<SPDPoint*>* findPoints(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		std::vector<SPDPoint*>* findPoints(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
        void thinPoints(std::vector<SPDPoint*> *points) throw(SPDProcessingException);
		bool initialised;
        boost::uint_fast16_t elevVal;
        float thinGridRes;
        bool thinData;
        boost::uint_fast16_t selectHighOrLow;
        boost::uint_fast16_t maxNumPtsPerBin;
        boost::uint_fast64_t totalNumPoints;
	};
	
	
	/**
	 * SPDTriangulationPointInterpolation is an abstract interface for the interpolation 
	 * of SPDPoint/SPDPulse data using a triangulation.
	 * 
	 * The 'init' functions initialise a triangulation (Using CGAL) and the desctructor 
	 * cleans up the triangulation.
	 *
	 */
	class DllExport SPDTriangulationPointInterpolator : public SPDPointInterpolator
	{
	public:
		SPDTriangulationPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		virtual void initInterpolator(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual void initInterpolator(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual void initInterpolator(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual void initInterpolator(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual float getValue(double eastings, double northings) throw(SPDProcessingException) = 0;
		virtual void resetInterpolator() throw(SPDProcessingException);
		virtual ~SPDTriangulationPointInterpolator();
	protected:
        DelaunayTriangulation *dt;
        PointValueMap *values;
        Face_handle fh;

	};
    
    /**
     * SPDTriangulationNNPointInterpolator is an abstract interface for the interpolation
     * of SPDPoint/SPDPulse data using a triangulation.
     *
     * Uses the NN code for the interpolation.
     *
     */
    //  No implementation yet???
    /*class DllExport SPDTriangulationNNPointInterpolator : public SPDPointInterpolator
    {
    public:
        SPDTriangulationNNPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
        virtual void initInterpolator(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
        virtual void initInterpolator(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
        virtual void initInterpolator(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
        virtual void initInterpolator(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
        virtual float getValue(double eastings, double northings) throw(SPDProcessingException);
        virtual void getValues(double *eastings, double *northings, float *zVals, unsigned long nPts) throw(SPDProcessingException) = 0;
        virtual void resetInterpolator() throw(SPDProcessingException);
        virtual bool useGetValues(){return true;};
        virtual ~SPDTriangulationNNPointInterpolator();
    protected:
        double *xPts;
        double *yPts;
        double *zPts;
        unsigned long nPts;
    };
    */
    
	
	/**
	 * SPDGridIndexPointInterpolator is an abstract interface for the interpolation 
	 * of SPDPoint/SPDPulse data using a spatial index (index is in the form of a 
	 * regular grid).
	 * 
	 * The 'init' functions initialise the index and the desctructor 
	 * cleans up the index.
	 *
	 */
	class DllExport SPDGridIndexPointInterpolator : public SPDPointInterpolator
	{
	public:
		SPDGridIndexPointInterpolator(double gridResolution, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		virtual void initInterpolator(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual void initInterpolator(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual void initInterpolator(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual void initInterpolator(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual float getValue(double eastings, double northings) throw(SPDProcessingException) = 0;
		virtual void resetInterpolator() throw(SPDProcessingException);
		virtual ~SPDGridIndexPointInterpolator();
	protected:
		SPDPointGridIndex *idx;
		double gridResolution;
	};
    
    
    /**
	 * SPDRFBPointInterpolator is an abstract interface for the interpolation
	 * of SPDPoint/SPDPulse data using a spatial index (index is in the form of a
	 * regular grid).
	 *
	 * The 'init' functions initialise the index and the desctructor
	 * cleans up the index.
	 *
	 */
	class DllExport SPDRFBPointInterpolator : public SPDPointInterpolator
	{
	public:
		SPDRFBPointInterpolator(double radius, boost::uint_fast16_t numLayers, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		virtual void initInterpolator(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual void initInterpolator(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual void initInterpolator(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual void initInterpolator(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual float getValue(double eastings, double northings) throw(SPDProcessingException);
		virtual void resetInterpolator() throw(SPDProcessingException);
		virtual ~SPDRFBPointInterpolator();
    protected:
        double radius;
        boost::uint_fast16_t numLayers;
        spd_alglib::rbfmodel *rbfModel;
        double midX;
        double midY;
        double minZ;
	};
	
	
	/**
	 *
	 * An implementation of a nearest neighbour interpolator which uses a triangulation
	 * to define the nearest neighbor. 
	 *
	 */
	class DllExport SPDNearestNeighbourInterpolator : public SPDTriangulationPointInterpolator
	{
	public:
		SPDNearestNeighbourInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		float getValue(double eastings, double northings) throw(SPDProcessingException);
		~SPDNearestNeighbourInterpolator();		
	};
	
	/**
	 *
	 * An implementation of a TIN plane fitting interpolator which uses a triangulation
	 * to define the three neighboring points. 
	 *
	 */
	class DllExport SPDTINPlaneFitInterpolator : public SPDTriangulationPointInterpolator
	{
	public:
		SPDTINPlaneFitInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		float getValue(double eastings, double northings) throw(SPDProcessingException);
		~SPDTINPlaneFitInterpolator();
	private:
        SPDMathsUtils *mathUtils;
        SPD3DDataPt *aPt;
        SPD3DDataPt *bPt;
        SPD3DDataPt *cPt;
	};
	
	/**
	 *
	 * An implementation of a Standard Deviation filtering interpolator
	 * from Lee and Lucas 2007.
	 */
	class DllExport SPDStdDevFilterInterpolator : public SPDGridIndexPointInterpolator
	{
	public:
		SPDStdDevFilterInterpolator(float stdDevThreshold, float lowDist, float highDist, float stdDevDist, double gridResolution, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		float getValue(double eastings, double northings) throw(SPDProcessingException);
		~SPDStdDevFilterInterpolator();
	private:
		float stdDevThreshold;
		float lowDist;
		float highDist;
		float stdDevDist;
	};
	
	class DllExport SPDTPSRadiusInterpolator : public SPDGridIndexPointInterpolator
	{
	public:
		SPDTPSRadiusInterpolator(float radius, boost::uint_fast16_t minNumPoints, double gridResolution, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		float getValue(double eastings, double northings) throw(SPDProcessingException);
		~SPDTPSRadiusInterpolator();
	private:
		float radius;
        boost::uint_fast16_t minNumPoints;
	};
    
    class DllExport SPDTPSNumPtsInterpolator : public SPDGridIndexPointInterpolator
	{
	public:
		SPDTPSNumPtsInterpolator(float radius, boost::uint_fast16_t numPoints, double gridResolution, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		float getValue(double eastings, double northings) throw(SPDProcessingException);
		~SPDTPSNumPtsInterpolator();
	private:
        float radius;
        boost::uint_fast16_t numPoints;
	};
    
	class DllExport SPDNaturalNeighborCGALPointInterpolator :public SPDTriangulationPointInterpolator
	{
	public:
		SPDNaturalNeighborCGALPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		float getValue(double eastings, double northings) throw(SPDProcessingException);
		~SPDNaturalNeighborCGALPointInterpolator();
	};
    
    
    // not currently implemented?
    /*class DllExport SPDNaturalNeighborNNPointInterpolator :public SPDTriangulationNNPointInterpolator
    {
    public:
        SPDNaturalNeighborNNPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
        void getValues(double *eastings, double *northings, float *zVals, unsigned long nPts) throw(SPDProcessingException);
        ~SPDNaturalNeighborNNPointInterpolator();
    };*/
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    /**
	 * SPDSphericalPointInterpolator is an abstract interface for the interpolation of 
	 * SPDPoint/SPDPulse data projected spherically.
	 *
	 */
	class DllExport SPDSphericalPointInterpolator
	{
	public:
		SPDSphericalPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		virtual void initInterpolator(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException) = 0;
		virtual void initInterpolator(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException) = 0;
		virtual void initInterpolator(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException) = 0;
		virtual void initInterpolator(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException) = 0;
		virtual float getValue(double azimuth, double zenith) throw(SPDProcessingException) = 0;
		virtual void resetInterpolator() throw(SPDProcessingException) = 0;
		virtual ~SPDSphericalPointInterpolator(){};
	protected:
        std::vector<SPDPoint*>* findPoints(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		std::vector<SPDPoint*>* findPoints(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		std::vector<SPDPoint*>* findPoints(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		std::vector<SPDPoint*>* findPoints(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
        void thinPoints(std::vector<SPDPoint*> *points) throw(SPDProcessingException);
		bool initialised;
        boost::uint_fast16_t elevVal;
        float thinGridRes;
        bool thinData;
        boost::uint_fast16_t selectHighOrLow;
        boost::uint_fast16_t maxNumPtsPerBin;
        boost::uint_fast64_t totalNumPoints;
	};
    
    /**
	 * SPDTriangulationSphericalPointInterpolator is an abstract interface for the interpolation 
	 * of SPDPoint/SPDPulse data using a triangulation.
	 * 
	 * The 'init' functions initialise a triangulation (Using CGAL) and the desctructor 
	 * cleans up the triangulation.
	 *
	 */
	class DllExport SPDTriangulationSphericalPointInterpolator : public SPDSphericalPointInterpolator
	{
	public:
		SPDTriangulationSphericalPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		virtual void initInterpolator(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual void initInterpolator(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual void initInterpolator(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual void initInterpolator(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual float getValue(double azimuth, double zenith) throw(SPDProcessingException) = 0;
		virtual void resetInterpolator() throw(SPDProcessingException);
		virtual ~SPDTriangulationSphericalPointInterpolator();
	protected:
        DelaunayTriangulation *dt;
        PointValueMap *values;
        std::vector<SPDPoint*> *points;
        bool returnNaNValue;
	};
    
    /**
	 * SPDGridIndexSphericalPointInterpolator is an abstract interface for the interpolation 
	 * of SPDPoint/SPDPulse data using a spatial index (index is in the form of a 
	 * regular grid).
	 * 
	 * The 'init' functions initialise the index and the desctructor 
	 * cleans up the index.
	 *
	 */
	class DllExport SPDGridIndexSphericalPointInterpolator : public SPDSphericalPointInterpolator
	{
	public:
		SPDGridIndexSphericalPointInterpolator(double gridResolution, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		virtual void initInterpolator(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual void initInterpolator(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual void initInterpolator(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual void initInterpolator(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException);
		virtual float getValue(double eastings, double northings) throw(SPDProcessingException) = 0;
		virtual void resetInterpolator() throw(SPDProcessingException);
		virtual ~SPDGridIndexSphericalPointInterpolator();
	protected:
		SPDPointGridIndex *idx;
		double gridResolution;
        std::vector<SPDPoint*> *points;
	};

    
    class DllExport SPDNaturalNeighborSphericalPointInterpolator :public SPDTriangulationSphericalPointInterpolator
	{
	public:
		SPDNaturalNeighborSphericalPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		float getValue(double azimuth, double zenith) throw(SPDProcessingException);
		~SPDNaturalNeighborSphericalPointInterpolator();
	};
    
    class DllExport SPDNearestNeighborSphericalPointInterpolator :public SPDTriangulationSphericalPointInterpolator
	{
	public:
		SPDNearestNeighborSphericalPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin, float distanceThreshold);
		float getValue(double azimuth, double zenith) throw(SPDProcessingException);
		~SPDNearestNeighborSphericalPointInterpolator();
    protected:
        float distanceThreshold;
	};
    
    class DllExport SPDTPSRadiusSphericalInterpolator : public SPDGridIndexSphericalPointInterpolator
	{
	public:
		SPDTPSRadiusSphericalInterpolator(float radius, boost::uint_fast16_t minNumPoints, double gridResolution, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		float getValue(double azimuth, double zenith) throw(SPDProcessingException);
		~SPDTPSRadiusSphericalInterpolator();
	private:
		float radius;
        boost::uint_fast16_t minNumPoints;
	};
    
    class DllExport SPDTPSNumPtsSphericalInterpolator : public SPDGridIndexSphericalPointInterpolator
	{
	public:
		SPDTPSNumPtsSphericalInterpolator(float radius, boost::uint_fast16_t numPoints, double gridResolution, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin);
		float getValue(double azimuth, double zenith) throw(SPDProcessingException);
		~SPDTPSNumPtsSphericalInterpolator();
	private:
        float radius;
        boost::uint_fast16_t numPoints;
	};

    
}

#endif



