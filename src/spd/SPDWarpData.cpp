/*
 *  SPDWarpData.cpp
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

#include "spd/SPDWarpData.h"

namespace spdlib
{

    SPDShiftData::SPDShiftData(float xShift, float yShift)
    {
        this->xShift = xShift;
        this->yShift = yShift;
    }

    void SPDShiftData::processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException)
    {
        std::vector<SPDPulse*>::iterator iterPulses;
        std::vector<SPDPoint*>::iterator iterPoints;

        for(boost::uint_fast32_t i = 0; i < ySize; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < xSize; ++j)
            {
                if(pulses[i][j]->size() > 0)
                {
                    for(iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                    {
                        (*iterPulses)->x0 += xShift;
                        (*iterPulses)->y0 += yShift;
                        (*iterPulses)->xIdx += xShift;
                        (*iterPulses)->yIdx += yShift;
                        if((*iterPulses)->numberOfReturns > 0)
                        {
                            for(iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
                            {
                                (*iterPoints)->x += xShift;
                                (*iterPoints)->y += yShift;
                            }
                        }
                    }
                }
            }
        }
    }

    SPDShiftData::~SPDShiftData()
    {

    }



    SPDGCPImg2MapNode::SPDGCPImg2MapNode(double eastings, double northings, float xOff, float yOff): eastings_(0), northings_(0), xOff_(0), yOff_(0)
	{
        this->eastings_ = eastings;
        this->northings_ = northings;
		this->xOff_ = xOff;
		this->yOff_ = yOff;
	}
	
	double SPDGCPImg2MapNode::eastings() const
	{
		return eastings_;
	}
	
	double SPDGCPImg2MapNode::northings() const
	{
		return northings_;
	}
	
	float SPDGCPImg2MapNode::xOff() const
	{
		return xOff_;
	}
	
	float SPDGCPImg2MapNode::yOff() const
	{
		return yOff_;
	}
	
	double SPDGCPImg2MapNode::distanceGeo(SPDGCPImg2MapNode *pt)
	{
		double sqSum = ((this->eastings_ - pt->eastings_)*(this->eastings_ - pt->eastings_)) + ((this->northings_ - pt->northings_)*(this->northings_ - pt->northings_));
		
		return sqrt(sqSum/2);
	}
	
	SPDGCPImg2MapNode::~SPDGCPImg2MapNode()
	{
		
	}





    SPDWarpPointData::SPDWarpPointData()
    {
        gcps = new std::vector<SPDGCPImg2MapNode*>();
    }


    void SPDWarpPointData::readGCPs(std::string gcpFile) throw(SPDException)
    {
        try
        {
            spdlib::SPDTextFileUtilities textUtils;
            spdlib::SPDTextFileLineReader lineReader;

            lineReader.openFile(gcpFile);
            std::vector<std::string> *tokens = new std::vector<std::string>();
            std::string strLine;
            SPDGCPImg2MapNode *gcp = NULL;
            while(!lineReader.endOfFile())
            {
                strLine = lineReader.readLine();
                if((!textUtils.lineStart(strLine, '#')) & (!textUtils.blankline(strLine)))
                {
                    textUtils.tokenizeString(strLine, ',', tokens, true);

                    if(tokens->size() != 4)
                    {
                        delete tokens;
                        lineReader.closeFile();
                        std::string message = "Line should have 4 tokens: \"" + strLine + "\"";
                        throw SPDException(message);
                    }


                    gcp = new SPDGCPImg2MapNode(textUtils.strtodouble(tokens->at(0)),
                                                textUtils.strtodouble(tokens->at(1)),
                                                textUtils.strtofloat(tokens->at(2)),
                                                textUtils.strtofloat(tokens->at(3)));

                    gcps->push_back(gcp);

                    tokens->clear();
                }
            }

            lineReader.closeFile();

            delete tokens;
        }
        catch(SPDException &e)
        {
            throw e;
        }
        catch(std::exception &e)
        {
            throw SPDException(e.what());
        }
    }

    void SPDWarpPointData::calcOffset(float eastings, float northings, float *xOff, float *yOff)throw(SPDWarpException)
    {
        try
        {
            *xOff = this->calcXOffset(eastings, northings);
            *yOff = this->calcYOffset(eastings, northings);
        }
        catch(SPDWarpException &e)
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDWarpException(e.what());
        }
        catch(std::exception &e)
        {
            throw SPDWarpException(e.what());
        }
    }


    SPDWarpPointData::~SPDWarpPointData()
    {
        delete gcps;
    }



    SPDNearestNeighbourWarp::SPDNearestNeighbourWarp() : SPDWarpPointData()
    {

    }

    bool SPDNearestNeighbourWarp::initWarp(std::string gcpFile)throw(SPDWarpException)
    {
        try
        {
            this->readGCPs(gcpFile);

            dt = new DelaunayTriangulation();
            values = new PointValueMap();

            std::vector<SPDGCPImg2MapNode*>::iterator iterGCPs;
            for(iterGCPs = gcps->begin(); iterGCPs != gcps->end(); ++iterGCPs)
            {
                K::Point_2 cgalPt((*iterGCPs)->eastings(),(*iterGCPs)->northings());
                dt->insert(cgalPt);

                values->insert(std::make_pair(cgalPt, (*iterGCPs)));
            }
        }
        catch(SPDWarpException &e)
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDWarpException(e.what());
        }
        catch(std::exception &e)
        {
            throw SPDWarpException(e.what());
        }

        return true;
    }

    float SPDNearestNeighbourWarp::calcXOffset(float eastings, float northings)throw(SPDWarpException)
    {
        CGALPoint p(eastings, northings);
        Vertex_handle vh = dt->nearest_vertex(p);
        CGALPoint nearestPt = vh->point();
        PointValueMap::iterator iterVal;
        Face_handle fh = dt->locate(nearestPt);
        Vertex_handle pt1Vh = fh->vertex(0);
        iterVal = values->find(pt1Vh->point());
        SPDGCPImg2MapNode *nodeA = (*iterVal).second;

        return nodeA->xOff();
    }

    float SPDNearestNeighbourWarp::calcYOffset(float eastings, float northings)throw(SPDWarpException)
    {
        CGALPoint p(eastings, northings);
        Vertex_handle vh = dt->nearest_vertex(p);
        CGALPoint nearestPt = vh->point();
        PointValueMap::iterator iterVal;
        Face_handle fh = dt->locate(nearestPt);
        Vertex_handle pt1Vh = fh->vertex(0);
        iterVal = values->find(pt1Vh->point());
        SPDGCPImg2MapNode *nodeA = (*iterVal).second;

        return nodeA->yOff();
    }

    void SPDNearestNeighbourWarp::calcOffset(float eastings, float northings, float *xOff, float *yOff)throw(SPDWarpException)
    {
        CGALPoint p(eastings, northings);
        Vertex_handle vh = dt->nearest_vertex(p);
        CGALPoint nearestPt = vh->point();
        PointValueMap::iterator iterVal;
        Face_handle fh = dt->locate(nearestPt);
        Vertex_handle pt1Vh = fh->vertex(0);
        iterVal = values->find(pt1Vh->point());
        SPDGCPImg2MapNode *nodeA = (*iterVal).second;

        *xOff = nodeA->xOff();
        *xOff = nodeA->yOff();
    }

    SPDNearestNeighbourWarp::~SPDNearestNeighbourWarp()
    {
        if(dt != NULL)
		{
			delete dt;
		}

        if(values != NULL)
        {
            delete values;
        }
    }




    SPDTriangulationPlaneFittingWarp::SPDTriangulationPlaneFittingWarp() : SPDWarpPointData()
    {

    }

    bool SPDTriangulationPlaneFittingWarp::initWarp(std::string gcpFile)throw(SPDWarpException)
    {
        try
        {
            this->readGCPs(gcpFile);

            dt = new DelaunayTriangulation();
            values = new PointValueMap();

            std::vector<SPDGCPImg2MapNode*>::iterator iterGCPs;
            for(iterGCPs = gcps->begin(); iterGCPs != gcps->end(); ++iterGCPs)
            {
                K::Point_2 cgalPt((*iterGCPs)->eastings(),(*iterGCPs)->northings());
                dt->insert(cgalPt);

                values->insert(std::make_pair(cgalPt, (*iterGCPs)));
            }
        }
        catch(SPDWarpException &e)
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDWarpException(e.what());
        }
        catch(std::exception &e)
        {
            throw SPDWarpException(e.what());
        }

        return true;
    }

    float SPDTriangulationPlaneFittingWarp::calcXOffset(float eastings, float northings)throw(SPDWarpException)
    {
        CGALPoint p(eastings, northings);
        Vertex_handle vh = dt->nearest_vertex(p);
        CGALPoint nearestPt = vh->point();
        PointValueMap::iterator iterVal;
        Face_handle fh = dt->locate(nearestPt);
        Vertex_handle pt1Vh = fh->vertex(0);
        iterVal = values->find(pt1Vh->point());
        SPDGCPImg2MapNode *nodeA = (*iterVal).second;
        Vertex_handle pt2Vh = fh->vertex(1);
        iterVal = values->find(pt2Vh->point());
        SPDGCPImg2MapNode *nodeB = (*iterVal).second;
        Vertex_handle pt3Vh = fh->vertex(2);
        iterVal = values->find(pt3Vh->point());
        SPDGCPImg2MapNode *nodeC = (*iterVal).second;

        std::list<const SPDGCPImg2MapNode*> *triPts = new std::list<const SPDGCPImg2MapNode*>();
		triPts->push_back(nodeA);
		triPts->push_back(nodeB);
		triPts->push_back(nodeC);

        std::list<SPDGCPImg2MapNode*> *normTriPts = normGCPs(triPts, eastings, northings);
		
		double planeA = 0;
		double planeB = 0;
		double planeC = 0;
		
		this->fitPlane2XPoints(normTriPts, &planeA, &planeB, &planeC);
		float xOff = planeC;
		
        std::list<SPDGCPImg2MapNode*>::iterator iterGCPs;
		for(iterGCPs = normTriPts->begin(); iterGCPs != normTriPts->end(); )
		{
			delete *iterGCPs;
			normTriPts->erase(iterGCPs++);
		}
		delete normTriPts;

        return xOff;
    }

    float SPDTriangulationPlaneFittingWarp::calcYOffset(float eastings, float northings)throw(SPDWarpException)
    {
        CGALPoint p(eastings, northings);
        Vertex_handle vh = dt->nearest_vertex(p);
        CGALPoint nearestPt = vh->point();
        PointValueMap::iterator iterVal;
        Face_handle fh = dt->locate(nearestPt);
        Vertex_handle pt1Vh = fh->vertex(0);
        iterVal = values->find(pt1Vh->point());
        SPDGCPImg2MapNode *nodeA = (*iterVal).second;
        Vertex_handle pt2Vh = fh->vertex(1);
        iterVal = values->find(pt2Vh->point());
        SPDGCPImg2MapNode *nodeB = (*iterVal).second;
        Vertex_handle pt3Vh = fh->vertex(2);
        iterVal = values->find(pt3Vh->point());
        SPDGCPImg2MapNode *nodeC = (*iterVal).second;

        std::list<const SPDGCPImg2MapNode*> *triPts = new std::list<const SPDGCPImg2MapNode*>();
		triPts->push_back(nodeA);
		triPts->push_back(nodeB);
		triPts->push_back(nodeC);

        std::list<SPDGCPImg2MapNode*> *normTriPts = normGCPs(triPts, eastings, northings);
		
		double planeA = 0;
		double planeB = 0;
		double planeC = 0;

		this->fitPlane2YPoints(normTriPts, &planeA, &planeB, &planeC);
		float yOff = planeC;
		
        std::list<SPDGCPImg2MapNode*>::iterator iterGCPs;
		for(iterGCPs = normTriPts->begin(); iterGCPs != normTriPts->end(); )
		{
			delete *iterGCPs;
			normTriPts->erase(iterGCPs++);
		}
		delete normTriPts;

        return yOff;
    }

    void SPDTriangulationPlaneFittingWarp::calcOffset(float eastings, float northings, float *xOff, float *yOff)throw(SPDWarpException)
    {
        CGALPoint p(eastings, northings);
        Vertex_handle vh = dt->nearest_vertex(p);
        CGALPoint nearestPt = vh->point();
        PointValueMap::iterator iterVal;
        Face_handle fh = dt->locate(nearestPt);
        Vertex_handle pt1Vh = fh->vertex(0);
        iterVal = values->find(pt1Vh->point());
        SPDGCPImg2MapNode *nodeA = (*iterVal).second;
        Vertex_handle pt2Vh = fh->vertex(1);
        iterVal = values->find(pt2Vh->point());
        SPDGCPImg2MapNode *nodeB = (*iterVal).second;
        Vertex_handle pt3Vh = fh->vertex(2);
        iterVal = values->find(pt3Vh->point());
        SPDGCPImg2MapNode *nodeC = (*iterVal).second;

        std::list<const SPDGCPImg2MapNode*> *triPts = new std::list<const SPDGCPImg2MapNode*>();
		triPts->push_back(nodeA);
		triPts->push_back(nodeB);
		triPts->push_back(nodeC);

        std::list<SPDGCPImg2MapNode*> *normTriPts = normGCPs(triPts, eastings, northings);
		
		double planeA = 0;
		double planeB = 0;
		double planeC = 0;
		
		this->fitPlane2XPoints(normTriPts, &planeA, &planeB, &planeC);
		*xOff = planeC;
		this->fitPlane2YPoints(normTriPts, &planeA, &planeB, &planeC);
		*yOff = planeC;
		
        std::list<SPDGCPImg2MapNode*>::iterator iterGCPs;
		for(iterGCPs = normTriPts->begin(); iterGCPs != normTriPts->end(); )
		{
			delete *iterGCPs;
			normTriPts->erase(iterGCPs++);
		}
		delete normTriPts;
    }

    std::list<SPDGCPImg2MapNode*>* SPDTriangulationPlaneFittingWarp::normGCPs(std::list<const SPDGCPImg2MapNode*> *gcps, double eastings, double northings)
	{
        std::list<SPDGCPImg2MapNode*> *normTriPts = new std::list<SPDGCPImg2MapNode*>();
		
		SPDGCPImg2MapNode *tmpGCP = NULL;
		
        std::list<const SPDGCPImg2MapNode*>::iterator iterGCPs;
		for(iterGCPs = gcps->begin(); iterGCPs != gcps->end(); ++iterGCPs)
		{
			tmpGCP = new SPDGCPImg2MapNode(((*iterGCPs)->eastings() - eastings),
											 ((*iterGCPs)->northings() - northings),
											 (*iterGCPs)->xOff(),
											 (*iterGCPs)->yOff());
			normTriPts->push_back(tmpGCP);
		}
		
		return normTriPts;
	}
	
	void SPDTriangulationPlaneFittingWarp::fitPlane2XPoints(std::list<SPDGCPImg2MapNode*> *normPts, double *a, double *b, double *c) throw(SPDWarpException)
	{
		SPDMatrixUtils matrices;
		
		try
		{
			double sXY = 0;
			double sX = 0;
			double sXSqu = 0;
			double sY = 0;
			double sYSqu = 0;
			double sXZ = 0;
			double sYZ = 0;
			double sZ = 0;
			
            std::list<SPDGCPImg2MapNode*>::iterator iterPts;
			
			for(iterPts = normPts->begin(); iterPts != normPts->end(); ++iterPts)
			{
				sXY += ((*iterPts)->eastings() * (*iterPts)->northings());
				sX += (*iterPts)->eastings();
				sXSqu += ((*iterPts)->eastings() * (*iterPts)->eastings());
				sY += (*iterPts)->northings();
				sYSqu += ((*iterPts)->northings() * (*iterPts)->northings());
				sXZ += ((*iterPts)->eastings() * (*iterPts)->xOff());
				sYZ += ((*iterPts)->northings() * (*iterPts)->xOff());
				sZ += (*iterPts)->xOff();
			}
			
			Matrix *matrixA = matrices.createMatrix(3, 3);
			matrixA->matrix[0] = sXSqu;
			matrixA->matrix[1] = sXY;
			matrixA->matrix[2] = sX;
			matrixA->matrix[3] = sXY;
			matrixA->matrix[4] = sYSqu;
			matrixA->matrix[5] = sY;
			matrixA->matrix[6] = sX;
			matrixA->matrix[7] = sY;
			matrixA->matrix[8] = normPts->size();
			Matrix *matrixB = matrices.createMatrix(1, 3);
			matrixB->matrix[0] = sXZ;
			matrixB->matrix[1] = sYZ;
			matrixB->matrix[2] = sZ;
			
			double determinantA = matrices.determinant(matrixA);
			Matrix *matrixCoFactors = matrices.cofactors(matrixA);
			Matrix *matrixCoFactorsT = matrices.transpose(matrixCoFactors);
			double multiplier = 1/determinantA;
			matrices.multipleSingle(matrixCoFactorsT, multiplier);
			Matrix *outputs = matrices.multiplication(matrixCoFactorsT, matrixB);
			*a = outputs->matrix[0];
			*b = outputs->matrix[1];
			*c = outputs->matrix[2];
			
			matrices.freeMatrix(matrixA);
			matrices.freeMatrix(matrixB);
			matrices.freeMatrix(matrixCoFactors);
			matrices.freeMatrix(matrixCoFactorsT);
			matrices.freeMatrix(outputs);
		}
		catch(SPDException e)
		{
			throw SPDWarpException(e.what());
		}
	}
	
	void SPDTriangulationPlaneFittingWarp::fitPlane2YPoints(std::list<SPDGCPImg2MapNode*> *normPts, double *a, double *b, double *c) throw(SPDWarpException)
	{
		SPDMatrixUtils matrices;
		
		try
		{
			double sXY = 0;
			double sX = 0;
			double sXSqu = 0;
			double sY = 0;
			double sYSqu = 0;
			double sXZ = 0;
			double sYZ = 0;
			double sZ = 0;
			
            std::list<SPDGCPImg2MapNode*>::iterator iterPts;
			
			for(iterPts = normPts->begin(); iterPts != normPts->end(); ++iterPts)
			{
				sXY += ((*iterPts)->eastings() * (*iterPts)->northings());
				sX += (*iterPts)->eastings();
				sXSqu += ((*iterPts)->eastings() * (*iterPts)->eastings());
				sY += (*iterPts)->northings();
				sYSqu += ((*iterPts)->northings() * (*iterPts)->northings());
				sXZ += ((*iterPts)->eastings() * (*iterPts)->yOff());
				sYZ += ((*iterPts)->northings() * (*iterPts)->yOff());
				sZ += (*iterPts)->yOff();
			}
			
			Matrix *matrixA = matrices.createMatrix(3, 3);
			matrixA->matrix[0] = sXSqu;
			matrixA->matrix[1] = sXY;
			matrixA->matrix[2] = sX;
			matrixA->matrix[3] = sXY;
			matrixA->matrix[4] = sYSqu;
			matrixA->matrix[5] = sY;
			matrixA->matrix[6] = sX;
			matrixA->matrix[7] = sY;
			matrixA->matrix[8] = normPts->size();
			Matrix *matrixB = matrices.createMatrix(1, 3);
			matrixB->matrix[0] = sXZ;
			matrixB->matrix[1] = sYZ;
			matrixB->matrix[2] = sZ;
			
			double determinantA = matrices.determinant(matrixA);
			Matrix *matrixCoFactors = matrices.cofactors(matrixA);
			Matrix *matrixCoFactorsT = matrices.transpose(matrixCoFactors);
			double multiplier = 1/determinantA;
			matrices.multipleSingle(matrixCoFactorsT, multiplier);
			Matrix *outputs = matrices.multiplication(matrixCoFactorsT, matrixB);
			*a = outputs->matrix[0];
			*b = outputs->matrix[1];
			*c = outputs->matrix[2];
			
			matrices.freeMatrix(matrixA);
			matrices.freeMatrix(matrixB);
			matrices.freeMatrix(matrixCoFactors);
			matrices.freeMatrix(matrixCoFactorsT);
			matrices.freeMatrix(outputs);
		}
		catch(SPDException e)
		{
			throw SPDWarpException(e.what());
		}
	}

    SPDTriangulationPlaneFittingWarp::~SPDTriangulationPlaneFittingWarp()
    {
        if(dt != NULL)
		{
			delete dt;
		}

        if(values != NULL)
        {
            delete values;
        }
    }





    SPDPolynomialWarp::SPDPolynomialWarp(int order) : SPDWarpPointData()
    {
        this->polyOrder = order;
    }

    bool SPDPolynomialWarp::initWarp(std::string gcpFile)throw(SPDWarpException)
    {
        try
        {
            this->readGCPs(gcpFile);

            /** Initialises warp by create polynominal models based on ground countrol points.
             Models are created expressing image pixels as a function of easting and northing, used for warping
             and expressing easting and northing as a function of image pixels to determine the corner location of the
             image to be warped.
             */


            std::cout << "Fitting polynomial..." << std::endl;

            unsigned int coeffSize = 3 * this->polyOrder; // x**N + y**N + xy**(N-1) + 1

            if(gcps->size() < coeffSize)
            {
                std::cout << "gcps->size() = " << gcps->size() << std::endl;
                std::cout << "coeffSize = " << coeffSize << std::endl;
                throw SPDWarpException("Too few gcp's have been provided you, either need to decrease the order of the polynomial or increase the number of GCPs.");
            }

            // Set up matrices
            gsl_matrix *eastNorthPow = gsl_matrix_alloc(gcps->size(), coeffSize); // Matrix to hold powers of easting and northing (used for both fits)
            gsl_matrix *xyPow = gsl_matrix_alloc(gcps->size(), coeffSize); // Matrix to hold powers of x and y (used for both fits)
            gsl_vector *pixValX = gsl_vector_alloc(gcps->size()); // Vector to hold pixel values (X)
            gsl_vector *pixValY = gsl_vector_alloc(gcps->size()); // Vector to hold pixel values (Y)
            gsl_vector *eastingVal = gsl_vector_alloc(gcps->size()); // Vector to hold easting values
            gsl_vector *northingVal = gsl_vector_alloc(gcps->size()); // Vector to hold northing values
            this->aX = gsl_vector_alloc(coeffSize); // Vector to hold coeffifients of X (Easting)
            this->aY = gsl_vector_alloc(coeffSize); // Vector to hold coeffifients of Y (Northing)
            this->aE = gsl_vector_alloc(coeffSize); // Vector to hold coeffifients of Easting (X)
            this->aN = gsl_vector_alloc(coeffSize); // Vector to hold coeffifients of Northing (Y)

            unsigned int pointN = 0;
            unsigned int offset = 0;

            std::vector<SPDGCPImg2MapNode*>::iterator iterGCPs;
            for(iterGCPs = gcps->begin(); iterGCPs != gcps->end(); ++iterGCPs) // Populate matrices using ground control points.
            {
                // Add values into vectors
                gsl_vector_set(pixValX, pointN, (*iterGCPs)->xOff()); // Offset X
                gsl_vector_set(pixValY, pointN, (*iterGCPs)->yOff()); // Offset Y
                gsl_vector_set(eastingVal, pointN, (*iterGCPs)->eastings()); // Easting
                gsl_vector_set(northingVal, pointN, (*iterGCPs)->northings()); // Northing

                gsl_matrix_set(eastNorthPow, pointN, 0, 1.);
                gsl_matrix_set(xyPow, pointN, 0, 1.);

                for(int j = 1; j < polyOrder; ++j)
                {
                    offset = 1 + (3 * (j - 1));
                    gsl_matrix_set(eastNorthPow, pointN, offset, pow((*iterGCPs)->eastings(), j));
                    gsl_matrix_set(eastNorthPow, pointN, offset+1, pow((*iterGCPs)->northings(), j));
                    gsl_matrix_set(eastNorthPow, pointN, offset+2, pow((*iterGCPs)->eastings()*(*iterGCPs)->northings(), j));

                    gsl_matrix_set(xyPow, pointN, offset, pow((*iterGCPs)->xOff(), j));
                    gsl_matrix_set(xyPow, pointN, offset+1, pow((*iterGCPs)->xOff(), j));
                    gsl_matrix_set(xyPow, pointN, offset+2, pow((*iterGCPs)->xOff()*(*iterGCPs)->yOff(), j));
                }

                offset = 1 + (3 * (this->polyOrder - 1));
                gsl_matrix_set(eastNorthPow, pointN, offset, pow((*iterGCPs)->eastings(), this->polyOrder));
                gsl_matrix_set(eastNorthPow, pointN, offset+1, pow((*iterGCPs)->northings(), this->polyOrder));

                gsl_matrix_set(xyPow, pointN, offset, pow((*iterGCPs)->xOff(), this->polyOrder));
                gsl_matrix_set(xyPow, pointN, offset+1, pow((*iterGCPs)->yOff(), this->polyOrder));

                ++pointN;
            }

            // Set up worksapce for fitting
            gsl_multifit_linear_workspace *workspace;
            workspace = gsl_multifit_linear_alloc(gcps->size(), coeffSize);
            gsl_matrix *cov = gsl_matrix_alloc(coeffSize, coeffSize);

            double chisq = 0;

            // Fit for X
            gsl_multifit_linear(eastNorthPow, pixValX, this->aX, cov, &chisq, workspace);
            // Fit for Y
            gsl_multifit_linear(eastNorthPow, pixValY, this->aY, cov, &chisq, workspace);
            // Fit for E
            gsl_multifit_linear(xyPow, eastingVal, this->aE, cov, &chisq, workspace);
            // Fit for N
            gsl_multifit_linear(xyPow, northingVal, this->aN, cov, &chisq, workspace);

            std::cout << "Fitted polynomial." << std::endl;

            // Test polynominal fit and calculate RMSE
            double sqSum = 0;

            for(iterGCPs = gcps->begin(); iterGCPs != gcps->end(); ++iterGCPs) // Populate matrices using ground control points.
            {
                double pX = 0;
                double pY = 0;

                // Add pixel values into vectors
                pX = pX + gsl_vector_get(this->aX, 0);
                pY = pY + gsl_vector_get(this->aY, 0);

                for(int j = 1; j < this->polyOrder; ++j)
                {
                    offset = 1 + (3 * (j - 1));

                    pX = pX + (gsl_vector_get(aX, offset) * pow((*iterGCPs)->eastings(), j));
                    pX = pX + (gsl_vector_get(aX, offset+1) * pow((*iterGCPs)->northings(), j));
                    pX = pX + (gsl_vector_get(aX, offset+2) * pow((*iterGCPs)->eastings()*(*iterGCPs)->northings(), j));

                    pY = pY + (gsl_vector_get(aY, offset) * pow((*iterGCPs)->eastings(), j));
                    pY = pY + (gsl_vector_get(aY, offset+1) * pow((*iterGCPs)->northings(), j));
                    pY = pY + (gsl_vector_get(aY, offset+2) * pow((*iterGCPs)->eastings()*(*iterGCPs)->northings(), j));
                }

                offset = 1 + (3 * (this->polyOrder - 1));
                pX = pX + (gsl_vector_get(aX, offset) * pow((*iterGCPs)->eastings(), this->polyOrder));
                pX = pX + (gsl_vector_get(aX, offset+1) * pow((*iterGCPs)->northings(), this->polyOrder));

                pY = pY + (gsl_vector_get(aY, offset) * pow((*iterGCPs)->eastings(), this->polyOrder));
                pY = pY + (gsl_vector_get(aY, offset+1) * pow((*iterGCPs)->northings(), this->polyOrder));

                sqSum = sqSum + (pow((*iterGCPs)->xOff() - pX ,2) + pow((*iterGCPs)->yOff() - pY ,2));

                std::cout << "[" << (*iterGCPs)->eastings() << "," << (*iterGCPs)->northings() << "]: " << (*iterGCPs)->xOff() << "= " << pX << ", " << (*iterGCPs)->yOff() << "= " << pY << std::endl;
            }

            double sqMean = sqSum / double(gcps->size());

            double rmse = sqrt(sqMean);

            std::cout << "RMSE = " << rmse << " metres " << std::endl;

            // Tidy up
            gsl_multifit_linear_free(workspace);
            gsl_matrix_free(eastNorthPow);
            gsl_matrix_free(xyPow);
            gsl_vector_free(pixValX);
            gsl_vector_free(pixValY);
            gsl_vector_free(eastingVal);
            gsl_vector_free(northingVal);
            gsl_matrix_free(cov);
        }
        catch(SPDWarpException &e)
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDWarpException(e.what());
        }
        catch(std::exception &e)
        {
            throw SPDWarpException(e.what());
        }

        return true;
    }

    float SPDPolynomialWarp::calcXOffset(float eastings, float northings)throw(SPDWarpException)
    {
        /* Return nearest pixel based on input easting and northing.
         Pixel x coordinate are found from polynominal model */
        double pX = 0;
        unsigned int offset = 0;

        // Add pixel values into vectors
        pX = pX + gsl_vector_get(this->aX, 0);

        for(int j = 1; j < this->polyOrder; ++j)
        {
            offset = 1 + (3 * (j - 1));

            pX = pX + (gsl_vector_get(aX, offset) * pow(eastings, j));
            pX = pX + (gsl_vector_get(aX, offset+1) * pow(northings, j));
            pX = pX + (gsl_vector_get(aX, offset+2) * pow(eastings*northings, j));
        }

        offset = 1 + (3 * (this->polyOrder - 1));
        pX = pX + (gsl_vector_get(aX, offset) * pow(eastings, this->polyOrder));
        pX = pX + (gsl_vector_get(aX, offset+1) * pow(northings, this->polyOrder));

		return pX;
    }

    float SPDPolynomialWarp::calcYOffset(float eastings, float northings)throw(SPDWarpException)
    {
        /* Return nearest pixel based on input easting and northing.
         Pixel y coordinate are found from polynominal model */
        double pY = 0;
        unsigned int offset = 0;

        // Add pixel values into vectors
        pY = pY + gsl_vector_get(this->aY, 0);

        for(int j = 1; j < this->polyOrder; ++j)
        {
            offset = 1 + (3 * (j - 1));

            pY = pY + (gsl_vector_get(aY, offset) * pow(eastings, j));
            pY = pY + (gsl_vector_get(aY, offset+1) * pow(northings, j));
            pY = pY + (gsl_vector_get(aY, offset+2) * pow(eastings*northings, j));
        }

        offset = 1 + (3 * (this->polyOrder - 1));

        pY = pY + (gsl_vector_get(aY, offset) * pow(eastings, this->polyOrder));
        pY = pY + (gsl_vector_get(aY, offset+1) * pow(northings, this->polyOrder));

		return pY;
    }

    void SPDPolynomialWarp::calcOffset(float eastings, float northings, float *xOff, float *yOff)throw(SPDWarpException)
    {
        std::cout << "Calculating Offset for [" << eastings << "," << northings << "]\n";

        /* Return nearest pixel based on input easting and northing.
         Pixel x and y coordinates are found from polynominal model */

        double pX = 0;
        double pY = 0;
        unsigned int offset = 0;

        // Add values into vectors
        pX = pX + gsl_vector_get(this->aX, 0);
        pY = pY + gsl_vector_get(this->aY, 0);

        for(int j = 1; j < this->polyOrder; ++j)
        {
            offset = 1 + (3 * (j - 1));

            pX = pX + (gsl_vector_get(aX, offset) * pow(eastings, j));
            pX = pX + (gsl_vector_get(aX, offset+1) * pow(northings, j));
            pX = pX + (gsl_vector_get(aX, offset+2) * pow(eastings*northings, j));

            pY = pY + (gsl_vector_get(aY, offset) * pow(eastings, j));
            pY = pY + (gsl_vector_get(aY, offset+1) * pow(northings, j));
            pY = pY + (gsl_vector_get(aY, offset+2) * pow(eastings*northings, j));
        }

        offset = 1 + (3 * (this->polyOrder - 1));
        pX = pX + (gsl_vector_get(aX, offset) * pow(eastings, this->polyOrder));
        pX = pX + (gsl_vector_get(aX, offset+1) * pow(northings, this->polyOrder));

        pY = pY + (gsl_vector_get(aY, offset) * pow(eastings, this->polyOrder));
        pY = pY + (gsl_vector_get(aY, offset+1) * pow(northings, this->polyOrder));

        //sqSum = sqSum + (pow((*iterGCPs)->xOff() - pX ,2) + pow((*iterGCPs)->yOff() - pY ,2));
		*xOff = pX;
		*yOff = pY;

        std::cout << "Offset is [" << pX << "," << pY << "]\n";
    }

    SPDPolynomialWarp::~SPDPolynomialWarp()
    {
        gsl_vector_free(this->aX);
        gsl_vector_free(this->aY);
        gsl_vector_free(this->aE);
        gsl_vector_free(this->aN);
    }







    SPDNonLinearWarp::SPDNonLinearWarp(SPDDataExporter *exporter, SPDFile *spdFileOut, SPDWarpPointData *calcOffsets, SPDWarpLocation warpLoc) throw(SPDException)
    {
        this->exporter = exporter;
        this->spdFileOut = spdFileOut;
        this->calcOffsets = calcOffsets;
        this->warpLoc = warpLoc;

        if(exporter->requireGrid())
        {
            throw SPDException("This class does not support the export of gridded formats.");
        }

        try
        {
            this->exporter->open(this->spdFileOut, this->spdFileOut->getFilePath());
        }
        catch (SPDException &e)
        {
            throw e;
        }
        this->pulses = new std::list<SPDPulse*>();
    }


    void SPDNonLinearWarp::processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException)
    {
        try
        {
            this->pulses->push_back(pulse);

            if(warpLoc == spdwarppulseidx)
            {
                float xOff = 0;
                float yOff = 0;
                calcOffsets->calcOffset(pulse->xIdx, pulse->yIdx, &xOff, &yOff);
                std::cout << "Offset: [" << xOff << "," << yOff << "]\n";

                if(pulse->numberOfReturns > 0)
                {
                    for(std::vector<SPDPoint*>::iterator iterPoints = pulse->pts->begin(); iterPoints != pulse->pts->end(); ++iterPoints)
                    {
                        (*iterPoints)->x += xOff;
                        (*iterPoints)->y += yOff;
                    }
                }

                pulse->x0 += xOff;
                pulse->y0 += yOff;

                pulse->xIdx += xOff;
                pulse->yIdx += yOff;

            }
            else if(warpLoc == spdwarpfromall)
            {
                float xOff = 0;
                float yOff = 0;
                calcOffsets->calcOffset(pulse->xIdx, pulse->yIdx, &xOff, &yOff);
                pulse->xIdx += xOff;
                pulse->yIdx += yOff;

                calcOffsets->calcOffset(pulse->x0, pulse->y0, &xOff, &yOff);
                pulse->x0 += xOff;
                pulse->y0 += yOff;

                if(pulse->numberOfReturns > 0)
                {
                    for(std::vector<SPDPoint*>::iterator iterPoints = pulse->pts->begin(); iterPoints != pulse->pts->end(); ++iterPoints)
                    {
                        calcOffsets->calcOffset((*iterPoints)->x, (*iterPoints)->y, &xOff, &yOff);
                        (*iterPoints)->x += xOff;
                        (*iterPoints)->y += yOff;
                    }
                }

            }
            else if(warpLoc == spdwarppulseorigin)
            {
                float xOff = 0;
                float yOff = 0;
                calcOffsets->calcOffset(pulse->x0, pulse->y0, &xOff, &yOff);

                if(pulse->numberOfReturns > 0)
                {
                    for(std::vector<SPDPoint*>::iterator iterPoints = pulse->pts->begin(); iterPoints != pulse->pts->end(); ++iterPoints)
                    {
                        (*iterPoints)->x += xOff;
                        (*iterPoints)->y += yOff;
                    }
                }

                pulse->x0 += xOff;
                pulse->y0 += yOff;

                pulse->xIdx += xOff;
                pulse->yIdx += yOff;
            }
            else
            {
                throw SPDException("The warp location has not been recognised.");
            }

            this->exporter->writeDataColumn(pulses, 0, 0);
        }
        catch (SPDIOException &e)
        {
            throw e;
        }
        catch (SPDException &e)
        {
            throw SPDIOException(e.what());
        }
    }

    void SPDNonLinearWarp::completeFileAndClose(SPDFile *spdFile)throw(SPDIOException)
    {
        try
        {
            spdFileOut->copyAttributesFrom(spdFile);
            exporter->finaliseClose();
        }
        catch (SPDIOException &e)
        {
            throw e;
        }
    }

    SPDNonLinearWarp::~SPDNonLinearWarp()
    {
        delete pulses;
    }

}