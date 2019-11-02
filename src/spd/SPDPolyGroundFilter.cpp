/*
 *  SPDPolyGroundFilter.cpp
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


#include "spd/SPDPolyGroundFilter.h"


namespace spdlib
{
    

    SPDPolyFitGroundFilter::SPDPolyFitGroundFilter()
    {
        
    }
    
    void SPDPolyFitGroundFilter::applyGlobalPolyFitGroundFilter(std::string inputFile, std::string outputFile, float grdThres, boost::uint_fast16_t degree, boost::uint_fast16_t iters, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize, float processingResolution, boost::uint_fast16_t ptSelectClass)
    {
        try 
        {
            // Read SPD file header.
            SPDFile *spdFile = new SPDFile(inputFile);
            SPDFileReader spdReader;
            spdReader.readHeaderInfo(spdFile->getFilePath(), spdFile);
                        
            // Find the minimum returns.
            std::vector<SPDPoint*> *minPts = new std::vector<SPDPoint*>();
            
            SPDPulseProcessor *processorMinPoints = new SPDFindMinReturnsProcessor(minPts, ptSelectClass);
            SPDSetupProcessPulses processPulses = SPDSetupProcessPulses(blockXSize, blockYSize, true);
            processPulses.processPulses(processorMinPoints, spdFile, processingResolution);
            delete processorMinPoints;
            
            // Create grid to store returns.
            std::vector<SPDPoint*> ***minPtGrid = new std::vector<SPDPoint*>**[spdFile->getNumberBinsY()];
            for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsY(); ++i)
            {
                minPtGrid[i] = new std::vector<SPDPoint*>*[spdFile->getNumberBinsX()];
                for(boost::uint_fast32_t j = 0; j < spdFile->getNumberBinsX(); ++j)
                {
                    minPtGrid[i][j] = new std::vector<SPDPoint*>();
                }
            }
            
            // Grid returns.
            this->buildMinGrid(spdFile, minPts, minPtGrid);
            
            /********** Find Surface ***************/
			
			// calculate number of coefficients
			//unsigned int degree=2; // this will be user defined via command prompt in final version
			//this->degree = degree;
			//this->iters = iters;
			unsigned int Ncoeffs;
			//int l=0;
			//int p=0;
			int ItersCount=0;
			
			
			if (degree > 1) 
			{
                Ncoeffs=((degree+1)*(2*degree))/2;
			}
			else
            {
				Ncoeffs=3; // plane fit (Const+Ax+By) hardcoded because above equation only works for orders of 2 or more
			}
            
			// Set up matrix of powers
			gsl_matrix *indVarPow;
			gsl_vector *depVar;
			gsl_vector *outCoefficients;
			gsl_vector *SurfaceZ;
            
			
			indVarPow = gsl_matrix_alloc(spdFile->getNumberBinsX()*spdFile->getNumberBinsY(),Ncoeffs); // Matrix to hold powers of x,y
			depVar = gsl_vector_alloc(spdFile->getNumberBinsX()*spdFile->getNumberBinsY()); // Vector to hold z term
			outCoefficients = gsl_vector_alloc(Ncoeffs); // Vector to hold output coefficients
            SurfaceZ = gsl_vector_alloc(spdFile->getNumberBinsX()*spdFile->getNumberBinsY());
			
			gsl_multifit_linear_workspace *workspace;
			workspace = gsl_multifit_linear_alloc(spdFile->getNumberBinsX()*spdFile->getNumberBinsY(), Ncoeffs);
			gsl_matrix *cov;
			double chisq;
			cov = gsl_matrix_alloc(Ncoeffs, Ncoeffs);
			
			//gsl_permutation * perm = gsl_permutation_alloc (spdFile->getNumberBinsX()*spdFile->getNumberBinsY());
            
            
			bool keepProcessing = true;
			
			std::cout << "Number of Y bins: " << spdFile->getNumberBinsY() << '\n';
			std::cout << "Number of X bins: " << spdFile->getNumberBinsX() << '\n';
			
			while(keepProcessing)
			{
				boost::uint_fast32_t p=0;
				for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsY(); ++i)
				{
					for(boost::uint_fast32_t j = 0; j < spdFile->getNumberBinsX(); ++j)
					{
						//std::cout << "i,j : " << i << " " << j << '\n';
						//std::cout << "Number of Y bins: " << spdFile->getNumberBinsY() << '\n';
						//std::cout << "Number of X bins: " << spdFile->getNumberBinsX() << '\n';
                        
						
						if(minPtGrid[i][j]->size() > 0)
						{
							//std::cout << "Pt: " << minPtGrid[i][j]->front()->x << ", " << minPtGrid[i][j]->front()->y << ", " << minPtGrid[i][j]->front()->z << '\n';
							// generate arrays for x^ny^m and z
							double xelement=minPtGrid[i][j]->front()->x;
							double yelement=minPtGrid[i][j]->front()->y;
							double zelement=minPtGrid[i][j]->front()->z;
							boost::uint_fast32_t l=0;
							gsl_vector_set(depVar, p, zelement);
							//++p;
							
							//std::cout << "NCoeffs: " << Ncoeffs << " Degree: " << degree << '\n';
							
							for (boost::uint_fast32_t m = 0; m < Ncoeffs ; m++)
							{
								for (boost::uint_fast32_t n=0; n < Ncoeffs ; n++)
								{
									if (n+m <= degree)
									{
										double xelementtPow = pow(xelement, ((int)(m)));
										double yelementtPow = pow(yelement, ((int)(n)));
										gsl_matrix_set(indVarPow,p,l, xelementtPow*yelementtPow); // was n,m instead of l
                                                                                                  //std::cout << "indvarpow: " << indVarPow << " " << xelementtPow << " " << yelementtPow << " " << "n,m: " << n << " " << m << '\n';
                                                                                                  //std::cout << "n,m: " << n << " " << m << " " << l << '\n';
										++l;
									}
								}
							}
							++p;
							
						}
					}
					
				}
				
				
				//std::cout << "Iters count" << ItersCount << " " << keepProcessing << '\n';
				
				
				// Find surface
				//matrix operations to find coefficients
				// K=U.inverse(Utrans.U)
				//Coeffs=K.z  
				
				//LU = gsl_linalg_LU_decomp (gsl_matrix * indVarPow, gsl_permutation * perm, int * signum);
				//gsl_linalg_LU_solve (const gsl_matrix * LU, const gsl_permutation * perm, const gsl_vector * depVar, gsl_vector * outCoefficients);
				
				// Perform Least Squared Fit
				
                
				gsl_multifit_linear(indVarPow, depVar, outCoefficients, cov, &chisq, workspace);
				
				// calculate vector of z values from outCoefficients and indVarPow to compare with original data in next bit...
				//compute SurfaceZ = indVarPow.outCoefficients
				//gsl_blas_dgemv (CblasNoTrans, CblasNoTrans, 1.0, indVarPow, outCoefficients, 0.0, SurfaceZ);
				
				gsl_blas_dgemv (CblasNoTrans, 1.0, indVarPow, outCoefficients, 0.0, SurfaceZ);
				
				// compare polynomial surface with minimum points grid.
				// if there are points above a certain threshold, delete them
				// and then start the interation again to find a better surface fit for the ground.
				// this step is necessary to remove minimum points in branches residing over shadowed areas.
                
				
                for(unsigned int j = 0; j < outCoefficients->size; j++)
                {
                    double outm = gsl_vector_get(outCoefficients, j); 
                    std::cout << outm << " , " ;
                }
                std::cout << '\n';
				
				
				
				
				// remove outlying points that are obviously not ground
				
				for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsY(); ++i)
				{
					for(boost::uint_fast32_t j = 0; j < spdFile->getNumberBinsX(); ++j)
					{
						if(minPtGrid[i][j]->size() > 0)
						{
							
							// to determine if points are above surface:
							double zelement=minPtGrid[i][j]->front()->z;
							double xelement=minPtGrid[i][j]->front()->x;
							double yelement=minPtGrid[i][j]->front()->y;
							double sz=0; // reset z value from surface coefficients
							boost::uint_fast32_t l=0;
							
							for (boost::uint_fast32_t m = 0; m < Ncoeffs ; m++)
							{
								for (boost::uint_fast32_t n=0; n < Ncoeffs ; n++)
								{
									if (n+m <= degree)
									{
										double xelementtPow = pow(xelement, ((int)(m)));
										double yelementtPow = pow(yelement, ((int)(n)));
										double outm = gsl_vector_get(outCoefficients, l);
										
										sz=sz+(outm*xelementtPow*yelementtPow);
										++l;
									}
								}
							}
							
							
							if (zelement-sz > 1.0)
							{
								minPtGrid[i][j]->clear(); // delete point
							}
							
							++l;
                            
                            
						}
					}
					
				}
				
				ItersCount++;
				if (ItersCount==iters)
				{
					keepProcessing = false;
				}
				
				
			}  
            // end of while loop
			
			
            // Clean up	
            gsl_multifit_linear_free(workspace);
			gsl_matrix_free(cov);
            gsl_matrix_free(indVarPow);
            gsl_vector_free(depVar);
            /***************************************/
            
            
            // Remove minimum returns grid and minimum returns from memory
            for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsY(); ++i)
            {
                for(boost::uint_fast32_t j = 0; j < spdFile->getNumberBinsX(); ++j)
                {
                    delete minPtGrid[i][j];
                }
                delete[] minPtGrid[i];
            }
            delete[] minPtGrid;
            
            for(std::vector<SPDPoint*>::iterator iterPts = minPts->begin(); iterPts != minPts->end(); )
            {
                delete *iterPts;
                iterPts = minPts->erase(iterPts);
            }
            delete minPts;
            
            // Classify ground returns using identified surface.
            SPDPulseProcessor *processorClassFromSurface = new SPDClassifyGrdReturnsFromSurfaceCoefficientsProcessor(grdThres, degree, iters, outCoefficients, ptSelectClass);
            processPulses.processPulsesWithOutputSPD(processorClassFromSurface, spdFile, outputFile, processingResolution);
            delete processorClassFromSurface;
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }

    }
    
    void SPDPolyFitGroundFilter::applyLocalPolyFitGroundFilter(std::string inputFile, std::string outputFile, float grdThres,boost::uint_fast16_t degree, boost::uint_fast16_t iters, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize, boost::uint_fast32_t overlap, float processingResolution, boost::uint_fast16_t ptSelectClass)
    {
        try
        {
            SPDFile *spdInFile = new SPDFile(inputFile);
            
            SPDPolyFitGroundLocalFilter *blockProcessor = new SPDPolyFitGroundLocalFilter(grdThres, degree, iters, ptSelectClass, processingResolution);
            SPDProcessDataBlocks processBlocks = SPDProcessDataBlocks(blockProcessor, overlap, blockXSize, blockYSize, true);
            
            processBlocks.processDataBlocksGridPulsesOutputSPD(spdInFile, outputFile, processingResolution);
            
            delete blockProcessor;
            delete spdInFile;
        }
        catch (spdlib::SPDProcessingException &e)
        {
            throw e;
        }
    }
    
    void SPDPolyFitGroundFilter::buildMinGrid(SPDFile *spdFile, std::vector<SPDPoint*> *minPts, std::vector<SPDPoint*> ***minPtGrid)
    {
        if(minPts->size() > 0)
		{
            double binSize = spdFile->getBinSize();
            boost::uint_fast32_t xBins = spdFile->getNumberBinsX();
            boost::uint_fast32_t yBins = spdFile->getNumberBinsY();
            
            if((xBins < 1) | (yBins < 1))
			{
				throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
			}
            
            double tlX = spdFile->getXMin();
            double tlY = spdFile->getYMax();
            
			std::vector<SPDPoint*>::iterator iterPts;
			
			try 
			{	
				double xDiff = 0;
				double yDiff = 0;
				boost::uint_fast32_t xIdx = 0;
				boost::uint_fast32_t yIdx = 0;
				
				SPDPoint *pt = NULL;
                std::vector<SPDPoint*>::iterator iterPts;
				for(iterPts = minPts->begin(); iterPts != minPts->end(); ++iterPts)
				{
					pt = *iterPts;
					
					xDiff = (pt->x - tlX)/binSize;
					yDiff = (tlY - pt->y)/binSize;				
					
					try 
					{
						xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
						yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
					}
					catch(boost::numeric::negative_overflow& e) 
					{
						throw SPDProcessingException(e.what());
					}
					catch(boost::numeric::positive_overflow& e) 
					{
						throw SPDProcessingException(e.what());
					}
					catch(boost::numeric::bad_numeric_cast& e) 
					{
						throw SPDProcessingException(e.what());
					}
					
					if(xIdx > (xBins-1))
					{
                        --xIdx;
                        if(xIdx > (xBins-1))
                        {
                            std::cout << "Point: [" << pt->x << "," << pt->y << "]\n";
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                            std::cout << "Size [" << xBins << "," << yBins << "]\n";
                            throw SPDProcessingException("Did not find x index within range.");
                        }
					}
					
					if(yIdx > (yBins-1))
					{
                        --yIdx;
                        if(yIdx > (yBins-1))
                        {
                            std::cout << "Point: [" << pt->x << "," << pt->y << "]\n";
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                            std::cout << "Size [" << xBins << "," << yBins << "]\n";
                            throw SPDProcessingException("Did not find y index within range.");
                        }
					}
					
					minPtGrid[yIdx][xIdx]->push_back(pt);
				}
			}
			catch (SPDProcessingException &e) 
			{
				throw e;
			}
			
		}
		else 
		{
			throw SPDProcessingException("Inputted list of points was empty.");
		}
    }
    
    SPDPolyFitGroundFilter::~SPDPolyFitGroundFilter()
    {
        
    }
    
    
    
    
    
    

    SPDPolyFitGroundLocalFilter::SPDPolyFitGroundLocalFilter(float grdThres, boost::uint_fast16_t degree, boost::uint_fast16_t iters, boost::uint_fast16_t ptSelectClass, float binWidth): SPDDataBlockProcessor()
    {
        this->grdThres = grdThres;
        this->degree = degree;
        this->iters = iters;
        this->ptSelectClass = ptSelectClass;
        this->binWidth = binWidth;
    }
        
    void SPDPolyFitGroundLocalFilter::processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) 
    {
        try
        {
            std::cout << "Block Y Size: " << ((float)ySize)*binWidth << " metres." << std::endl;
			std::cout << "Block X Size: " << ((float)xSize)*binWidth << " metres." << std::endl;
            
            bool first = true;
            SPDPoint *minPt = NULL;
            float minZ = 0;
            
            std::vector<SPDPoint*> **minPtGrid = new std::vector<SPDPoint*>*[ySize];
            for(boost::uint_fast32_t i = 0; i < ySize; ++i)
            {
                minPtGrid[i] = new std::vector<SPDPoint*>[xSize];
                for(boost::uint_fast32_t j = 0; j < xSize; ++j)
                {
                    first = true;
                    minPt = NULL;
                    minZ = 0;
                    for(std::vector<SPDPulse*>::iterator iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
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
                    if(!first)
                    {
                        minPtGrid[i][j].push_back(minPt);
                    }
                }
            }
            
            /********** Find Surface ***************/
			
			// calculate number of coefficients
			//unsigned int degree=2; // this will be user defined via command prompt in final version
			//this->degree = degree;
			//this->iters = iters;
			unsigned int Ncoeffs;
			//int l=0;
			//int p=0;
			int ItersCount=0;
			
			
			if (degree > 1)
			{
                Ncoeffs=((degree+1)*(2*degree))/2;
			}
			else
            {
				Ncoeffs=3; // plane fit (Const+Ax+By) hardcoded because above equation only works for orders of 2 or more
			}
            
			// Set up matrix of powers
			gsl_matrix *indVarPow = gsl_matrix_alloc(xSize*ySize,Ncoeffs); // Matrix to hold powers of x,y
			gsl_vector *depVar = gsl_vector_alloc(xSize*ySize); // Vector to hold z term
			gsl_vector *outCoefficients = gsl_vector_alloc(Ncoeffs); // Vector to hold output coefficients
            gsl_vector *SurfaceZ = gsl_vector_alloc(xSize*ySize);
			
			gsl_multifit_linear_workspace *workspace;
			workspace = gsl_multifit_linear_alloc(xSize*ySize, Ncoeffs);
			gsl_matrix *cov;
			double chisq;
			cov = gsl_matrix_alloc(Ncoeffs, Ncoeffs);
            
			bool keepProcessing = true;
			while(keepProcessing)
			{
				boost::uint_fast32_t p = 0;
				for(boost::uint_fast32_t i = 0; i < ySize; ++i)
				{
					for(boost::uint_fast32_t j = 0; j < xSize; ++j)
					{
						if(minPtGrid[i][j].size() > 0)
						{
							//std::cout << "Pt: " << minPtGrid[i][j]->front()->x << ", " << minPtGrid[i][j]->front()->y << ", " << minPtGrid[i][j]->front()->z << '\n';
							// generate arrays for x^ny^m and z
							double xelement=minPtGrid[i][j].front()->x;
							double yelement=minPtGrid[i][j].front()->y;
							double zelement=minPtGrid[i][j].front()->z;
							boost::uint_fast32_t l=0;
							gsl_vector_set(depVar, p, zelement);
							//++p;
							
							//std::cout << "NCoeffs: " << Ncoeffs << " Degree: " << degree << '\n';
							
							for (boost::uint_fast32_t m = 0; m < Ncoeffs ; m++)
							{
								for (boost::uint_fast32_t n=0; n < Ncoeffs ; n++)
								{
									if (n+m <= degree)
									{
										double xelementtPow = pow(xelement, ((int)(m)));
										double yelementtPow = pow(yelement, ((int)(n)));
										gsl_matrix_set(indVarPow,p,l, xelementtPow*yelementtPow); // was n,m instead of l
                                                                                                  //std::cout << "indvarpow: " << indVarPow << " " << xelementtPow << " " << yelementtPow << " " << "n,m: " << n << " " << m << '\n';
                                                                                                  //std::cout << "n,m: " << n << " " << m << " " << l << '\n';
										++l;
									}
								}
							}
							++p;
							
						}
					}
					
				}
				
				
				//std::cout << "Iters count" << ItersCount << " " << keepProcessing << '\n';
				
				
				// Find surface
				//matrix operations to find coefficients
				// K=U.inverse(Utrans.U)
				//Coeffs=K.z
				
				//LU = gsl_linalg_LU_decomp (gsl_matrix * indVarPow, gsl_permutation * perm, int * signum);
				//gsl_linalg_LU_solve (const gsl_matrix * LU, const gsl_permutation * perm, const gsl_vector * depVar, gsl_vector * outCoefficients);
				
				// Perform Least Squared Fit
				
                
				gsl_multifit_linear(indVarPow, depVar, outCoefficients, cov, &chisq, workspace);
				
				// calculate vector of z values from outCoefficients and indVarPow to compare with original data in next bit...
				//compute SurfaceZ = indVarPow.outCoefficients
				//gsl_blas_dgemv (CblasNoTrans, CblasNoTrans, 1.0, indVarPow, outCoefficients, 0.0, SurfaceZ);
				
				gsl_blas_dgemv (CblasNoTrans, 1.0, indVarPow, outCoefficients, 0.0, SurfaceZ);
				
				// compare polynomial surface with minimum points grid.
				// if there are points above a certain threshold, delete them
				// and then start the interation again to find a better surface fit for the ground.
				// this step is necessary to remove minimum points in branches residing over shadowed areas.
                
				/*
                for(unsigned int j = 0; j < outCoefficients->size; j++)
                {
                    double outm = gsl_vector_get(outCoefficients, j);
                    std::cout << outm << " , " ;
                }
                std::cout << '\n';
				*/
				
				
				
				// remove outlying points that are obviously not ground
				for(boost::uint_fast32_t i = 0; i < ySize; ++i)
				{
					for(boost::uint_fast32_t j = 0; j < xSize; ++j)
					{
						if(minPtGrid[i][j].size() > 0)
						{
							// to determine if points are above surface:
							double zelement=minPtGrid[i][j].front()->z;
							double xelement=minPtGrid[i][j].front()->x;
							double yelement=minPtGrid[i][j].front()->y;
							double sz=0; // reset z value from surface coefficients
							boost::uint_fast32_t l=0;
							
							for(boost::uint_fast32_t m = 0; m < Ncoeffs ; m++)
							{
								for(boost::uint_fast32_t n=0; n < Ncoeffs ; n++)
								{
									if(n+m <= degree)
									{
										double xelementtPow = pow(xelement, ((int)(m)));
										double yelementtPow = pow(yelement, ((int)(n)));
										double outm = gsl_vector_get(outCoefficients, l);
										
										sz=sz+(outm*xelementtPow*yelementtPow);
										++l;
									}
								}
							}
							
							if (zelement-sz > 1.0)
							{
								minPtGrid[i][j].clear(); // delete point
							}
							
							++l;
						}
					}
				}
				
				ItersCount++;
				if (ItersCount==iters)
				{
					keepProcessing = false;
				}
			}
            // end of while loop
			
            // Clean up
            gsl_multifit_linear_free(workspace);
			gsl_matrix_free(cov);
            gsl_matrix_free(indVarPow);
            gsl_vector_free(depVar);
            /***************************************/
            
            
            for(boost::uint_fast32_t i = 0; i < ySize; ++i)
            {
                delete[] minPtGrid[i];
                for(boost::uint_fast32_t j = 0; j < xSize; ++j)
                {
                    for(std::vector<SPDPulse*>::iterator iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
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
                                
                                for (boost::uint_fast32_t m = 0; m < outCoefficients->size ; m++)
                                {
                                    for (boost::uint_fast32_t n=0; n < outCoefficients->size ; n++)
                                    {
                                        if (n+m <= degree)
                                        {
                                            double xelementtPow = pow(xcoord, ((int)(m)));
                                            double yelementtPow = pow(ycoord, ((int)(n)));
                                            double outm = gsl_vector_get(outCoefficients, l);
                                            
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
                                
                                for (boost::uint_fast32_t m = 0; m < outCoefficients->size ; m++)
                                {
                                    for (boost::uint_fast32_t n=0; n < outCoefficients->size ; n++)
                                    {
                                        if (n+m <= degree)
                                        {
                                            double xelementtPow = pow(xcoord, ((int)(m)));
                                            double yelementtPow = pow(ycoord, ((int)(n)));
                                            double outm = gsl_vector_get(outCoefficients, l);
                                            
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
            }
            delete[] minPtGrid;
            
            gsl_vector_free(outCoefficients);
            gsl_vector_free(SurfaceZ);
        }
        catch (spdlib::SPDProcessingException &e)
        {
            throw e;
        }
    }
    
    
    SPDPolyFitGroundLocalFilter::~SPDPolyFitGroundLocalFilter()
    {
        
    }
    
}


