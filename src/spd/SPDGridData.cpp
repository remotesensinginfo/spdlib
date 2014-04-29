/*
 *  SPDGridData.cpp
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

#include "spd/SPDGridData.h"

namespace spdlib
{
	/* Public functions */
	 
	SPDGridData::SPDGridData()
	{
		
	}
	
	std::list<SPDPulse*>*** SPDGridData::gridData(std::list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException)
	{
		try 
		{
			if(pls->size() < 1)
			{
				throw SPDProcessingException("Inputted pulses list does not contain any pulses.");
			}
			
			if(spdFile->getIndexType() == SPD_CARTESIAN_IDX)
			{
				return this->gridDataCartesian(pls, spdFile);
			}
			else if (spdFile->getIndexType() == SPD_SPHERICAL_IDX)
			{
				return this->gridDataSpherical(pls, spdFile);
			}
            else if (spdFile->getIndexType() == SPD_CYLINDRICAL_IDX)
			{
				return this->gridDataCylindrical(pls, spdFile);
			}
            else if (spdFile->getIndexType() == SPD_POLAR_IDX)
            {
                return this->gridDataPolar(pls, spdFile);
            }
            else if (spdFile->getIndexType() == SPD_SCAN_IDX)
            {
                return this->gridDataScan(pls, spdFile);
            }
			else 
			{
				throw SPDProcessingException("Index type is not recognised");
			}
		}
		catch (SPDProcessingException &e) 
		{
			throw e;
		}
	}
	
	std::list<SPDPulse*>*** SPDGridData::gridData(std::vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException)
	{
		try 
		{
			if(pls->size() < 1)
			{
				throw SPDProcessingException("Inputted pulses list does not contain any pulses.");
			}
			
			if(spdFile->getIndexType() == SPD_CARTESIAN_IDX)
			{
				return this->gridDataCartesian(pls, spdFile);
			}
			else if (spdFile->getIndexType() == SPD_SPHERICAL_IDX)
			{
				return this->gridDataSpherical(pls, spdFile);
			}
            else if (spdFile->getIndexType() == SPD_CYLINDRICAL_IDX)
			{
				return this->gridDataCylindrical(pls, spdFile);
			}
            else if (spdFile->getIndexType() == SPD_POLAR_IDX)
			{
				return this->gridDataPolar(pls, spdFile);
			}
            else if (spdFile->getIndexType() == SPD_SCAN_IDX)
            {
                return this->gridDataScan(pls, spdFile);
            }
			else 
			{
				throw SPDProcessingException("Index type is not recognised");
			}
		}
		catch (SPDProcessingException &e) 
		{
			throw e;
		}
	}

    void SPDGridData::gridData(std::list<SPDPulse*>* pls, SPDFile *spdFile, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException)
    {
        try 
		{
			if(spdFile->getIndexType() == SPD_CARTESIAN_IDX)
			{
				this->gridDataCartesian(pls, grid, env, xSize, ySize, binsize);
			}
			else if (spdFile->getIndexType() == SPD_SPHERICAL_IDX)
			{
				this->gridDataSpherical(pls, grid, env, xSize, ySize, binsize);
			}
            else if (spdFile->getIndexType() == SPD_CYLINDRICAL_IDX)
			{
				this->gridDataCylindrical(pls, grid, env, xSize, ySize, binsize);
			}
            else if (spdFile->getIndexType() == SPD_POLAR_IDX)
			{
				this->gridDataPolar(pls, grid, env, xSize, ySize, binsize);
			}
            else if (spdFile->getIndexType() == SPD_SCAN_IDX)
			{
				this->gridDataScan(pls, grid, env, xSize, ySize, binsize);
			}
			else 
			{
				throw SPDProcessingException("Index type is not recognised");
			}
		}
		catch (SPDProcessingException &e) 
		{
			throw e;
		}
    }
    
    void SPDGridData::gridData(std::vector<SPDPulse*>* pls, SPDFile *spdFile, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException)
    {
        try 
		{
			if(spdFile->getIndexType() == SPD_CARTESIAN_IDX)
			{
				this->gridDataCartesian(pls, grid, env, xSize, ySize, binsize);
			}
			else if (spdFile->getIndexType() == SPD_SPHERICAL_IDX)
			{
				this->gridDataSpherical(pls, grid, env, xSize, ySize, binsize);
			}
            else if (spdFile->getIndexType() == SPD_CYLINDRICAL_IDX)
			{
				this->gridDataCylindrical(pls, grid, env, xSize, ySize, binsize);
			}
            else if (spdFile->getIndexType() == SPD_POLAR_IDX)
			{
				this->gridDataPolar(pls, grid, env, xSize, ySize, binsize);
			}
            else if (spdFile->getIndexType() == SPD_SCAN_IDX)
			{
				this->gridDataScan(pls, grid, env, xSize, ySize, binsize);
			}
			else 
			{
				throw SPDProcessingException("Index type is not recognised");
			}
		}
		catch (SPDProcessingException &e) 
		{
			throw e;
		}
    }
    
    void SPDGridData::reGridData(boost::uint_fast16_t indexType, std::vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, std::vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double originX, double originY, float outBinSize) throw(SPDProcessingException)
    {
        try 
		{
			if(indexType == SPD_CARTESIAN_IDX)
			{
                this->reGridDataCartesian(inGridPls, inXSize, inYSize, outGridPls, outXSize, outYSize, originX, originY, outBinSize);
			}
			else if (indexType == SPD_CYLINDRICAL_IDX)
			{
                this->reGridDataCylindrical(inGridPls, inXSize, inYSize, outGridPls, outXSize, outYSize, originX, originY, outBinSize);
			}
			else if (indexType == SPD_SPHERICAL_IDX)
			{
                this->reGridDataSpherical(inGridPls, inXSize, inYSize, outGridPls, outXSize, outYSize, originX, originY, outBinSize);
			}
            else if (indexType == SPD_POLAR_IDX)
			{
                this->reGridDataPolar(inGridPls, inXSize, inYSize, outGridPls, outXSize, outYSize, originX, originY, outBinSize);
			}
            else if (indexType == SPD_SCAN_IDX)
			{
                this->reGridDataScan(inGridPls, inXSize, inYSize, outGridPls, outXSize, outYSize, originX, originY, outBinSize);
			}
			else 
			{
				throw SPDProcessingException("Index type is not recognised");
			}
		}
		catch (SPDProcessingException &e) 
		{
			throw e;
		}
    }
    
    void SPDGridData::cartGridDataIgnoringOutGrid(std::vector<SPDPulse*>* pls, std::vector<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException)
    {
        if((xSize < 1) | (ySize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
        
        try
		{
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
            std::cout << "Started (Grid " << pls->size() << " Pulses) ." << std::flush;
            
			boost::uint_fast32_t feedback = pls->size()/10;
			boost::uint_fast32_t feedbackCounter = 0;
			
            bool ignorePls = false;
            boost::uint_fast64_t i = 0;
			for(std::vector<SPDPulse*>::iterator iterPls = pls->begin(); iterPls != pls->end(); ++iterPls)
            {
				if((pls->size() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
                ignorePls = false;
				xDiff = ((*iterPls)->xIdx - env->MinX)/binsize;
				yDiff = (env->MaxY - (*iterPls)->yIdx)/binsize;
                
				try
				{
					xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
					yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
				}
				catch(boost::numeric::negative_overflow& e)
				{
                    ignorePls = true;
				}
				catch(boost::numeric::positive_overflow& e)
				{
                    ignorePls = true;
				}
				catch(boost::numeric::bad_numeric_cast& e)
				{
                    ignorePls = true;
				}
				
                //std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                
				if((xIdx > ((xSize)-1)) | (yIdx > ((ySize)-1)))
				{
					ignorePls = true;
				}
                
                if(!ignorePls)
                {
                    //std::cout << "pushing back to the grid\n";
                    grid[yIdx][xIdx]->push_back((*iterPls));
                    //std::cout << "added to grid\n";
                }
                
                ++i;
			}
			
			std::cout << " Complete.\n";
		}
		catch (SPDProcessingException &e)
		{
            std::cout << e.what() << std::endl;
			throw e;
		}
    }
    
	/* Private functions */
	
	std::list<SPDPulse*>*** SPDGridData::gridDataCartesian(std::list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException)
	{
		boost::uint_fast32_t xSize = 0;
		boost::uint_fast32_t ySize = 0;
		boost::uint_fast32_t roundingAddition = 0;
		try
		{
			if(spdFile->getBinSize() < 1)
			{
				roundingAddition = boost::numeric_cast<boost::uint_fast32_t>(1/spdFile->getBinSize());
			}
			else 
			{
				roundingAddition = 2;
			}
			
			xSize = boost::numeric_cast<boost::uint_fast32_t>(((spdFile->getXMax()-spdFile->getXMin())/spdFile->getBinSize())+roundingAddition);
			ySize = boost::numeric_cast<boost::uint_fast32_t>(((spdFile->getYMax()-spdFile->getYMin())/spdFile->getBinSize())+roundingAddition);
			
			spdFile->setNumberBinsX(xSize);
			spdFile->setNumberBinsY(ySize);
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
				
		if((xSize < 1) | (ySize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
		
		std::list<SPDPulse*> ***griddedPls = new std::list<SPDPulse*>**[ySize];
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			griddedPls[i] = new std::list<SPDPulse*>*[xSize];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				griddedPls[i][j] = new std::list<SPDPulse*>();
			}
		}
		
		try 
		{	
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			boost::uint_fast32_t feedback = pls->size()/10;
			boost::uint_fast32_t feedbackCounter = 0;
			
			std::cout << "Started (Grid " << pls->size() << " Pulses) ." << std::flush;
			
			SPDPulse *pl = NULL;
			boost::uint_fast64_t iend = pls->size();
			for(boost::uint_fast64_t i = 0; i < iend; ++i)
			{
				if((pls->size() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
				pl = pls->back();
				pls->pop_back();
				
				xDiff = (pl->xIdx - spdFile->getXMin())/spdFile->getBinSize();
				yDiff = (spdFile->getYMax() - pl->yIdx)/spdFile->getBinSize();				
				
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
				
				if(xIdx > ((xSize)-1))
				{
					std::cout << "Point: [" << pl->xIdx << "," << pl->yIdx << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find x index within range.");
				}
				
				if(yIdx > ((ySize)-1))
				{
					std::cout << "Point: [" << pl->xIdx << "," << pl->yIdx << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find y index within range.");
				}
				
				griddedPls[yIdx][xIdx]->push_back(pl);
			}
			
			std::cout << " Complete.\n";
		}
		catch (SPDProcessingException &e) 
		{
			std::list<SPDPulse*>::iterator iterPls;
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					for(iterPls = griddedPls[i][j]->begin(); iterPls != griddedPls[i][j]->end(); )
					{
						delete *iterPls;
						griddedPls[i][j]->erase(iterPls++);
					}
					delete griddedPls[i][j];
				}
				delete[] griddedPls[i];
			}
			delete[] griddedPls;
			
			throw e;
		}
		
		return griddedPls;
	}

	std::list<SPDPulse*>*** SPDGridData::gridDataSpherical(std::list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException)
	{
		boost::uint_fast32_t xSize = 0;
		boost::uint_fast32_t ySize = 0;
		boost::uint_fast32_t roundingAddition = 0;
		try
		{
			if(spdFile->getBinSize() < 1)
			{
				roundingAddition = 2;//boost::numeric_cast<boost::uint_fast32_t>(1/spdFile->getBinSize());
			}
			else 
			{
				roundingAddition = 1;
			}
            			
			xSize = boost::numeric_cast<boost::uint_fast32_t>(((spdFile->getAzimuthMax()-spdFile->getAzimuthMin())/spdFile->getBinSize())+roundingAddition);
			ySize = boost::numeric_cast<boost::uint_fast32_t>(((spdFile->getZenithMax()-spdFile->getZenithMin())/spdFile->getBinSize())+roundingAddition);
			            
			spdFile->setNumberBinsX(xSize);
			spdFile->setNumberBinsY(ySize);
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
		
		if((xSize < 1) | (ySize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
		
		std::list<SPDPulse*> ***griddedPls = new std::list<SPDPulse*>**[ySize];
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			griddedPls[i] = new std::list<SPDPulse*>*[xSize];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				griddedPls[i][j] = new std::list<SPDPulse*>();
			}
		}
		
		try 
		{	
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			boost::uint_fast32_t feedback = pls->size()/10;
			boost::uint_fast32_t feedbackCounter = 0;
			
			std::cout << "Started (Grid " << pls->size() << " Pulses) ." << std::flush;
			SPDPulse *pl = NULL;
			boost::uint_fast64_t iend = pls->size();
			for(boost::uint_fast64_t i = 0; i < iend; ++i)
			{
				if((pls->size() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
				pl = pls->back();
				pls->pop_back();
				
				yDiff = (pl->zenith - spdFile->getZenithMin())/spdFile->getBinSize();
				xDiff = (pl->azimuth - spdFile->getAzimuthMin())/spdFile->getBinSize();				
				
				xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
				yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
				
				if(xIdx > ((xSize)-1))
				{
					std::cout << "Point: [" << pl->zenith << "," << pl->azimuth << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find x index within range.");
				}
				
				if(yIdx > ((ySize)-1))
				{
					std::cout << "Point: [" << pl->zenith << "," << pl->azimuth << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find y index within range.");
				}
				
				griddedPls[yIdx][xIdx]->push_back(pl);
			}
			
			std::cout << " Complete.\n";
		}
		catch (SPDProcessingException &e) 
		{
			std::list<SPDPulse*>::iterator iterPls;
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					for(iterPls = griddedPls[i][j]->begin(); iterPls != griddedPls[i][j]->end(); )
					{
						delete *iterPls;
						griddedPls[i][j]->erase(iterPls++);
					}
					delete griddedPls[i][j];
				}
				delete[] griddedPls[i];
			}
			delete[] griddedPls;
			
			throw e;
		}
		
		return griddedPls;
	}
    
    std::list<SPDPulse*>*** SPDGridData::gridDataCylindrical(std::list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException)
	{
	    throw SPDProcessingException("Cylindrical gridding not implemented yet... gridDataCylindrical");      
    }  
      
    std::list<SPDPulse*>*** SPDGridData::gridDataPolar(std::list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException)
	{
		boost::uint_fast32_t xSize = 0;
		boost::uint_fast32_t ySize = 0;
        boost::uint_fast32_t zenithRadiusInBins = 0;
		boost::uint_fast32_t roundingAddition = 0;
		try
		{
			if(spdFile->getBinSize() < 1)
			{
				roundingAddition = 2;//boost::numeric_cast<boost::uint_fast32_t>(1/spdFile->getBinSize());
			}
			else 
			{
				roundingAddition = 1;
			}
			
            zenithRadiusInBins = boost::numeric_cast<boost::uint_fast32_t>(((spdFile->getZenithMax()-spdFile->getZenithMin())/spdFile->getBinSize())+roundingAddition);
            
			xSize = (zenithRadiusInBins+5) * 2;
			ySize = (zenithRadiusInBins+5) * 2;
			
			spdFile->setNumberBinsX(xSize);
			spdFile->setNumberBinsY(ySize);
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
		
		if((xSize < 1) | (ySize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
		
		std::list<SPDPulse*> ***griddedPls = new std::list<SPDPulse*>**[ySize];
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			griddedPls[i] = new std::list<SPDPulse*>*[xSize];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				griddedPls[i][j] = new std::list<SPDPulse*>();
			}
		}
        
        boost::uint_fast32_t xCentre = zenithRadiusInBins+5;
		boost::uint_fast32_t yCentre = zenithRadiusInBins+5;
        double constRange = (double)zenithRadiusInBins;
        
        try 
		{	
            double xTmp = 0;
			double yTmp = 0;
            double zTmp = 0;
            
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			boost::uint_fast32_t feedback = pls->size()/10;
			boost::uint_fast32_t feedbackCounter = 0;
			
			std::cout << "Started (Grid " << pls->size() << " Pulses) ." << std::flush;
			SPDPulse *pl = NULL;
			boost::uint_fast64_t iend = pls->size();
			for(boost::uint_fast64_t i = 0; i < iend; ++i)
			{
				if((pls->size() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
				pl = pls->back();
				pls->pop_back();
                
                SPDConvertToCartesian(pl->zenith, pl->azimuth, constRange, 0, 0, 0, &xTmp, &yTmp, &zTmp);
				
				xIdx = boost::numeric_cast<boost::uint_fast32_t>(xCentre + xTmp);
				yIdx = boost::numeric_cast<boost::uint_fast32_t>(yCentre + yTmp);
                
				if(xIdx > ((xSize)-1))
				{
					std::cout << "Point: [" << pl->zenith << "," << pl->azimuth << "]\n";
                    std::cout << "Out [X,Y,Z] = [" << xTmp << "," << yTmp << "," << zTmp << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
                    std::cout << "Centre: [" << xCentre << "," << yCentre << "]\n";
					throw SPDProcessingException("Did not find x index within range.");
				}
				
				if(yIdx > ((ySize)-1))
				{
					std::cout << "Point: [" << pl->zenith << "," << pl->azimuth << "]\n";
					std::cout << "Out [X,Y,Z] = [" << xTmp << "," << yTmp << "," << zTmp << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
                    std::cout << "Centre: [" << xCentre << "," << yCentre << "]\n";
					throw SPDProcessingException("Did not find y index within range.");
				}
				
				griddedPls[yIdx][xIdx]->push_back(pl);
			}
			
			std::cout << " Complete.\n";
		}
		catch (SPDProcessingException &e) 
		{
			std::list<SPDPulse*>::iterator iterPls;
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					for(iterPls = griddedPls[i][j]->begin(); iterPls != griddedPls[i][j]->end(); )
					{
						delete *iterPls;
						griddedPls[i][j]->erase(iterPls++);
					}
					delete griddedPls[i][j];
				}
				delete[] griddedPls[i];
			}
			delete[] griddedPls;
			
			throw e;
		}
        
        return griddedPls;
	}
    
    std::list<SPDPulse*>*** SPDGridData::gridDataScan(std::list<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException)
	{
		boost::uint_fast32_t xSize = 0;
		boost::uint_fast32_t ySize = 0;
		boost::uint_fast32_t roundingAddition = 1;	
		
		try
		{			
			xSize = boost::numeric_cast<boost::uint_fast32_t>(((spdFile->getScanlineIdxMax()-spdFile->getScanlineIdxMin())/spdFile->getBinSize())+roundingAddition);
			ySize = boost::numeric_cast<boost::uint_fast32_t>(((spdFile->getScanlineMax()-spdFile->getScanlineMin())/spdFile->getBinSize())+roundingAddition);
			
			spdFile->setNumberBinsX(xSize);
			spdFile->setNumberBinsY(ySize);
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
				
		if((xSize < 1) | (ySize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
		
		std::list<SPDPulse*> ***griddedPls = new std::list<SPDPulse*>**[ySize];
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			griddedPls[i] = new std::list<SPDPulse*>*[xSize];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				griddedPls[i][j] = new std::list<SPDPulse*>();
			}
		}
		
		try 
		{	
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			boost::uint_fast32_t feedback = pls->size()/10;
			boost::uint_fast32_t feedbackCounter = 0;
			
			std::cout << "Started (Grid " << pls->size() << " Pulses) ." << std::flush;
			
			SPDPulse *pl = NULL;
			boost::uint_fast64_t iend = pls->size();
			for(boost::uint_fast64_t i = 0; i < iend; ++i)
			{
				if((pls->size() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
				pl = pls->back();
				pls->pop_back();
				
				xDiff = (pl->scanlineIdx - spdFile->getScanlineIdxMin())/spdFile->getBinSize();		
				yDiff = (pl->scanline - spdFile->getScanlineMin())/spdFile->getBinSize();		
				
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
				
				if(xIdx > ((xSize)-1))
				{
					std::cout << "Point: [" << pl->scanlineIdx << "," << pl->scanline << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find ScanlineIdx (x) index within range.");
				}
				
				if(yIdx > ((ySize)-1))
				{
					std::cout << "Point: [" << pl->scanlineIdx << "," << pl->scanline << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find Scanline (y) index within range.");
				}
				
				griddedPls[yIdx][xIdx]->push_back(pl);
			}
			
			std::cout << " Complete.\n";
		}
		catch (SPDProcessingException &e) 
		{
			std::list<SPDPulse*>::iterator iterPls;
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					for(iterPls = griddedPls[i][j]->begin(); iterPls != griddedPls[i][j]->end(); )
					{
						delete *iterPls;
						griddedPls[i][j]->erase(iterPls++);
					}
					delete griddedPls[i][j];
				}
				delete[] griddedPls[i];
			}
			delete[] griddedPls;
			
			throw e;
		}
		
		return griddedPls;
	}
    
    
	std::list<SPDPulse*>*** SPDGridData::gridDataCartesian(std::vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException)
	{
		boost::uint_fast32_t xSize = 0;
		boost::uint_fast32_t ySize = 0;
		boost::uint_fast32_t roundingAddition = 0;
		try
		{
			if(spdFile->getBinSize() < 1)
			{
				roundingAddition = boost::numeric_cast<boost::uint_fast32_t>(1/spdFile->getBinSize());
			}
			else 
			{
				roundingAddition = 2;
			}
            
			xSize = boost::numeric_cast<boost::uint_fast32_t>(((spdFile->getXMax()-spdFile->getXMin())/spdFile->getBinSize())+roundingAddition);
			ySize = boost::numeric_cast<boost::uint_fast32_t>(((spdFile->getYMax()-spdFile->getYMin())/spdFile->getBinSize())+roundingAddition);
			
			spdFile->setNumberBinsX(xSize);
			spdFile->setNumberBinsY(ySize);
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
        
		if((xSize < 1) | (ySize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
		
		std::list<SPDPulse*> ***griddedPls = new std::list<SPDPulse*>**[ySize];
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			griddedPls[i] = new std::list<SPDPulse*>*[xSize];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				griddedPls[i][j] = new std::list<SPDPulse*>();
			}
		}
		
		try 
		{	
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			boost::uint_fast32_t feedback = pls->size()/10;
			boost::uint_fast32_t feedbackCounter = 0;
			
			std::cout << "Started (Grid " << pls->size() << " Pulses) ." << std::flush;
			
			SPDPulse *pl = NULL;
			boost::uint_fast64_t iend = pls->size();
			for(boost::uint_fast64_t i = 0; i < iend; ++i)
			{
				if((pls->size() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
				pl = pls->back();
				pls->pop_back();
				
				xDiff = (pl->xIdx - spdFile->getXMin())/spdFile->getBinSize();
				yDiff = (spdFile->getYMax() - pl->yIdx)/spdFile->getBinSize();				
				
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
				
				if(xIdx > ((xSize)-1))
				{
					std::cout << "Point: [" << pl->xIdx << "," << pl->yIdx << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find x index within range.");
				}
				
				if(yIdx > ((ySize)-1))
				{
					std::cout << "Point: [" << pl->xIdx << "," << pl->yIdx << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find y index within range.");
				}
				
				griddedPls[yIdx][xIdx]->push_back(pl);
			}
			
			std::cout << " Complete.\n";
		}
		catch (SPDProcessingException &e) 
		{
			std::list<SPDPulse*>::iterator iterPls;
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					for(iterPls = griddedPls[i][j]->begin(); iterPls != griddedPls[i][j]->end(); )
					{
						delete *iterPls;
						griddedPls[i][j]->erase(iterPls++);
					}
					delete griddedPls[i][j];
				}
				delete[] griddedPls[i];
			}
			delete[] griddedPls;
			
			throw e;
		}
		
		return griddedPls;
	}
	
	std::list<SPDPulse*>*** SPDGridData::gridDataSpherical(std::vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException)
	{
		boost::uint_fast32_t xSize = 0;
		boost::uint_fast32_t ySize = 0;
		boost::uint_fast32_t roundingAddition = 0;
		try
		{
			if(spdFile->getBinSize() < 1)
			{
				roundingAddition = 2;//boost::numeric_cast<boost::uint_fast32_t>(1/spdFile->getBinSize());
			}
			else 
			{
				roundingAddition = 1;
			}
            
			xSize = boost::numeric_cast<boost::uint_fast32_t>(((spdFile->getAzimuthMax()-spdFile->getAzimuthMin())/spdFile->getBinSize())+roundingAddition);
			ySize = boost::numeric_cast<boost::uint_fast32_t>(((spdFile->getZenithMax()-spdFile->getZenithMin())/spdFile->getBinSize())+roundingAddition);
			
			spdFile->setNumberBinsX(xSize);
			spdFile->setNumberBinsY(ySize);
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
		
		if((xSize < 1) | (ySize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
		
		std::list<SPDPulse*> ***griddedPls = new std::list<SPDPulse*>**[ySize];
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			griddedPls[i] = new std::list<SPDPulse*>*[xSize];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				griddedPls[i][j] = new std::list<SPDPulse*>();
			}
		}
		
		try 
		{	
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			boost::uint_fast32_t feedback = pls->size()/10;
			boost::uint_fast32_t feedbackCounter = 0;
			
			std::cout << "Started (Grid " << pls->size() << " Pulses) ." << std::flush;
			SPDPulse *pl = NULL;
			boost::uint_fast64_t iend = pls->size();
			for(boost::uint_fast64_t i = 0; i < iend; ++i)
			{
				if((pls->size() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
				pl = pls->back();
				pls->pop_back();
				
				yDiff = (pl->zenith - spdFile->getZenithMin())/spdFile->getBinSize();
				xDiff = (pl->azimuth - spdFile->getAzimuthMin())/spdFile->getBinSize();				
				
				xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
				yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
				
				if(xIdx > ((xSize)-1))
				{
					std::cout << "Point: [" << pl->zenith << "," << pl->azimuth << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find x index within range.");
				}
				
				if(yIdx > ((ySize)-1))
				{
					std::cout << "Point: [" << pl->zenith << "," << pl->azimuth << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find y index within range.");
				}
				
				griddedPls[yIdx][xIdx]->push_back(pl);
			}
			
			std::cout << " Complete.\n";
		}
		catch (SPDProcessingException &e) 
		{
			std::list<SPDPulse*>::iterator iterPls;
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					for(iterPls = griddedPls[i][j]->begin(); iterPls != griddedPls[i][j]->end(); )
					{
						delete *iterPls;
						griddedPls[i][j]->erase(iterPls++);
					}
					delete griddedPls[i][j];
				}
				delete[] griddedPls[i];
			}
			delete[] griddedPls;
			
			throw e;
		}
		
		return griddedPls;
    }
    
    std::list<SPDPulse*>*** SPDGridData::gridDataCylindrical(std::vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException)
	{
	    throw SPDProcessingException("Cylindrical gridding not implemented yet... gridDataCylindrical");    
    }
    
    std::list<SPDPulse*>*** SPDGridData::gridDataPolar(std::vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException)
	{
		boost::uint_fast32_t xSize = 0;
		boost::uint_fast32_t ySize = 0;
        boost::uint_fast32_t zenithRadiusInBins = 0;
		boost::uint_fast32_t roundingAddition = 0;
		try
		{
			if(spdFile->getBinSize() < 1)
			{
				roundingAddition = 2;//boost::numeric_cast<boost::uint_fast32_t>(1/spdFile->getBinSize());
			}
			else 
			{
				roundingAddition = 1;
			}
			
            zenithRadiusInBins = boost::numeric_cast<boost::uint_fast32_t>(((spdFile->getZenithMax()-spdFile->getZenithMin())/spdFile->getBinSize())+roundingAddition);
            
			xSize = (zenithRadiusInBins+5) * 2;
			ySize = (zenithRadiusInBins+5) * 2;
			
			spdFile->setNumberBinsX(xSize);
			spdFile->setNumberBinsY(ySize);
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
		
		if((xSize < 1) | (ySize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
		
		std::list<SPDPulse*> ***griddedPls = new std::list<SPDPulse*>**[ySize];
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			griddedPls[i] = new std::list<SPDPulse*>*[xSize];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				griddedPls[i][j] = new std::list<SPDPulse*>();
			}
		}
        
        boost::uint_fast32_t xCentre = zenithRadiusInBins+5;
		boost::uint_fast32_t yCentre = zenithRadiusInBins+5;
        double constRange = (double)zenithRadiusInBins;
        
        try 
		{	
            double xTmp = 0;
			double yTmp = 0;
            double zTmp = 0;
            
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			boost::uint_fast32_t feedback = pls->size()/10;
			boost::uint_fast32_t feedbackCounter = 0;
			
			std::cout << "Started (Grid " << pls->size() << " Pulses) ." << std::flush;
			SPDPulse *pl = NULL;
			boost::uint_fast64_t iend = pls->size();
			for(boost::uint_fast64_t i = 0; i < iend; ++i)
			{
				if((pls->size() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
				pl = pls->back();
				pls->pop_back();
                
                SPDConvertToCartesian(pl->zenith, pl->azimuth, constRange, 0, 0, 0, &xTmp, &yTmp, &zTmp);
				
				xIdx = boost::numeric_cast<boost::uint_fast32_t>(xCentre + xTmp);
				yIdx = boost::numeric_cast<boost::uint_fast32_t>(yCentre + yTmp);
                				
				if(xIdx > ((xSize)-1))
				{
					std::cout << "Point: [" << pl->zenith << "," << pl->azimuth << "]\n";
                    std::cout << "Out [X,Y,Z] = [" << xTmp << "," << yTmp << "," << zTmp << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
                    std::cout << "Centre: [" << xCentre << "," << yCentre << "]\n";
					throw SPDProcessingException("Did not find x index within range.");
				}
				
				if(yIdx > ((ySize)-1))
				{
					std::cout << "Point: [" << pl->zenith << "," << pl->azimuth << "]\n";
					std::cout << "Out [X,Y,Z] = [" << xTmp << "," << yTmp << "," << zTmp << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
                    std::cout << "Centre: [" << xCentre << "," << yCentre << "]\n";
					throw SPDProcessingException("Did not find y index within range.");
				}
				
				griddedPls[yIdx][xIdx]->push_back(pl);
			}
			
			std::cout << " Complete.\n";
		}
		catch (SPDProcessingException &e) 
		{
			std::list<SPDPulse*>::iterator iterPls;
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					for(iterPls = griddedPls[i][j]->begin(); iterPls != griddedPls[i][j]->end(); )
					{
						delete *iterPls;
						griddedPls[i][j]->erase(iterPls++);
					}
					delete griddedPls[i][j];
				}
				delete[] griddedPls[i];
			}
			delete[] griddedPls;
			
			throw e;
		}
        
        return griddedPls;
	}

    std::list<SPDPulse*>*** SPDGridData::gridDataScan(std::vector<SPDPulse*>* pls, SPDFile *spdFile) throw(SPDProcessingException)
	{
		boost::uint_fast32_t xSize = 0;
		boost::uint_fast32_t ySize = 0;
		boost::uint_fast32_t roundingAddition = 1;
		
		try
		{			
			xSize = boost::numeric_cast<boost::uint_fast32_t>(((spdFile->getScanlineIdxMax()-spdFile->getScanlineIdxMin())/spdFile->getBinSize())+roundingAddition);
			ySize = boost::numeric_cast<boost::uint_fast32_t>(((spdFile->getScanlineMax()-spdFile->getScanlineMin())/spdFile->getBinSize())+roundingAddition);
			
			spdFile->setNumberBinsX(xSize);
			spdFile->setNumberBinsY(ySize);
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
				
		if((xSize < 1) | (ySize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
		
		std::list<SPDPulse*> ***griddedPls = new std::list<SPDPulse*>**[ySize];
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			griddedPls[i] = new std::list<SPDPulse*>*[xSize];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				griddedPls[i][j] = new std::list<SPDPulse*>();
			}
		}
		
		try 
		{	
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			boost::uint_fast32_t feedback = pls->size()/10;
			boost::uint_fast32_t feedbackCounter = 0;
			
			std::cout << "Started (Grid " << pls->size() << " Pulses) ." << std::flush;
			
			SPDPulse *pl = NULL;
			boost::uint_fast64_t iend = pls->size();
			for(boost::uint_fast64_t i = 0; i < iend; ++i)
			{
				if((pls->size() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
				pl = pls->back();
				pls->pop_back();
				
				xDiff = (pl->scanlineIdx - spdFile->getScanlineIdxMin())/spdFile->getBinSize();	
				yDiff = (pl->scanline - spdFile->getScanlineMin())/spdFile->getBinSize();			
				
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
				
				if(xIdx > ((xSize)-1))
				{
					std::cout << "Point: [" << pl->scanlineIdx << "," << pl->scanline << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find ScanlineIdx (x) index within range.");
				}
				
				if(yIdx > ((ySize)-1))
				{
					std::cout << "Point: [" << pl->scanlineIdx << "," << pl->scanline << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find Scanline (y) index within range.");
				}
				
				griddedPls[yIdx][xIdx]->push_back(pl);
			}
			
			std::cout << " Complete.\n";
		}
		catch (SPDProcessingException &e) 
		{
			std::list<SPDPulse*>::iterator iterPls;
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					for(iterPls = griddedPls[i][j]->begin(); iterPls != griddedPls[i][j]->end(); )
					{
						delete *iterPls;
						griddedPls[i][j]->erase(iterPls++);
					}
					delete griddedPls[i][j];
				}
				delete[] griddedPls[i];
			}
			delete[] griddedPls;
			
			throw e;
		}
		
		return griddedPls;
	}
    
    void SPDGridData::gridDataCartesian(std::list<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException)
    {
        if((xSize < 1) | (ySize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
        
        try 
		{	
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			boost::uint_fast32_t feedback = pls->size()/10;
			boost::uint_fast32_t feedbackCounter = 0;
			
			std::cout << "Started (Grid " << pls->size() << " Pulses) ." << std::flush;
			SPDPulse *pl = NULL;
			boost::uint_fast64_t iend = pls->size();
			for(boost::uint_fast64_t i = 0; i < iend; ++i)
			{
				if((pls->size() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
				pl = pls->back();
				pls->pop_back();
				
				xDiff = (pl->xIdx - env->MinX)/binsize;
				yDiff = (env->MaxY - pl->yIdx)/binsize;				
				
				try 
				{
					xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
					yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
				}
				catch(boost::numeric::negative_overflow& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				catch(boost::numeric::positive_overflow& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				catch(boost::numeric::bad_numeric_cast& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				
				if(xIdx > ((xSize)-1))
				{
					std::cout << "Point: [" << pl->xIdx << "," << pl->yIdx << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find x index within range.");
				}
				
				if(yIdx > ((ySize)-1))
				{
					std::cout << "Point: [" << pl->xIdx << "," << pl->yIdx << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find y index within range.");
				}
				
				grid[yIdx][xIdx]->push_back(pl);
			}
			
			std::cout << " Complete.\n";
		}
		catch (SPDProcessingException &e) 
		{			
			throw e;
		}
    }
    
    void SPDGridData::gridDataSpherical(std::list<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException)
    {
        if((xSize < 1) | (ySize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
        
        try 
		{	
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			boost::uint_fast32_t feedback = pls->size()/10;
			boost::uint_fast32_t feedbackCounter = 0;
			
			std::cout << "Started (Grid " << pls->size() << " Pulses) ." << std::flush;
			SPDPulse *pl = NULL;
			boost::uint_fast64_t iend = pls->size();
			for(boost::uint_fast64_t i = 0; i < iend; ++i)
			{
				if((pls->size() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
				pl = pls->back();
				pls->pop_back();
				
                yDiff = (pl->zenith - env->MinY)/binsize;
				xDiff = (pl->azimuth - env->MinX)/binsize;								
                
                try 
				{
					xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
					yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
				}
				catch(boost::numeric::negative_overflow& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				catch(boost::numeric::positive_overflow& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				catch(boost::numeric::bad_numeric_cast& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				
				if(xIdx > ((xSize)-1))
				{
					std::cout << "Point: [" << pl->zenith << "," << pl->azimuth << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find x index within range.");
				}
				
				if(yIdx > ((ySize)-1))
				{
					std::cout << "Point: [" << pl->zenith << "," << pl->azimuth << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find y index within range.");
				}
				
				grid[yIdx][xIdx]->push_back(pl);
			}
			
			std::cout << " Complete.\n";
		}
		catch (SPDProcessingException &e) 
		{
			throw e;
		}

    }

    void SPDGridData::gridDataCylindrical(std::list<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException)
    {
        throw SPDProcessingException("Cylindrical gridding not implemented yet... gridDataCylindrical");
    }
    
    void SPDGridData::gridDataPolar(std::list<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException)
    {
        throw SPDProcessingException("Polar gridding not implemented yet... gridDataPolar");
    }
    
    void SPDGridData::gridDataScan(std::list<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException)
    {
        if((xSize < 1) | (ySize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
        
        try 
		{	
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			boost::uint_fast32_t feedback = pls->size()/10;
			boost::uint_fast32_t feedbackCounter = 0;
			
			std::cout << "Started (Grid " << pls->size() << " Pulses) ." << std::flush;
			SPDPulse *pl = NULL;
			boost::uint_fast64_t iend = pls->size();
			for(boost::uint_fast64_t i = 0; i < iend; ++i)
			{
				if((pls->size() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
				pl = pls->back();
				pls->pop_back();
				
				xDiff = (pl->scanlineIdx - env->MinX)/binsize;
				yDiff = (pl->scanline - env->MinY)/binsize;
				
				try 
				{
					xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
					yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
				}
				catch(boost::numeric::negative_overflow& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				catch(boost::numeric::positive_overflow& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				catch(boost::numeric::bad_numeric_cast& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				
				if(xIdx > ((xSize)-1))
				{
					std::cout << "Point: [" << pl->scanlineIdx << "," << pl->scanline << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find scanlineIdx (x) index within range.");
				}
				
				if(yIdx > ((ySize)-1))
				{
					std::cout << "Point: [" << pl->scanlineIdx << "," << pl->scanline << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find scanline (y) index within range.");
				}
				
				grid[yIdx][xIdx]->push_back(pl);
			}
			
			std::cout << " Complete.\n";
		}
		catch (SPDProcessingException &e) 
		{			
			throw e;
		}
    }
    
    void SPDGridData::gridDataCartesian(std::vector<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException)
    {
        if((xSize < 1) | (ySize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
        
        try 
		{	
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
            std::cout << "Started (Grid " << pls->size() << " Pulses) ." << std::flush;
            
			boost::uint_fast32_t feedback = pls->size()/10;
			boost::uint_fast32_t feedbackCounter = 0;
			
			SPDPulse *pl = NULL;
			boost::uint_fast64_t iend = pls->size();
			for(boost::uint_fast64_t i = 0; i < iend; ++i)
			{
				if((pls->size() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
                //std::cout << "Getting pulse " << i << std::endl;
				pl = pls->back();
				pls->pop_back();
                //std::cout << "Got pulse " << i << std::endl;
				
				xDiff = (pl->xIdx - env->MinX)/binsize;
				yDiff = (env->MaxY - pl->yIdx)/binsize;
								
				try 
				{
					xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
					yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
				}
				catch(boost::numeric::negative_overflow& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				catch(boost::numeric::positive_overflow& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				catch(boost::numeric::bad_numeric_cast& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				
                //std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                
				if(xIdx > ((xSize)-1))
				{
					std::cout << "Point: [" << pl->xIdx << "," << pl->yIdx << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find x index within range.");
				}
				
				if(yIdx > ((ySize)-1))
				{
					std::cout << "Point: [" << pl->xIdx << "," << pl->yIdx << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find y index within range.");
				}
				
                //std::cout << "pushing back to the grid\n";
				grid[yIdx][xIdx]->push_back(pl);
                //std::cout << "added to grid\n";
			}
			
			std::cout << " Complete.\n";
		}
		catch (SPDProcessingException &e) 
		{			
            std::cout << e.what() << std::endl;
			throw e;
		}
    }
    
    void SPDGridData::gridDataSpherical(std::vector<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException)
    {
        if((xSize < 1) | (ySize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
        
        try 
		{	
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			boost::uint_fast32_t feedback = pls->size()/10;
			boost::uint_fast32_t feedbackCounter = 0;
			
			std::cout << "Started (Grid " << pls->size() << " Pulses) ." << std::flush;
			SPDPulse *pl = NULL;
			boost::uint_fast64_t iend = pls->size();
			for(boost::uint_fast64_t i = 0; i < iend; ++i)
			{
				if((pls->size() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
				pl = pls->back();
				pls->pop_back();
				
                yDiff = (pl->zenith - env->MinY)/binsize;
				xDiff = (pl->azimuth - env->MinX)/binsize;								
                
                try 
				{
					xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
					yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
				}
				catch(boost::numeric::negative_overflow& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				catch(boost::numeric::positive_overflow& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				catch(boost::numeric::bad_numeric_cast& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				
				if(xIdx > ((xSize)-1))
				{
					std::cout << "Point: [" << pl->zenith << "," << pl->azimuth << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find x index within range.");
				}
				
				if(yIdx > ((ySize)-1))
				{
					std::cout << "Point: [" << pl->zenith << "," << pl->azimuth << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find y index within range.");
				}
				
				grid[yIdx][xIdx]->push_back(pl);
			}
			
			std::cout << " Complete.\n";
		}
		catch (SPDProcessingException &e) 
		{
			throw e;
		}
    }

    void SPDGridData::gridDataCylindrical(std::vector<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException)
    {
        throw SPDProcessingException("Cylindrical gridding not implemented yet... gridDataCylindrical");
    }
    
    void SPDGridData::gridDataPolar(std::vector<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException)
    {
        throw SPDProcessingException("Polar gridding not implemented yet... gridDataPolar");
    }
    
    void SPDGridData::gridDataScan(std::vector<SPDPulse*>* pls, std::list<SPDPulse*>*** grid, OGREnvelope *env, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binsize) throw(SPDProcessingException)
    {
        if((xSize < 1) | (ySize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
        
        try 
		{	
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			boost::uint_fast32_t feedback = pls->size()/10;
			boost::uint_fast32_t feedbackCounter = 0;
			
			std::cout << "Started (Grid " << pls->size() << " Pulses) ." << std::flush;
			SPDPulse *pl = NULL;
			boost::uint_fast64_t iend = pls->size();
			for(boost::uint_fast64_t i = 0; i < iend; ++i)
			{
				if((pls->size() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
				pl = pls->back();
				pls->pop_back();
				
				xDiff = (pl->scanlineIdx - env->MinX)/binsize;
				yDiff = (pl->scanline - env->MinY)/binsize;				
				
				try 
				{
					xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
					yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
				}
				catch(boost::numeric::negative_overflow& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				catch(boost::numeric::positive_overflow& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				catch(boost::numeric::bad_numeric_cast& e) 
				{
                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					throw SPDProcessingException(e.what());
				}
				
				if(xIdx > ((xSize)-1))
				{
					std::cout << "Point: [" << pl->scanlineIdx << "," << pl->scanline << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find scanlineIdx (x) index within range.");
				}
				
				if(yIdx > ((ySize)-1))
				{
					std::cout << "Point: [" << pl->scanlineIdx << "," << pl->scanline << "]\n";
					std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
					std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
					std::cout << "Size [" << xSize << "," << ySize << "]\n";
					throw SPDProcessingException("Did not find scanline (y) index within range.");
				}
				
				grid[yIdx][xIdx]->push_back(pl);
			}
			
			std::cout << " Complete.\n";
		}
		catch (SPDProcessingException &e) 
		{			
			throw e;
		}
    }    
    
    
    
    
    
	void SPDGridData::reGridDataCartesian(std::vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, std::vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double originX, double originY, float outBinSize) throw(SPDProcessingException)
    {
        if((outXSize < 1) | (outYSize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
        
        try 
		{	
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			
			SPDPulse *pl = NULL;
			for(boost::uint_fast64_t i = 0; i < inYSize; ++i)
			{
                for(boost::uint_fast64_t j = 0; j < inXSize; ++j)
                {
                    for(std::vector<SPDPulse*>::iterator iterPls = inGridPls[i][j]->begin(); iterPls != inGridPls[i][j]->end(); ++iterPls)
                    {
                        pl = (*iterPls);
				
                        xDiff = (pl->xIdx - originX)/outBinSize;
                        yDiff = (originY - pl->yIdx)/outBinSize;
                        
                        try 
                        {
                            xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
                            yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
                        }
                        catch(boost::numeric::negative_overflow& e) 
                        {
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            throw SPDProcessingException(e.what());
                        }
                        catch(boost::numeric::positive_overflow& e) 
                        {
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            throw SPDProcessingException(e.what());
                        }
                        catch(boost::numeric::bad_numeric_cast& e) 
                        {
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            throw SPDProcessingException(e.what());
                        }
                        
                        //std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                        
                        if(xIdx > ((outXSize)-1))
                        {
                            std::cout << "Point: [" << pl->xIdx << "," << pl->yIdx << "]\n";
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                            std::cout << "Size [" << outXSize << "," << outYSize << "]\n";
                            throw SPDProcessingException("Did not find x index within range.");
                        }
                        
                        if(yIdx > ((outYSize)-1))
                        {
                            std::cout << "Point: [" << pl->xIdx << "," << pl->yIdx << "]\n";
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                            std::cout << "Size [" << outXSize << "," << outYSize << "]\n";
                            throw SPDProcessingException("Did not find y index within range.");
                        }
                        
                        outGridPls[yIdx][xIdx]->push_back(pl);
                    }
                }
			}			
		}
		catch (SPDProcessingException &e) 
		{			
            std::cout << e.what() << std::endl;
			throw e;
		}
    }
    
    void SPDGridData::reGridDataSpherical(std::vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, std::vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double originX, double originY, float outBinSize) throw(SPDProcessingException)
    {
        if((outXSize < 1) | (outYSize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
        
        try 
		{	
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			
			SPDPulse *pl = NULL;
			for(boost::uint_fast64_t i = 0; i < inYSize; ++i)
			{
                for(boost::uint_fast64_t j = 0; j < inXSize; ++j)
                {
                    for(std::vector<SPDPulse*>::iterator iterPls = inGridPls[i][j]->begin(); iterPls != inGridPls[i][j]->end(); ++iterPls)
                    {
                        pl = (*iterPls);
                        
                        yDiff = (pl->zenith - originY)/outBinSize;
                        xDiff = (pl->azimuth - originX)/outBinSize;								
                        
                        try 
                        {
                            xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
                            yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
                        }
                        catch(boost::numeric::negative_overflow& e) 
                        {
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            throw SPDProcessingException(e.what());
                        }
                        catch(boost::numeric::positive_overflow& e) 
                        {
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            throw SPDProcessingException(e.what());
                        }
                        catch(boost::numeric::bad_numeric_cast& e) 
                        {
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            throw SPDProcessingException(e.what());
                        }
                        
                        if(xIdx > ((outXSize)-1))
                        {
                            std::cout << "Point: [" << pl->zenith << "," << pl->azimuth << "]\n";
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                            std::cout << "Size [" << outXSize << "," << outYSize << "]\n";
                            throw SPDProcessingException("Did not find x index within range.");
                        }
                        
                        if(yIdx > ((outYSize)-1))
                        {
                            std::cout << "Point: [" << pl->zenith << "," << pl->azimuth << "]\n";
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                            std::cout << "Size [" << outXSize << "," << outYSize << "]\n";
                            throw SPDProcessingException("Did not find y index within range.");
                        }

                        
                        outGridPls[yIdx][xIdx]->push_back(pl);
                    }
                }
			}			
		}
		catch (SPDProcessingException &e) 
		{			
            std::cout << e.what() << std::endl;
			throw e;
		}
    }
    
    void SPDGridData::reGridDataCylindrical(std::vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, std::vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double originX, double originY, float outBinSize) throw(SPDProcessingException)
    {
        throw SPDProcessingException("Cylindrical gridding not implemented yet... reGridDataCylindrical");
    }
    
    void SPDGridData::reGridDataPolar(std::vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, std::vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double originX, double originY, float outBinSize) throw(SPDProcessingException)
    {
        throw SPDProcessingException("Polar gridding not implemented yet... reGridDataPolar");
    }
        
    void SPDGridData::reGridDataScan(std::vector<SPDPulse*> ***inGridPls, boost::uint_fast32_t inXSize, boost::uint_fast32_t inYSize, std::vector<SPDPulse*> ***outGridPls, boost::uint_fast32_t outXSize, boost::uint_fast32_t outYSize, double originX, double originY, float outBinSize) throw(SPDProcessingException)
    {
        if((outXSize < 1) | (outYSize < 1))
		{
			throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
		}
        
        try 
		{	
			double xDiff = 0;
			double yDiff = 0;
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			
			SPDPulse *pl = NULL;
			for(boost::uint_fast64_t i = 0; i < inYSize; ++i)
			{
                for(boost::uint_fast64_t j = 0; j < inXSize; ++j)
                {
                    for(std::vector<SPDPulse*>::iterator iterPls = inGridPls[i][j]->begin(); iterPls != inGridPls[i][j]->end(); ++iterPls)
                    {
                        pl = (*iterPls);
				
                        xDiff = (pl->scanlineIdx - originX)/outBinSize;
                        yDiff = (pl->scanline - originY)/outBinSize;
                        
                        try 
                        {
                            xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
                            yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
                        }
                        catch(boost::numeric::negative_overflow& e) 
                        {
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            throw SPDProcessingException(e.what());
                        }
                        catch(boost::numeric::positive_overflow& e) 
                        {
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            throw SPDProcessingException(e.what());
                        }
                        catch(boost::numeric::bad_numeric_cast& e) 
                        {
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            throw SPDProcessingException(e.what());
                        }
                        
                        //std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                        
                        if(xIdx > ((outXSize)-1))
                        {
                            std::cout << "Point: [" << pl->scanlineIdx << "," << pl->scanline << "]\n";
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                            std::cout << "Size [" << outXSize << "," << outYSize << "]\n";
                            throw SPDProcessingException("Did not find scanlineIdx (x) index within range.");
                        }
                        
                        if(yIdx > ((outYSize)-1))
                        {
                            std::cout << "Point: [" << pl->scanlineIdx << "," << pl->scanline << "]\n";
                            std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                            std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                            std::cout << "Size [" << outXSize << "," << outYSize << "]\n";
                            throw SPDProcessingException("Did not find scanlineIdx (y) index within range.");
                        }
                        
                        outGridPls[yIdx][xIdx]->push_back(pl);
                    }
                }
			}			
		}
		catch (SPDProcessingException &e) 
		{			
            std::cout << e.what() << std::endl;
			throw e;
		}
    }
    
    
	SPDGridData::~SPDGridData()
	{
		
	}
}

