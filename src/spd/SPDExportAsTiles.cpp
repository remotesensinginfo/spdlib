/*
 *  SPDExportAsTiles.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 04/12/2010.
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

#include "spd/SPDExportAsTiles.h"

namespace spdlib
{
	

	SPDExportAsRowTiles::SPDExportAsRowTiles(PointDataTileFile *tiles, boost::uint_fast32_t numOfTiles, SPDFile *overallSPD, double tileHeight, bool useSphericIdx, bool useScanIdx) throw(SPDException): SPDImporterProcessor(), tiles(NULL), numOfTiles(0), overallSPD(NULL), tileHeight(0), useSphericIdx(false), useScanIdx(false), filesOpen(false)
	{
		this->tiles = tiles;
		this->numOfTiles = numOfTiles;
		this->overallSPD = overallSPD;
		this->tileHeight = tileHeight;
        this->useSphericIdx = useSphericIdx;
		this->useScanIdx = useScanIdx;
        
		for(boost::uint_fast32_t i = 0; i < this->numOfTiles; ++i)
		{
			this->tiles[i].exporter->open(this->tiles[i].spdFile, this->tiles[i].spdFile->getFilePath());
		}
		filesOpen = true;
	}
	
	void SPDExportAsRowTiles::processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException)
	{
        double yDiff = 0.0;
        if(useSphericIdx)
        {          
            yDiff = pulse->zenith - overallSPD->getZenithMin();
        }
        else if(useScanIdx)
        {          
            yDiff = pulse->scanline - overallSPD->getScanlineMin();
        }
        else
        {
            yDiff = overallSPD->getYMax() - pulse->yIdx;
        }
		
        boost::uint_fast32_t tileIdx = 0;
		try 
		{
            tileIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff/tileHeight);
		}
		catch(boost::numeric::negative_overflow& e) 
		{
			throw SPDIOException(e.what());
		}
		catch(boost::numeric::positive_overflow& e) 
		{
			throw SPDIOException(e.what());
		}
		catch(boost::numeric::bad_numeric_cast& e) 
		{
			throw SPDIOException(e.what());
		}
		
		if(tileIdx >= numOfTiles)
		{
			std::cerr.precision(15);
			std::cerr << "Array Index is greater equal than = " << tileIdx << " of "<< numOfTiles << std::endl;
			std::cerr << "Pulse: " << pulse << std::endl;
		}
		
		try 
		{
			this->tiles[tileIdx].pulses->push_back(pulse);
			this->tiles[tileIdx].exporter->writeDataColumn(this->tiles[tileIdx].pulses, 0, 0);
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
	}
	
	void SPDExportAsRowTiles::completeFileAndClose()throw(SPDIOException)
	{
		try 
		{
			std::list<SPDPulse*>::iterator iterPulses;
			for(boost::uint_fast32_t i = 0; i < this->numOfTiles; ++i)
			{
				this->tiles[i].exporter->finaliseClose();
			}
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
	}

	SPDExportAsRowTiles::~SPDExportAsRowTiles()
	{
		
	}
    
    
    SPDExportAsBlockTiles::SPDExportAsBlockTiles(PointDataTileFile *tiles, boost::uint_fast32_t numOfTiles, boost::uint_fast32_t numOfXTiles, boost::uint_fast32_t numOfYTiles, SPDFile *overallSPD, double tileHeight, double tileWidth, bool useSphericIdx, bool useScanIdx) throw(SPDException)
    {
        this->tiles = tiles;
		this->numOfTiles = numOfTiles;
        this->numOfYTiles = numOfYTiles;
        this->numOfXTiles = numOfXTiles;
		this->overallSPD = overallSPD;
		this->tileHeight = tileHeight;
        this->tileWidth = tileWidth;
        this->useSphericIdx = useSphericIdx;
        this->useScanIdx = useScanIdx;
        
		for(boost::uint_fast32_t i = 0; i < this->numOfTiles; ++i)
		{
			this->tiles[i].exporter->open(this->tiles[i].spdFile, this->tiles[i].spdFile->getFilePath());
		}
		filesOpen = true;
    }
    
    void SPDExportAsBlockTiles::processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException)
    {
        double yDiff = 0.0;
        double xDiff = 0.0;
        if(useSphericIdx)
        {          
            yDiff = pulse->zenith - overallSPD->getZenithMin();
            xDiff = pulse->azimuth - overallSPD->getAzimuthMin();
        }
        else if(useScanIdx)
        {          
            yDiff = pulse->scanline - overallSPD->getScanlineMin();
            xDiff = pulse->scanlineIdx - overallSPD->getScanlineIdxMin();
        }
        else
        {
            yDiff = overallSPD->getYMax() - pulse->yIdx;
            xDiff = pulse->xIdx - overallSPD->getXMin();
        }
		
        boost::uint_fast32_t tileXIdx = 0;
        boost::uint_fast32_t tileYIdx = 0;
		try 
		{
            tileYIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff/tileHeight);
            tileXIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff/tileWidth);
		}
		catch(boost::numeric::negative_overflow& e) 
		{
			throw SPDIOException(e.what());
		}
		catch(boost::numeric::positive_overflow& e) 
		{
			throw SPDIOException(e.what());
		}
		catch(boost::numeric::bad_numeric_cast& e) 
		{
			throw SPDIOException(e.what());
		}
		
		if(tileYIdx >= numOfYTiles)
		{
			std::cerr.precision(15);
			std::cerr << "Array Index is greater equal than = " << tileYIdx << " of "<< numOfYTiles << std::endl;
			std::cerr << "Pulse: " << pulse << std::endl;
		}
        
        if(tileXIdx >= numOfXTiles)
		{
			std::cerr.precision(15);
			std::cerr << "Array Index is greater equal than = " << tileXIdx << " of "<< numOfXTiles << std::endl;
			std::cerr << "Pulse: " << pulse << std::endl;
		}
		
		try 
		{
            boost::uint_fast32_t tileIdx = (tileYIdx * numOfXTiles) + tileXIdx;
			this->tiles[tileIdx].pulses->push_back(pulse);
			this->tiles[tileIdx].exporter->writeDataColumn(this->tiles[tileIdx].pulses, 0, 0);
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
    }
    
    void SPDExportAsBlockTiles::completeFileAndClose()throw(SPDIOException)
    {
        try 
		{
			std::list<SPDPulse*>::iterator iterPulses;
			for(boost::uint_fast32_t i = 0; i < this->numOfTiles; ++i)
			{
				this->tiles[i].exporter->finaliseClose();
			}
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
    }
    
    SPDExportAsBlockTiles::~SPDExportAsBlockTiles()
    {
        
    }
    
    
}



