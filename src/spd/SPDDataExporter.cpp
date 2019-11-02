/*
 *  SPDDataExporter.cpp
 *  spdlib_prj
 *
 *  Created by Pete Bunting on 28/09/2009.
 *  Copyright 2009 SPDLib. All rights reserved.
 *
 *  This file is part of SPDLib.
 *
 *  Permission is hereby granted, free of charge, to any person 
 *  obtaining a copy of this software and associated documentation 
 *  files (the "Software"), to deal in the Software without restriction, 
 *  including without limitation the rights to use, copy, modify, 
 *  merge, publish, distribute, sublicense, and/or sell copies of the 
 *  Software, and to permit persons to whom the Software is furnished 
 *  to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be 
 *  included in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 *  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
 *  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR 
 *  ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
 *  CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
 *  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "spd/SPDDataExporter.h"

namespace spdlib
{
	
	SPDDataExporter::SPDDataExporter(std::string filetype) : spdFile(NULL), outputFile(""), fileOpened(false), filetype(""), numOutPts(0), numOutPtsDefined(false)
	{
		this->filetype = filetype;
	}
	
	SPDDataExporter::SPDDataExporter(const SPDDataExporter &dataExporter) : spdFile(NULL), outputFile(""), fileOpened(false), filetype(""), numOutPts(0), numOutPtsDefined(false) 
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
		
		this->spdFile = dataExporter.spdFile;
		this->outputFile = dataExporter.outputFile;
		this->fileOpened = false;
	}
	
	SPDDataExporter& SPDDataExporter::operator=(const SPDDataExporter& dataExporter) 
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}	
		
		this->spdFile = dataExporter.spdFile;
		this->outputFile = dataExporter.outputFile;
		this->fileOpened = false;
		return *this;
	}
	
	void SPDDataExporter::writeData(std::list<SPDPulse*> ***griddedPls, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize)
	{
		if(!fileOpened)
		{
			throw SPDIOException("File has not been opened.");
		}
		try 
		{
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					this->writeDataColumn(griddedPls[i][j], j, i);
				}
			}
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
	}
	
	void SPDDataExporter::writeData(std::vector<SPDPulse*> ***griddedPls, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize)
	{
		if(!fileOpened)
		{
			throw SPDIOException("File has not been opened.");
		}
		try 
		{
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					this->writeDataColumn(griddedPls[i][j], j, i);
				}
			}
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
	}
    
    void SPDDataExporter::writeData(std::list<SPDPulse*> ***griddedPls, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t startBinX, boost::uint_fast32_t startBinY, boost::uint_fast32_t startIdxX, boost::uint_fast32_t startIdxY)
    {
        if(!fileOpened)
		{
			throw SPDIOException("File has not been opened.");
		}
		try 
		{
            boost::uint_fast32_t endBinX = startBinX + xSize;
            boost::uint_fast32_t endBinY = startBinY + ySize;
            
            boost::uint_fast32_t x = 0;
            boost::uint_fast32_t y = 0;
            
			for(boost::uint_fast32_t i = startBinY; i < endBinY; ++i)
			{
                x = 0;
				for(boost::uint_fast32_t j = startBinX; j < endBinX; ++j)
				{
					this->writeDataColumn(griddedPls[i][j], (startIdxX+x), (startIdxY+y));
                    ++x;
				}
                ++y;
			}
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
    }
    
    void SPDDataExporter::writeData(std::vector<SPDPulse*> ***griddedPls, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t startBinX, boost::uint_fast32_t startBinY, boost::uint_fast32_t startIdxX, boost::uint_fast32_t startIdxY)
    {
        if(!fileOpened)
		{
			throw SPDIOException("File has not been opened.");
		}
		try 
		{
            boost::uint_fast32_t endBinX = startBinX + xSize;
            boost::uint_fast32_t endBinY = startBinY + ySize;
            
            boost::uint_fast32_t x = 0;
            boost::uint_fast32_t y = 0;
            
			for(boost::uint_fast32_t i = startBinY; i < endBinY; ++i)
			{
                x = 0;
				for(boost::uint_fast32_t j = startBinX; j < endBinX; ++j)
				{
					this->writeDataColumn(griddedPls[i][j], (startIdxX+x), (startIdxY+y));
                    ++x;
				}
                ++y;
			}
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
    }
	
	bool SPDDataExporter::isFileType(std::string filetype)
	{
		if(this->filetype == filetype)
		{
			return true;
		}
		return false;
	}
	
	void SPDDataExporter::setNumOutPts(boost::uint_fast64_t numOutPts)
	{
		this->numOutPts = numOutPts;
		this->numOutPtsDefined = true;
	}
   
	bool SPDDataExporter::opened()
    {
		return fileOpened;
    }
		   
		   
	SPDDataExporter::~SPDDataExporter()
	{
		
	}
}


