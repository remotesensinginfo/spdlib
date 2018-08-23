/*
 *  SPDTextFileLineReader.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 01/12/2010.
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

#include "spd/SPDTextFileLineReader.h"

namespace spdlib
{

	SPDTextFileLineReader::SPDTextFileLineReader(): inputFileStream(), fileOpened(false)
	{
		
	}
	
	void SPDTextFileLineReader::openFile(std::string filepath)throw(SPDIOException)
	{
		inputFileStream.open(filepath.c_str(), std::ios_base::in);
		if(!inputFileStream.is_open())
		{
			throw SPDIOException("File could not be opened.");
		}
		fileOpened = true;
	}
	
	bool SPDTextFileLineReader::endOfFile()
	{
		if(fileOpened)
		{
			return inputFileStream.eof();
		}
		return true;
	}
	
	std::string SPDTextFileLineReader::readLine()throw(SPDIOException)
	{
		std::string strLine = "";
		bool lineEnding = false;
		char ch = ' ';
		char lastch = ' ';
		inputFileStream.get(ch);
		while (!inputFileStream.eof()) 
		{					
			if ((ch == 0x0a) && (lastch == 0x0d))
			{
				lineEnding = true; // Windows Line Ending
			}
			else if ((lastch == 0x0d) && (ch != 0x0a)) 
			{
				lineEnding = true; // Mac Line Ending
			} 
			else if (ch == 0x0a) 
			{
				lineEnding = true; // UNIX Line Ending
			}
			
			if(lineEnding)
			{
				break;
			}
			else 
			{
				strLine += ch;
			}
			
			lastch = ch;
			inputFileStream.get(ch);      
		}
		
		boost::algorithm::trim(strLine);
		
		return strLine;
	}
	
	void SPDTextFileLineReader::closeFile()throw(SPDIOException)
	{
		inputFileStream.close();
		fileOpened = false;
	}
	
	SPDTextFileLineReader::~SPDTextFileLineReader()
	{
		
	}

}



