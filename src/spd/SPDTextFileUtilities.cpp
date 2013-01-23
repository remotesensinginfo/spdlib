/*
 *  SPDTextFileUtilities.cpp
 *  spdlib_prj
 *
 *  Created by Pete Bunting on 09/10/2009.
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

#include "spd/SPDTextFileUtilities.h"

namespace spdlib{
	
	SPDTextFileUtilities::SPDTextFileUtilities()
	{
		
	}
	
	boost::uint_fast64_t SPDTextFileUtilities::countLines(std::string input) throw(SPDIOException)
	{
		boost::uint_fast64_t count = 0;
		std::string strLine;
		std::ifstream inputFile;
		inputFile.open(input.c_str(), std::ios_base::in);
		if(inputFile.is_open())
		{
			char ch = ' ';
			char lastch = ' ';
			inputFile.get(ch);
			while (!inputFile.eof()) 
			{
				if ((ch == 0x0a) && (lastch == 0x0d))
				{
					++count; // Windows Line Ending
				}
				else if ((lastch == 0x0d) && (ch != 0x0a)) 
				{
					++count; // Mac Line Ending
				} 
				else if (ch == 0x0a) 
				{
					++count; // UNIX Line Ending
				}
				lastch = ch;
				inputFile.get(ch);      
			}
			
			inputFile.close();
		}
		else
		{
			std::string message = std::string("Text file ") + input + std::string(" could not be openned.");
			throw SPDIOException(message.c_str());
		}
		
		return count;
	}
	
	bool SPDTextFileUtilities::lineStart(std::string line, char token)
	{
		int lineLength = line.length();
		for(int i = 0; i < lineLength; i++)
		{
			if((line.at(i) == ' ') | (line.at(i) == '\t') | (line.at(i) == '\n'))
			{
				// spaces and tabs at the beginning of a line can be ignored.
			}
			else if(line.at(i) == token)
			{
				return true;
			}
			else
			{
				return false;
			}
		}
		return false;
	}
	
	bool SPDTextFileUtilities::blankline(std::string line)
	{
		int lineLength = line.length();
		if(lineLength < 1)
		{
			return true;
		}
		else
		{
			for(int i = 0; i < lineLength; i++)
			{
				if((line.at(i) == ' ') | (line.at(i) == '\t') | (line.at(i) == '\n'))
				{
					// spaces and tabs at the beginning of a line can be ignored.
				}
				else
				{
					return false;
				}
			}
		}
		return true;
	}
	
	std::string SPDTextFileUtilities::removeWhiteSpace(std::string line)
	{
		int lineLength = line.length();
		int firstChar = 0;
		int lastChar = 0;
		for(int i = 0; i < lineLength; i++)
		{
			if((line.at(i) != ' ') | (line.at(i) != '\t') | (line.at(i) != '\n'))
			{
				firstChar = i;
				break;
			}
		}
		
		for(int i = (lineLength-1); i >= 0; --i)
		{
			if((line.at(i) != ' ') | (line.at(i) != '\t') | (line.at(i) != '\n'))
			{
				lastChar = i;
				break;
			}
		}
		
		return line.substr(firstChar, lastChar-firstChar);
	}
	
	void SPDTextFileUtilities::tokenizeString(std::string line, char token, std::vector<std::string> *tokens, bool ignoreDuplicateTokens)
	{
		std::string word;
		int start = 0;
		int lineLength = line.length();
		for(int i = 0; i < lineLength; i++)
		{
			if(line.at(i) == token)
			{
				word = line.substr(start, i-start);								
				if(ignoreDuplicateTokens)
				{
					if((word.length() > 0) & (word != "\n"))
					{
						tokens->push_back(word);
					}
				}
				else if((word.length() > 0) & (word != "\n"))
				{
					tokens->push_back(word);
				}
				
				start = start + i-start+1;
			}
		}
		word = line.substr(start);
		if((word.length() > 0) & (word != "\n"))
		{
			tokens->push_back(word);
		}		
	}
	
	std::string SPDTextFileUtilities::readFileToString(std::string input) throw(SPDIOException)
	{
		std::string wholeFile = "";
		std::ifstream inputFileStream;
		inputFileStream.open(input.c_str(), std::ios_base::in);
		if(!inputFileStream.is_open())
		{
			throw SPDIOException("File could not be opened.");
		}
		
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
				boost::algorithm::trim(strLine);
				wholeFile += strLine;
				strLine = "";
			}
			else 
			{
				strLine += ch;
			}
			
			lastch = ch;
			inputFileStream.get(ch);      
		}
		wholeFile += strLine;
		inputFileStream.close();
		
		return wholeFile;
	}
	
	
	double SPDTextFileUtilities::strtodouble(std::string inValue) throw(SPDTextFileException)
	{
		double outValue = 0;
		try
        {
            boost::algorithm::trim(inValue);
            outValue = boost::lexical_cast<double>(inValue);
        }
        catch(boost::bad_lexical_cast &e)
        {
			std::string message = std::string("Trying to convert \"") + inValue + std::string("\" - ") + std::string(e.what());
            throw SPDTextFileException(message);
        }
		return outValue;
	}
	
	float SPDTextFileUtilities::strtofloat(std::string inValue) throw(SPDTextFileException)
	{
		float outValue = 0;
		try
        {
            boost::algorithm::trim(inValue);
            outValue = boost::lexical_cast<float>(inValue);
        }
        catch(boost::bad_lexical_cast &e)
        {
            std::string message = std::string("Trying to convert \"") + inValue + std::string("\" - ") + std::string(e.what());
            throw SPDTextFileException(message);
        }
		return outValue;
	}
	
	boost::uint_fast8_t SPDTextFileUtilities::strto8bitUInt(std::string inValue) throw(SPDTextFileException)
	{
		boost::uint_fast8_t outValue = 0;
		try
        {
            boost::algorithm::trim(inValue);
            outValue = boost::lexical_cast<boost::uint_fast8_t>(inValue);
        }
        catch(boost::bad_lexical_cast &e)
        {
            std::string message = std::string("Trying to convert \"") + inValue + std::string("\" - ") + std::string(e.what());
            throw SPDTextFileException(message);
        }
		return outValue;
	}
	
	boost::uint_fast16_t SPDTextFileUtilities::strto16bitUInt(std::string inValue) throw(SPDTextFileException)
	{
		boost::uint_fast16_t outValue = 0;
		try
        {
            boost::algorithm::trim(inValue);
            outValue = boost::lexical_cast<boost::uint_fast16_t>(inValue);
        }
        catch(boost::bad_lexical_cast &e)
        {
            std::string message = std::string("Trying to convert \"") + inValue + std::string("\" - ") + std::string(e.what());
            throw SPDTextFileException(message);
        }
		return outValue;
	}
	
	boost::uint_fast32_t SPDTextFileUtilities::strto32bitUInt(std::string inValue) throw(SPDTextFileException)
	{
		boost::uint_fast32_t outValue = 0;
		try
        {
            boost::algorithm::trim(inValue);
            outValue = boost::lexical_cast<boost::uint_fast32_t>(inValue);
        }
        catch(boost::bad_lexical_cast &e)
        {
            std::string message = std::string("Trying to convert \"") + inValue + std::string("\" - ") + std::string(e.what());
            throw SPDTextFileException(message);
        }
		return outValue;
	}
	
	boost::uint_fast64_t SPDTextFileUtilities::strto64bitUInt(std::string inValue) throw(SPDTextFileException)
	{
		boost::uint_fast64_t outValue = 0;
		try
        {
            boost::algorithm::trim(inValue);
            outValue = boost::lexical_cast<boost::uint_fast64_t>(inValue);
        }
        catch(boost::bad_lexical_cast &e)
        {
            std::string message = std::string("Trying to convert \"") + inValue + std::string("\" - ") + std::string(e.what());
            throw SPDTextFileException(message);
        }
		return outValue;
	}
	
    boost::int_fast8_t SPDTextFileUtilities::strto8bitInt(std::string inValue) throw(SPDTextFileException)
	{
        boost::int_fast8_t outValue = 0;
		try
        {
            boost::algorithm::trim(inValue);
            outValue = boost::lexical_cast<int_fast8_t>(inValue);
        }
        catch(boost::bad_lexical_cast &e)
        {
            std::string message = std::string("Trying to convert \"") + inValue + std::string("\" - ") + std::string(e.what());
            throw SPDTextFileException(message);
        }
		return outValue;
	}
	
    boost::int_fast16_t SPDTextFileUtilities::strto16bitInt(std::string inValue) throw(SPDTextFileException)
	{
        boost::int_fast16_t outValue = 0;
		try
        {
            boost::algorithm::trim(inValue);
            outValue = boost::lexical_cast<int_fast16_t>(inValue);
        }
        catch(boost::bad_lexical_cast &e)
        {
            std::string message = std::string("Trying to convert \"") + inValue + std::string("\" - ") + std::string(e.what());
            throw SPDTextFileException(message);
        }
		return outValue;
	}
	
    boost::int_fast32_t SPDTextFileUtilities::strto32bitInt(std::string inValue) throw(SPDTextFileException)
	{
        boost::int_fast32_t outValue = 0;
		try
        {
            boost::algorithm::trim(inValue);
            outValue = boost::lexical_cast<int_fast32_t>(inValue);
        }
        catch(boost::bad_lexical_cast &e)
        {
            std::string message = std::string("Trying to convert \"") + inValue + std::string("\" - ") + std::string(e.what());
            throw SPDTextFileException(message);
        }
		return outValue;
	}
	
    boost::int_fast64_t SPDTextFileUtilities::strto64bitInt(std::string inValue) throw(SPDTextFileException)
	{
        boost::int_fast64_t outValue = 0;
		try
        {
            boost::algorithm::trim(inValue);
            outValue = boost::lexical_cast<int_fast64_t>(inValue);
        }
        catch(boost::bad_lexical_cast &e)
        {
            std::string message = std::string("Trying to convert \"") + inValue + std::string("\" - ") + std::string(e.what());
            throw SPDTextFileException(message);
        }
		return outValue;
	}
	
	std::string SPDTextFileUtilities::doubletostring(double number) throw(SPDTextFileException)
	{
		std::string outValue = "";
		try
        {
            outValue = boost::lexical_cast<std::string>(number);
        }
        catch(boost::bad_lexical_cast &e)
        {
            throw SPDTextFileException(e.what());
        }
		return outValue;
	}
	
	std::string SPDTextFileUtilities::floattostring(float number) throw(SPDTextFileException)
	{
		std::string outValue = "";
		try
        {
            outValue = boost::lexical_cast<std::string>(number);
        }
        catch(boost::bad_lexical_cast &e)
        {
            throw SPDTextFileException(e.what());
        }
		return outValue;
	}
	
	std::string SPDTextFileUtilities::uInt8bittostring(boost::uint_fast8_t number) throw(SPDTextFileException)
	{
		std::string outValue = "";
		try
        {
            outValue = boost::lexical_cast<std::string>(number);
        }
        catch(boost::bad_lexical_cast &e)
        {
            throw SPDTextFileException(e.what());
        }
		return outValue;
	}
	
	std::string SPDTextFileUtilities::uInt16bittostring(boost::uint_fast16_t number) throw(SPDTextFileException)
	{
		std::string outValue = "";
		try
        {
            outValue = boost::lexical_cast<std::string>(number);
        }
        catch(boost::bad_lexical_cast &e)
        {
            throw SPDTextFileException(e.what());
        }
		return outValue;
	}
	
	std::string SPDTextFileUtilities::uInt32bittostring(boost::uint_fast32_t number) throw(SPDTextFileException)
	{
		std::string outValue = "";
		try
        {
            outValue = boost::lexical_cast<std::string>(number);
        }
        catch(boost::bad_lexical_cast &e)
        {
            throw SPDTextFileException(e.what());
        }
		return outValue;
	}
	
	std::string SPDTextFileUtilities::uInt64bittostring(boost::uint_fast64_t number) throw(SPDTextFileException)
	{
		std::string outValue = "";
		try
        {
            outValue = boost::lexical_cast<std::string>(number);
        }
        catch(boost::bad_lexical_cast &e)
        {
            throw SPDTextFileException(e.what());
        }
		return outValue;
	}
	
	std::string SPDTextFileUtilities::int8bittostring(boost::int_fast8_t number) throw(SPDTextFileException)
	{
		std::string outValue = "";
		try
        {
            outValue = boost::lexical_cast<std::string>(number);
        }
        catch(boost::bad_lexical_cast &e)
        {
            throw SPDTextFileException(e.what());
        }
		return outValue;
	}
	
	std::string SPDTextFileUtilities::int16bittostring(boost::int_fast16_t number) throw(SPDTextFileException)
	{
		std::string outValue = "";
		try
        {
            outValue = boost::lexical_cast<std::string>(number);
        }
        catch(boost::bad_lexical_cast &e)
        {
            throw SPDTextFileException(e.what());
        }
		return outValue;
	}
	
	std::string SPDTextFileUtilities::int32bittostring(boost::int_fast32_t number) throw(SPDTextFileException)
	{
		std::string outValue = "";
		try
        {
            outValue = boost::lexical_cast<std::string>(number);
        }
        catch(boost::bad_lexical_cast &e)
        {
            throw SPDTextFileException(e.what());
        }
		return outValue;
	}
	
	std::string SPDTextFileUtilities::int64bittostring(boost::int_fast64_t number) throw(SPDTextFileException)
	{
		std::string outValue = "";
		try
        {
            outValue = boost::lexical_cast<std::string>(number);
        }
        catch(boost::bad_lexical_cast &e)
        {
            throw SPDTextFileException(e.what());
        }
		return outValue;
	}
	
	SPDTextFileUtilities::~SPDTextFileUtilities()
	{
		
	}
	
}




