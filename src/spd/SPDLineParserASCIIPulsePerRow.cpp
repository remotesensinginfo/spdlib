/*
 *  SPDLineParserASCIIPulsePerRow.cpp
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

#include "spd/SPDLineParserASCIIPulsePerRow.h"

namespace spdlib
{

	SPDLineParserASCIIPulsePerRow::SPDLineParserASCIIPulsePerRow() : SPDTextLineProcessor()
	{
		headerRead = true;
	}
	
	bool SPDLineParserASCIIPulsePerRow::haveReadheader()
	{
		return headerRead;
	}
	
	void SPDLineParserASCIIPulsePerRow::parseHeader(string) throw(SPDIOException)
	{
		
	}
	
	bool SPDLineParserASCIIPulsePerRow::parseLine(string line, SPDPulse *pl, boost::uint_fast16_t indexCoords)throw(SPDIOException)
	{
		SPDTextFileUtilities textUtils;
		SPDPointUtils ptUtils;
		SPDPoint *pt = NULL;
		
		if((!textUtils.blankline(line)) & (!textUtils.lineStart(line, '#')))
		{
			vector<string> *tokens = new vector<string>();
			textUtils.tokenizeString(line, ' ', tokens, true);
			if(((tokens->size()-1) % 4) == 0)
			{
				boost::uint_fast16_t numOfReturns = (tokens->size()-1)/4;
				pl->numberOfReturns = numOfReturns;

				double gpsTime = textUtils.strtodouble(tokens->at(0));
				
				for(boost::uint_fast16_t n = 0; n < numOfReturns; ++n)
				{
					pt = new SPDPoint();
					ptUtils.initSPDPoint(pt);
					
					pt->x = textUtils.strtodouble(tokens->at((1 + (n * 4))));
					//cout << "tokens->at(" << (1 + (n * 4)) << ") = " << tokens->at((1 + (n * 4))) << endl;
					pt->y = textUtils.strtodouble(tokens->at((1 + (n * 4)) + 1));
					//cout << "tokens->at(" << (1 + (n * 4)) + 1 << ") = " << tokens->at((1 + (n * 4)) + 1) << endl;
					pt->z = textUtils.strtofloat(tokens->at((1 + (n * 4)) + 2));
					//cout << "tokens->at(" << (1 + (n * 4)) + 2 << ") = " << tokens->at((1 + (n * 4)) + 2) << endl;
					pt->amplitudeReturn = textUtils.strtofloat(tokens->at((1 + (n * 4)) + 3));
					//cout << "tokens->at(" << (1 + (n * 4)) + 3 << ") = " << tokens->at((1 + (n * 4)) + 3) << endl << endl;
					pt->gpsTime = gpsTime;
					pt->returnID = (n+1);
					
					pl->pts->push_back(pt);
				}
				
				if(indexCoords == SPD_FIRST_RETURN)
				{
					pl->xIdx = pl->pts->front()->x;
					pl->yIdx = pl->pts->front()->y;
				}
				else if(indexCoords == SPD_LAST_RETURN)
				{
					pl->xIdx = pl->pts->back()->x;
					pl->yIdx = pl->pts->back()->y;
				}
				else if(indexCoords == SPD_MAX_INTENSITY)
				{
					unsigned int maxIdx = 0;
					double maxVal = 0;
					bool first = true;
					for(unsigned int i = 0; i < pl->pts->size(); ++i)
					{
						if(first)
						{
							maxIdx = i;
							maxVal = pl->pts->at(i)->amplitudeReturn;
							first = false;
						}
						else if(pl->pts->at(i)->amplitudeReturn > maxVal)
						{
							maxIdx = i;
							maxVal = pl->pts->at(i)->amplitudeReturn;
						}
					}
					
					pl->xIdx = pl->pts->at(maxIdx)->x;
					pl->yIdx = pl->pts->at(maxIdx)->y;
				}
				else
				{
					throw SPDIOException("Indexing type unsupported");
				}
			}
			tokens->clear();
			delete tokens;
			return true;
		}
		return false;
	}
	
	bool SPDLineParserASCIIPulsePerRow::isFileType(string fileType)
	{
		if(fileType == "ASCIIPULSEROW")
		{
			return true;
		}
		return false;
	}
	
	void SPDLineParserASCIIPulsePerRow::saveHeaderValues(SPDFile *spdFile)
	{
		spdFile->setDiscretePtDefined(SPD_TRUE);
		spdFile->setDecomposedPtDefined(SPD_FALSE);
		spdFile->setTransWaveformDefined(SPD_FALSE);
        spdFile->setReceiveWaveformDefined(SPD_FALSE);
		spdFile->setRGBDefined(SPD_FALSE);
	}
	
	void SPDLineParserASCIIPulsePerRow::reset()
	{
		
	}
	
	SPDLineParserASCIIPulsePerRow::~SPDLineParserASCIIPulsePerRow()
	{
		
	}
	
}

