/*
 *  SPDGeneralASCIIFileWriter.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 24/01/2011.
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

#include "spd/SPDGeneralASCIIFileWriter.h"


namespace spdlib
{
	SPDGeneralASCIIFileWriter::SPDGeneralASCIIFileWriter() : SPDDataExporter("ASCII"), outASCIIFile(NULL), fileType(0)
	{
		
	}
	
	SPDGeneralASCIIFileWriter::SPDGeneralASCIIFileWriter(const SPDDataExporter &dataExporter) throw(SPDException) : SPDDataExporter(dataExporter), outASCIIFile(NULL), fileType(0)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
	}
	
	SPDGeneralASCIIFileWriter& SPDGeneralASCIIFileWriter::operator=(const SPDGeneralASCIIFileWriter& dataExporter) throw(SPDException)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
		
		this->spdFile = dataExporter.spdFile;
		this->outputFile = dataExporter.outputFile;
		return *this;
	}
    
    SPDDataExporter* SPDGeneralASCIIFileWriter::getInstance()
    {
        return new SPDGeneralASCIIFileWriter();
    }
	
	bool SPDGeneralASCIIFileWriter::open(SPDFile *spdFile, string outputFile) throw(SPDIOException)
	{
		outASCIIFile = new ofstream();
		outASCIIFile->open(outputFile.c_str(), ios::out | ios::trunc);
		
		if(!outASCIIFile->is_open())
		{
			fileOpened = false;
			
			string message = string("Could not open file ") + outputFile;
			throw SPDIOException(message);
		}
		fileOpened = true;
		
		(*outASCIIFile).precision(12);
		if((spdFile->getReceiveWaveformDefined() == SPD_TRUE) & (spdFile->getTransWaveformDefined() == SPD_TRUE))
		{
			(*outASCIIFile) << "#WAVEFORM\n";
			(*outASCIIFile) << "PULSE_ID PULSE_TIME ORIGIN_EASTINGS ORIGIN_NORTHINGS ORIGIN_ELEVATION AZIMUTH ZENITH RANGE_TO_RECIEVED TRANSMITTED(0)RECEIVED(1) NO_WAVEFORM_VALUES WAVEFORM\n";
			fileType = SPD_WAVEFORM_PT;
		}
		else if(spdFile->getDecomposedPtDefined() == SPD_TRUE)
		{
			(*outASCIIFile) << "#DECOMPOSED\n";
			(*outASCIIFile) << "PULSE_ID PULSE_TIME ORIGIN_EASTINGS ORIGIN_NORTHINGS ORIGIN_ELEVATION ORIGIN_HEIGHT SIGNAL_INTENSITY SIGNAL_WIDTH NO_OF_RETURNS RETURN_NO EASTINGS NORTHINGS ELEVATION HEIGHT RETURN_TIME RETURN_INTENSITY RETURN_WIDTH CLASSIFICATION\n";
			fileType = SPD_DECOMPOSED_PT;
		}
		else if(spdFile->getDiscretePtDefined() == SPD_TRUE)
		{
			(*outASCIIFile) << "#DISCRETE\n";
			(*outASCIIFile) << "PULSE_ID PULSE_TIME ORIGIN_EASTINGS ORIGIN_NORTHINGS ORIGIN_ELEVATION ORIGIN_HEIGHT NO_OF_RETURNS RETURN_NO EASTINGS NORTHINGS ELEVATION HEIGHT RETURN_TIME INTENSITY CLASSIFICATION\n";
			fileType = SPD_DISCRETE_PT;
		}
		else
		{
			throw SPDIOException("File type is not defined.");
		}
		
		return fileOpened;
	}
	
	void SPDGeneralASCIIFileWriter::writeDataColumn(list<SPDPulse*> *plsIn, boost::uint_fast32_t col, boost::uint_fast32_t row)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		
		if(!fileOpened)
		{
			throw SPDIOException("Output file not open, cannot write to the file.");
		}
		
		try 
		{
			Exception::dontPrint();
			
			vector<SPDPoint*>::iterator iterPts;
			list<SPDPulse*>::iterator iterInPls;
			for(iterInPls = plsIn->begin(); iterInPls != plsIn->end(); )
			{
				if(fileType == SPD_WAVEFORM_PT)
				{
					(*outASCIIFile) << (*iterInPls)->pulseID << " " << (*iterInPls)->gpsTime << " " << (*iterInPls)->x0 << " " << (*iterInPls)->y0 << " " << (*iterInPls)->z0 << " " << (*iterInPls)->azimuth << " " << (*iterInPls)->zenith << " " << (*iterInPls)->rangeToWaveformStart << " 0 " << (*iterInPls)->numOfTransmittedBins << " "; 
					for(boost::uint_fast16_t i = 0; i < (*iterInPls)->numOfTransmittedBins; ++i)
					{
						if(i == 0)
						{
							(*outASCIIFile) << (*iterInPls)->transmitted[i];
						}
						else 
						{
							(*outASCIIFile) << " " << (*iterInPls)->transmitted[i];
						}
					}
					(*outASCIIFile) << endl;
					
					(*outASCIIFile) << (*iterInPls)->pulseID << " " << (*iterInPls)->gpsTime << " " << (*iterInPls)->x0 << " " << (*iterInPls)->y0 << " " << (*iterInPls)->z0 << " " << (*iterInPls)->azimuth << " " << (*iterInPls)->zenith << " " << (*iterInPls)->rangeToWaveformStart << " 1 " << (*iterInPls)->numOfReceivedBins << " ";
					for(boost::uint_fast16_t i = 0; i < (*iterInPls)->numOfReceivedBins; ++i)
					{
						if(i == 0)
						{
							(*outASCIIFile) << (*iterInPls)->received[i];
						}
						else 
						{
							(*outASCIIFile) << " " << (*iterInPls)->received[i];
						}
					}
					(*outASCIIFile) << endl;
				}
				else if(fileType == SPD_DECOMPOSED_PT)
				{
					for(iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
					{
						(*outASCIIFile) << (*iterInPls)->pulseID << " " << (*iterInPls)->gpsTime << " " << (*iterInPls)->x0 << " " << (*iterInPls)->y0 << " " << (*iterInPls)->z0 << " " << (*iterInPls)->h0 << " " << (*iterInPls)->amplitudePulse << " " << (*iterInPls)->widthPulse << " " << (*iterInPls)->numberOfReturns << " " << (*iterPts)->returnID << " " << (*iterPts)->x << " " << (*iterPts)->y << " " << (*iterPts)->z << " " << (*iterPts)->height  << " " << (*iterPts)->gpsTime << " " << (*iterPts)->amplitudeReturn << " " << (*iterPts)->widthReturn << " " << (*iterPts)->classification << endl;
					}
				}
				else if(fileType == SPD_DISCRETE_PT)
				{
					for(iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
					{
						(*outASCIIFile) << (*iterInPls)->pulseID << " " << (*iterInPls)->gpsTime << " " << (*iterInPls)->x0 << " " << (*iterInPls)->y0 << " " << (*iterInPls)->z0 << " " << (*iterInPls)->h0 << " " << (*iterInPls)->numberOfReturns << " " << (*iterPts)->returnID << " " << (*iterPts)->x << " " << (*iterPts)->y << " " << (*iterPts)->z  << " " << (*iterPts)->height  << " " << (*iterPts)->gpsTime  << " " << (*iterPts)->amplitudeReturn << " " << (*iterPts)->classification << endl;
					}
				}
				else
				{
					throw SPDIOException("File type is not defined.");
				}
				SPDPulseUtils::deleteSPDPulse(*iterInPls);
				iterInPls = plsIn->erase(iterInPls);
			}
			
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
		
	}
	
	void SPDGeneralASCIIFileWriter::writeDataColumn(vector<SPDPulse*> *plsIn, boost::uint_fast32_t col, boost::uint_fast32_t row)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		
		if(!fileOpened)
		{
			throw SPDIOException("Output file not open, cannot write to the file.");
		}
		
		try 
		{
			Exception::dontPrint();
			
			vector<SPDPoint*>::iterator iterPts;
			vector<SPDPulse*>::iterator iterInPls;
			for(iterInPls = plsIn->begin(); iterInPls != plsIn->end(); )
			{
				if(fileType == SPD_WAVEFORM_PT)
				{
					(*outASCIIFile) << (*iterInPls)->pulseID << " " << (*iterInPls)->gpsTime << " " << (*iterInPls)->x0 << " " << (*iterInPls)->y0 << " " << (*iterInPls)->z0 << " " << (*iterInPls)->azimuth << " " << (*iterInPls)->zenith << " " << (*iterInPls)->rangeToWaveformStart << " 0 " << (*iterInPls)->numOfTransmittedBins << " "; 
					for(boost::uint_fast16_t i = 0; i < (*iterInPls)->numOfTransmittedBins; ++i)
					{
						if(i == 0)
						{
							(*outASCIIFile) << (*iterInPls)->transmitted[i];
						}
						else 
						{
							(*outASCIIFile) << " " << (*iterInPls)->transmitted[i];
						}
					}
					(*outASCIIFile) << endl;
					
					(*outASCIIFile) << (*iterInPls)->pulseID << " " << (*iterInPls)->gpsTime << " " << (*iterInPls)->x0 << " " << (*iterInPls)->y0 << " " << (*iterInPls)->z0 << " " << (*iterInPls)->azimuth << " " << (*iterInPls)->zenith << " " << (*iterInPls)->rangeToWaveformStart << " 1 " << (*iterInPls)->numOfReceivedBins << " ";
					for(boost::uint_fast16_t i = 0; i < (*iterInPls)->numOfReceivedBins; ++i)
					{
						if(i == 0)
						{
							(*outASCIIFile) << (*iterInPls)->received[i];
						}
						else 
						{
							(*outASCIIFile) << " " << (*iterInPls)->received[i];
						}
					}
					(*outASCIIFile) << endl;
				}
				else if(fileType == SPD_DECOMPOSED_PT)
				{
					for(iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
					{
						(*outASCIIFile) << (*iterInPls)->pulseID << " " << (*iterInPls)->gpsTime << " " << (*iterInPls)->x0 << " " << (*iterInPls)->y0 << " " << (*iterInPls)->z0 << " " << (*iterInPls)->amplitudePulse << " " << (*iterInPls)->widthPulse << " " << (*iterInPls)->numberOfReturns << " " << (*iterPts)->returnID << " " << (*iterPts)->x << " " << (*iterPts)->y << " " << (*iterPts)->z << " " << (*iterPts)->gpsTime << " " << (*iterPts)->amplitudeReturn << " " << (*iterPts)->widthReturn << " " << (*iterPts)->classification << endl;
					}
				}
				else if(fileType == SPD_DISCRETE_PT)
				{
					for(iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
					{
						(*outASCIIFile) << (*iterInPls)->pulseID << " " << (*iterInPls)->gpsTime << " " << (*iterInPls)->x0 << " " << (*iterInPls)->y0 << " " << (*iterInPls)->z0 << " " << (*iterInPls)->numberOfReturns << " " << (*iterPts)->returnID << " " << (*iterPts)->x << " " << (*iterPts)->y << " " << (*iterPts)->z  << " " << (*iterPts)->gpsTime  << " " << (*iterPts)->amplitudeReturn << " " << (*iterPts)->classification << endl;
					}
				}
				else
				{
					throw SPDIOException("File type is not defined.");
				}
				SPDPulseUtils::deleteSPDPulse(*iterInPls);
				iterInPls = plsIn->erase(iterInPls);
			}
			
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
		
	}
	
	void SPDGeneralASCIIFileWriter::finaliseClose() throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Output file not open, cannot finalise.");
		}
		(*outASCIIFile).flush();
		(*outASCIIFile).close();
	}
	
	bool SPDGeneralASCIIFileWriter::requireGrid()
	{
		return false;
	}
	
	bool SPDGeneralASCIIFileWriter::needNumOutPts()
	{
		return false;
	}
	
	SPDGeneralASCIIFileWriter::~SPDGeneralASCIIFileWriter()
	{
		if(fileOpened)
		{
			try 
			{
				this->finaliseClose();
			}
			catch (SPDIOException &e) 
			{
				cerr << "WARNING: " << e.what() << endl;
			}
		}
	}
}


