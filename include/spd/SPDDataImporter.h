/*
 *  SPDDataImporter.h
 *  spdlib_prj
 *
 *  Created by Pete Bunting on 15/09/2009.
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
 */

#ifndef SPDDataImporter_H
#define SPDDataImporter_H

#include <list>
#include <string>

#include <boost/cstdint.hpp>

#include "ogrsf_frmts.h"
#include "ogr_spatialref.h"

#include "spd/SPDCommon.h"
#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"

//#include "spd/cmpfit/mpfit.h"

namespace spdlib
{
	class SPDImporterProcessor
	{
	public:
		SPDImporterProcessor() throw(SPDException){};
		virtual void processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException) = 0;
		virtual ~SPDImporterProcessor(){};
	};
	
	class SPDDataImporter
	{
	public:
		SPDDataImporter(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold):convertCoords(false),outputProjWKT(""),indexCoords(SPD_FIRST_RETURN),pj_in(),pj_out(),coordTransform(NULL),haveCoordsBeenInit(false),defineOrigin(false),originX(0),originY(0),originZ(0),waveNoiseThreshold(0)
		{
			this->convertCoords = convertCoords;
			this->outputProjWKT = outputProjWKT;
            this->schema = schema;
			this->indexCoords = indexCoords;
            this->defineOrigin = defineOrigin;
            this->originX = originX;
            this->originY = originY;
            this->originZ = originZ;
            this->waveNoiseThreshold = waveNoiseThreshold;
		};
        virtual SPDDataImporter* getInstance(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold)=0;
		virtual std::list<SPDPulse*>* readAllDataToList(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)=0;
		virtual std::vector<SPDPulse*>* readAllDataToVector(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)=0;
		virtual void readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) throw(SPDIOException)=0;
		virtual bool isFileType(std::string fileType)=0;
        virtual void readHeaderInfo(std::string inputFile, SPDFile *spdFile) throw(SPDIOException)=0;
        virtual void readSchema()throw(SPDIOException){};
		virtual void setOutputProjWKT(std::string outputProjWKT){this->outputProjWKT = outputProjWKT;this->convertCoords = true;};
		virtual void setConvertCoords(bool convertCoords){this->convertCoords = convertCoords;};
		virtual void setIndexCoordsType(boost::uint_fast16_t indexCoords=SPD_FIRST_RETURN){this->indexCoords = indexCoords;};
        virtual void setDefineOrigin(bool defineOrigin=false){this->defineOrigin = defineOrigin;};
        virtual void setOriginX(double originX=0){this->originX = originX;};
        virtual void setOriginY(double originY=0){this->originY = originY;};
        virtual void setOriginZ(float originZ=0){this->originZ = originZ;};
        virtual void setWaveNoiseThreshold(float waveNoiseThreshold=0){this->waveNoiseThreshold = waveNoiseThreshold;};
		virtual ~SPDDataImporter()
		{
			if(haveCoordsBeenInit)
			{
				delete coordTransform;
				delete pj_in;
				delete pj_out;
			}
		};
	protected:
		bool convertCoords;
		std::string outputProjWKT;
        std::string schema;
        boost::uint_fast16_t indexCoords;
		OGRSpatialReference *pj_in;
		OGRSpatialReference *pj_out;
		OGRCoordinateTransformation *coordTransform;
		bool haveCoordsBeenInit;
        bool defineOrigin;
        double originX;
        double originY;
        float originZ;
        float waveNoiseThreshold;
		void transformCoordinateSystem(double *x, double *y, double *z) throw(SPDIOException)
		{
			if(!haveCoordsBeenInit)
			{
				throw SPDIOException("Coordinate systems have not been initialised.");
			}
			
			try 
			{
				if( (coordTransform == NULL) || !coordTransform->Transform( 1, x, y, z ) )
				{
					throw SPDIOException("Transformation of coordinates failed.");
				}
			}
			catch (SPDIOException &e) 
			{
				throw e;
			}
		};
		void transformCoordinateSystem(double *x, double *y, float *z) throw(SPDIOException)
		{
			if(!haveCoordsBeenInit)
			{
				throw SPDIOException("Coordinate systems have not been initialised.");
			}
			
			try 
			{
                double val = *z;
				if( (coordTransform == NULL) || !coordTransform->Transform( 1, x, y, &val ) )
				{
					throw SPDIOException("Transformation of coordinates failed.");
				}
                *z = val;
			}
			catch (SPDIOException &e) 
			{
				throw e;
			}
		};
		void transformCoordinateSystem(double *x, double *y) throw(SPDIOException)
		{
			if(!haveCoordsBeenInit)
			{
				throw SPDIOException("Coordinate systems have not been initialised.");
			}
			
			try 
			{
				if( (coordTransform == NULL) || !coordTransform->Transform( 1, x, y ) )
				{
					throw SPDIOException("Transformation of coordinates failed.");
				}
			}
			catch (SPDIOException &e) 
			{
				throw e;
			}
		};
		void initCoordinateSystemTransformation(SPDFile *spdFile) throw(SPDIOException)
		{
			pj_in = new OGRSpatialReference();
			char **inProjWKT = new char*[1];
			inProjWKT[0] = const_cast<char *>(spdFile->getSpatialReference().c_str());
			if(pj_in->importFromWkt(inProjWKT) != OGRERR_NONE)
			{
				std::string message = std::string("Could not create projection for \'") + spdFile->getSpatialReference() + std::string("\': ") + std::string(CPLGetLastErrorMsg());
				throw SPDIOException(message);
			}
			
			pj_out = new OGRSpatialReference();
			char **outProjWKT = new char*[1];
			outProjWKT[0] = const_cast<char *>(outputProjWKT.c_str());
			if(pj_out->importFromWkt(outProjWKT) != OGRERR_NONE)
			{
				std::string message = std::string("Could not create projection for \'") + outputProjWKT + std::string("\': ") + std::string(CPLGetLastErrorMsg());
				throw SPDIOException(message);			
			}
			
			coordTransform = OGRCreateCoordinateTransformation(pj_in, pj_out);
			
			if(coordTransform == NULL)
			{
				throw SPDIOException("Could not create coordinate transformation object.");
			}
			
			haveCoordsBeenInit = true;
		};
		void defineIdxCoords(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException)
		{
			if((spdFile->getDiscretePtDefined() == SPD_TRUE) | (spdFile->getDecomposedPtDefined() == SPD_TRUE))
			{
				if(indexCoords == SPD_FIRST_RETURN)
				{
					if(pulse->numberOfReturns > 0)
					{
						SPDPoint *pt = NULL;
						pt = pulse->pts->front();
						pulse->xIdx = pt->x;
						pulse->yIdx = pt->y;
					}
				}
				else if(indexCoords == SPD_LAST_RETURN)
				{
					if(pulse->numberOfReturns > 0)
					{
						SPDPoint *pt = NULL;
						pt = pulse->pts->back();
						pulse->xIdx = pt->x;
						pulse->yIdx = pt->y;
					}
				}
				else if(indexCoords == SPD_START_OF_RECEIVED_WAVEFORM)
				{
					if((spdFile->getReceiveWaveformDefined() == SPD_TRUE) & (pulse->numOfReceivedBins > 0))
					{
						double tempX = 0;
						double tempY = 0;
						double tempZ = 0;
						
						SPDConvertToCartesian(pulse->zenith, pulse->azimuth, pulse->rangeToWaveformStart, pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
						
						pulse->xIdx = tempX;
						pulse->yIdx = tempY;
					}
					else
					{
						throw SPDIOException("A waveform is not define for this pulse.");
					}
				}
				else if(indexCoords == SPD_END_OF_RECEIVED_WAVEFORM)
				{
					if((spdFile->getReceiveWaveformDefined() == SPD_TRUE) & (pulse->numOfReceivedBins > 0))
					{
						double tempX = 0;
						double tempY = 0;
						double tempZ = 0;
						
						SPDConvertToCartesian(pulse->zenith, pulse->azimuth, (pulse->rangeToWaveformStart+((((float)pulse->numOfReceivedBins)/2)*SPD_SPEED_OF_LIGHT_NS)), pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
						
						pulse->xIdx = tempX;
						pulse->yIdx = tempY;
					}
					else
					{
						throw SPDIOException("A waveform is not define for this pulse.");
					}
				}
				else if(indexCoords == SPD_ORIGIN)
				{
					if(spdFile->getOriginDefined() == SPD_FALSE)
					{
						throw SPDIOException("Origin is undefined.");
					}
					
					pulse->xIdx = pulse->x0;
					pulse->yIdx = pulse->y0;
				}
				else if(indexCoords == SPD_MAX_INTENSITY)
				{
					unsigned int maxIdx = 0;
					double maxVal = 0;
					bool first = true;
					for(unsigned int i = 0; i < pulse->pts->size(); ++i)
					{
						if(first)
						{
							maxIdx = i;
							maxVal = pulse->pts->at(i)->amplitudeReturn;
							first = false;
						}
						else if(pulse->pts->at(i)->amplitudeReturn > maxVal)
						{
							maxIdx = i;
							maxVal = pulse->pts->at(i)->amplitudeReturn;
						}
					}
					
					pulse->xIdx = pulse->pts->at(maxIdx)->x;
					pulse->yIdx = pulse->pts->at(maxIdx)->y;
				}
				else if(indexCoords == SPD_IDX_UNCHANGED)
				{
					// Do nothing...
				}
				else
				{
					throw SPDIOException("New index method is unknown.");
				}
			}
			else if((spdFile->getReceiveWaveformDefined() == SPD_TRUE) & (pulse->numOfReceivedBins > 0))
			{
				if((indexCoords == SPD_START_OF_RECEIVED_WAVEFORM) | (indexCoords == SPD_FIRST_RETURN))
				{
					double tempX = 0;
					double tempY = 0;
					double tempZ = 0;
					
					SPDConvertToCartesian(pulse->zenith, pulse->azimuth, pulse->rangeToWaveformStart, pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
					
					pulse->xIdx = tempX;
					pulse->yIdx = tempY;
				}
				else if((indexCoords == SPD_END_OF_RECEIVED_WAVEFORM) | (indexCoords == SPD_LAST_RETURN))
				{
					double tempX = 0;
					double tempY = 0;
					double tempZ = 0;
					
					SPDConvertToCartesian(pulse->zenith, pulse->azimuth, (pulse->rangeToWaveformStart+((((float)pulse->numOfReceivedBins)/2)*SPD_SPEED_OF_LIGHT_NS)), pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
					
					pulse->xIdx = tempX;
					pulse->yIdx = tempY;
				}
				else if(indexCoords == SPD_ORIGIN)
				{
					if(spdFile->getOriginDefined() == SPD_FALSE)
					{
						throw SPDIOException("Origin is undefined.");
					}
					pulse->xIdx = pulse->x0;
					pulse->yIdx = pulse->y0;
				}
				else if(indexCoords == SPD_MAX_INTENSITY)
				{
					double tempX = 0;
					double tempY = 0;
					double tempZ = 0;
					
					unsigned int maxIdx = 0;
					double maxVal = 0;
					bool first = true;
					for(unsigned int i = 0; i < pulse->numOfReceivedBins; ++i)
					{
						if(first)
						{
							maxIdx = i;
							maxVal = pulse->received[i];
							first = false;
						}
						else if(pulse->received[i] > maxVal)
						{
							maxIdx = i;
							maxVal = pulse->received[i];
						}
					}
					
					SPDConvertToCartesian(pulse->zenith, pulse->azimuth, (pulse->rangeToWaveformStart+((((float)maxIdx)/2)*SPD_SPEED_OF_LIGHT_NS)), pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
					
					pulse->xIdx = tempX;
					pulse->yIdx = tempY;
				}
				else if(indexCoords == SPD_IDX_UNCHANGED)
				{
					// Do nothing...
				}
				else
				{
					throw SPDIOException("New index method is unknown.");
				}
			}
			else if(spdFile->getReceiveWaveformDefined() == SPD_TRUE)
			{
				if(spdFile->getOriginDefined() == SPD_FALSE)
				{
					throw SPDIOException("Origin is undefined.");
				}
				
				pulse->xIdx = pulse->x0;
				pulse->yIdx = pulse->y0; 
			}
			else
			{
				throw SPDIOException("Neither a waveform or point data is defined.");
			}
		};
		void transformPulseCoords(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException)
		{
			try 
			{
				if(spdFile->getOriginDefined() == SPD_TRUE)
				{
					this->transformCoordinateSystem(&pulse->x0, &pulse->y0, &pulse->z0);
				}
				
				this->transformCoordinateSystem(&pulse->xIdx, &pulse->yIdx);
				
				if((spdFile->getDiscretePtDefined() == SPD_TRUE) | (spdFile->getDecomposedPtDefined() == SPD_TRUE))
				{
					if(pulse->numberOfReturns > 0)
					{
						for(std::vector<SPDPoint*>::iterator iterPts = pulse->pts->begin(); iterPts != pulse->pts->end(); ++iterPts)
						{
							this->transformCoordinateSystem(&((*iterPts)->x), &((*iterPts)->y), &((*iterPts)->z));
						}
					}
				}
			}
			catch (SPDIOException &e) 
			{
				throw e;
			}
		};
	};
}

#endif

