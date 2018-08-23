/*
 *  SPDImportSALCAData2SPD.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 04/12/2013.
 *  Copyright 2013 SPDLib. All rights reserved.
 *
 *  Code within this file has been provided by 
 *  Steven Hancock for reading the SALCA data
 *  and sorting out the geometry. This has been 
 *  adapted and brought across into the SPD
 *  importer interface.
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

#include "spd/SPDImportSALCAData2SPD.h"

namespace spdlib
{
    
    SPDSALCADataBinaryImporter::SPDSALCADataBinaryImporter(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold):SPDDataImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold)
    {
        
    }
		
    SPDDataImporter* SPDSALCADataBinaryImporter::getInstance(bool convertCoords, std::string outputProjWKT,std::string schema,boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold)
    {
        return new SPDSALCADataBinaryImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
    }
    
    std::list<SPDPulse*>* SPDSALCADataBinaryImporter::readAllDataToList(std::string, SPDFile *spdFile)throw(SPDIOException)
    {
        std::list<SPDPulse*> *pulses = new std::list<SPDPulse*>();
        
        return pulses;
    }
    
    std::vector<SPDPulse*>* SPDSALCADataBinaryImporter::readAllDataToVector(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)
    {
        std::vector<SPDPulse*> *pulses = new std::vector<SPDPulse*>();
        
        return pulses;
    }
    
    void SPDSALCADataBinaryImporter::readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) throw(SPDIOException)
    {
        try
        {
            std::vector<std::pair<float, std::string> > *inputBinFiles = new std::vector<std::pair<float, std::string> >();
            SalcaHDRParams *inParams = this->readHeaderParameters(inputFile, inputBinFiles);

            std::cout << "Header Parameters: \n";
            std::cout << "\tmaxR = " << inParams->maxR << std::endl;
            std::cout << "\tnAz = " << inParams->nAz << std::endl;
            std::cout << "\tnZen = " << inParams->nZen << std::endl;
            std::cout << "\tazStep = " << inParams->azStep << std::endl;
            std::cout << "\tzStep = " << inParams->zStep << std::endl;
            std::cout << "\tmaxZen = " << inParams->maxZen << std::endl;
            std::cout << "\tazStart = " << inParams->azStart << std::endl;
            std::cout << "\tazSquint = " << inParams->azSquint << std::endl;
            std::cout << "\tzenSquint = " << inParams->zenSquint << std::endl;
            std::cout << "\tomega = " << inParams->omega << std::endl;
            
            if(inputBinFiles->size() != inParams->nAz)
            {
                throw SPDIOException("The number of specified azimuth values is not equal to the number of input files.");
            }
            std::vector<float> wavelengths;
            wavelengths.push_back(1063);
            wavelengths.push_back(1545);
            spdFile->setWavelengths(wavelengths);
            std::vector<float> bandWidths;
            bandWidths.push_back(1);
            bandWidths.push_back(1);
            spdFile->setBandwidths(bandWidths);
            spdFile->setNumOfWavelengths(2);
            spdFile->setReceiveWaveformDefined(SPD_TRUE);
            spdFile->setTemporalBinSpacing(0.5);
            
            
            unsigned int numb = 0;           // Number of zenith steps
            unsigned int nBins = 0;          // Range bins per file (one file per azimuth)
            unsigned int length = 0;         // Total file length
            unsigned int maxRangeNBins = 0;  // Number of bins which represented the maximum range extent.
            float zen = 0.0;                 // True zenith
            float az = 0.0;                  // True azimuth
            int *data = NULL;                // Waveform data. Holds whole file.
            //char ringTest = 0;             // Ringing and saturation indicator
            SPDPulse *pulse = NULL;
            SPDPulseUtils plsUtils;
            size_t pulseID = 0;
            boost::uint_fast32_t scanline = 0;
            boost::uint_fast16_t scanlineIdx = 0;
            
            //sOffset=7;   // 1.05 m
            
            maxRangeNBins = inParams->maxR / 0.15;
            
            numb=(int)(inParams->maxZen/0.059375); // Number of zenith steps. Zenith res is hard wired for SALCA
            nBins=(int)((150.0+inParams->maxR)/0.15);
            
            for(std::vector<std::pair<float, std::string> >::iterator iterFiles = inputBinFiles->begin(); iterFiles != inputBinFiles->end(); ++iterFiles)
            {
                std::cout << (*iterFiles).second << " with azimuth = " << (*iterFiles).first << std::endl;
                
                az = (float)(*iterFiles).first;
                
                // Read binary data into an array
                data=this->readData((*iterFiles).second, (*iterFiles).first, numb, nBins, &length);
                
                
                std::cout << "Number = " << numb << std::endl;
                std::cout << "nBins = " << nBins << std::endl;
                std::cout << "Length = " << length << std::endl;
                std::cout << "Max Range (nBins) = " << maxRangeNBins << std::endl;
                
                
                if(data)
                {
                    zen = 0.0;
                    unsigned int wl1StartIdxRec = 0;
                    unsigned int wl1EndIdxRec = 0;
                    unsigned int wl2StartIdxRec = 0;
                    unsigned int wl2EndIdxRec = 0;
                    
                    unsigned int wl1StartIdxTrans = 0;
                    unsigned int wl1EndIdxTrans = 0;
                    unsigned int wl2StartIdxTrans = 0;
                    unsigned int wl2EndIdxTrans = 0;
                    
                    unsigned int prevWL2EndIdx = 0;
                    
                    for(unsigned int z = 0; z < numb; ++z)
                    {
                        //std::cout << z << ") Zenith = " << zen << std::endl;
                        
                        wl1StartIdxRec = 0;
                        wl1EndIdxRec = 0;
                        
                        wl2StartIdxRec = 0;
                        wl2EndIdxRec = 0;
                        
                        wl1StartIdxTrans = 0;
                        wl1EndIdxTrans = 0;
                        
                        wl2StartIdxTrans = 0;
                        wl2EndIdxTrans = 0;
                        
                        this->findWaveformsBinIdxes(data, length, maxRangeNBins, prevWL2EndIdx, &wl1StartIdxTrans, &wl2StartIdxTrans, &wl1EndIdxTrans, &wl2EndIdxTrans, &wl1StartIdxRec, &wl2StartIdxRec, &wl1EndIdxRec, &wl2EndIdxRec);
                        /*
                        std::cout << "Wavelength #1: Trans = [" << wl1StartIdxTrans << ", " << wl1EndIdxTrans << "] (" << wl1EndIdxTrans - wl1StartIdxTrans << ") Rec = [" << wl1StartIdxRec << ", " << wl1EndIdxRec << "] (" << wl1EndIdxRec - wl1StartIdxRec << ")\n";
                        
                        std::cout << "Wavelength #2: Trans = [" << wl2StartIdxTrans << ", " << wl2EndIdxTrans << "] (" << wl2EndIdxTrans - wl2StartIdxTrans << ") Rec = [" << wl2StartIdxRec << ", " << wl2EndIdxRec << "] (" << wl2EndIdxRec - wl2StartIdxRec << ")\n";
                          */
                        // WAVELENGTH #1
                        pulse = new SPDPulse();
                        plsUtils.initSPDPulse(pulse);
                        pulse->pulseID = pulseID++;
                        
                        pulse->wavelength = 1063;
                        pulse->numOfReceivedBins = wl1EndIdxRec - wl1StartIdxRec;
                        pulse->received = new uint_fast32_t[pulse->numOfReceivedBins];
                        for(unsigned int i = wl1StartIdxRec, n = 0; i < wl1EndIdxRec; ++i, ++n)
                        {
                            pulse->received[n] = data[i];
                        }
                        
                        pulse->numOfTransmittedBins = wl1EndIdxTrans - wl1StartIdxTrans;
                        pulse->transmitted = new uint_fast32_t[pulse->numOfTransmittedBins];
                        for(unsigned int i = wl1StartIdxTrans, n = 0; i < wl1EndIdxTrans; ++i, ++n)
                        {
                            pulse->transmitted[n] = data[i];
                        }
                        
                        if(this->defineOrigin)
                        {
                            pulse->x0 = this->originX;
                            pulse->y0 = this->originY;
                            pulse->z0 = this->originZ;
                        }
                        else
                        {
                            pulse->x0 = 0;
                            pulse->y0 = 0;
                            pulse->z0 = 0;
                        }
                        
                        pulse->azimuth = ((*iterFiles).first/180.0)*M_PI;
                        pulse->zenith = ((zen/180.0)*M_PI)+(M_PI/2);
                        pulse->scanline = scanlineIdx;
                        pulse->scanlineIdx = scanline;
                        pulse->receiveWaveNoiseThreshold = 30;
                        pulse->rangeToWaveformStart = 0;
                        pulse->sourceID = 1;
                        
                        processor->processImportedPulse(spdFile, pulse);
                        
                        
                        
                        // WAVELENGTH #2
                        pulse = new SPDPulse();
                        plsUtils.initSPDPulse(pulse);
                        pulse->pulseID = pulseID++;
                        
                        pulse->wavelength = 1545;
                        pulse->numOfReceivedBins = wl2EndIdxRec - wl2StartIdxRec;
                        pulse->received = new uint_fast32_t[pulse->numOfReceivedBins];
                        for(unsigned int i = wl2StartIdxRec, n = 0; i < wl2EndIdxRec; ++i, ++n)
                        {
                            pulse->received[n] = data[i];
                        }
                        
                        pulse->numOfTransmittedBins = wl2EndIdxTrans - wl2StartIdxTrans;
                        pulse->transmitted = new uint_fast32_t[pulse->numOfTransmittedBins];
                        for(unsigned int i = wl2StartIdxTrans, n = 0; i < wl2EndIdxTrans; ++i, ++n)
                        {
                            pulse->transmitted[n] = data[i];
                        }
                        
                        if(this->defineOrigin)
                        {
                            pulse->x0 = this->originX;
                            pulse->y0 = this->originY;
                            pulse->z0 = this->originZ;
                        }
                        else
                        {
                            pulse->x0 = 0;
                            pulse->y0 = 0;
                            pulse->z0 = 0;
                        }
                        
                        pulse->azimuth = ((*iterFiles).first/180.0)*M_PI;
                        pulse->zenith = ((zen/180.0)*M_PI)+(M_PI/2);
                        pulse->scanline = scanlineIdx;
                        pulse->scanlineIdx = scanline;
                        pulse->receiveWaveNoiseThreshold = 30;
                        pulse->rangeToWaveformStart = 0;
                        pulse->sourceID = 2;
                        
                        processor->processImportedPulse(spdFile, pulse);
                        
                        
                        prevWL2EndIdx = wl2EndIdxRec;
                        zen += 0.059375;
                        ++scanlineIdx;
                    }
                    delete[] data;
                }
                else
                {
                    throw SPDIOException("The data file was not opened or data was not read in correctly.");
                }
                
                ++scanline;
                scanlineIdx = 0;
            }
            
            if(inParams->azOff != NULL)
            {
                delete[] inParams->azOff;
            }
            if(inParams->zen != NULL)
            {
                delete[] inParams->zen;
            }
            
            delete inParams;
            delete inputBinFiles;
        }
        catch (SPDIOException &e)
        {
            throw e;
        }
        catch (std::exception &e)
        {
            throw SPDIOException(e.what());
        }
    }
    
    bool SPDSALCADataBinaryImporter::isFileType(std::string fileType)
    {
        if(fileType == "SALCA")
		{
			return true;
		}
		return false;
    }
    
    void SPDSALCADataBinaryImporter::readHeaderInfo(std::string inputFile, SPDFile *spdFile) throw(SPDIOException)
    {
        
    }
    
    SalcaHDRParams* SPDSALCADataBinaryImporter::readHeaderParameters(std::string headerFilePath, std::vector<std::pair<float,std::string> > *fileList)throw(SPDIOException)
    {
        SalcaHDRParams *hdrParams = new SalcaHDRParams();
        try
        {
            if(boost::filesystem::exists(headerFilePath))
            {
                boost::filesystem::path filePath = boost::filesystem::path(headerFilePath);
                filePath = boost::filesystem::absolute(filePath).parent_path();
                boost::filesystem::path tmpFilePath;
                
                
                SPDTextFileLineReader lineReader;
                SPDTextFileUtilities txtUtils;
                std::vector<std::string> *tokens = new std::vector<std::string>();
                lineReader.openFile(headerFilePath);
                std::string line = "";
                std::string filePathStr = "";
                float azimuthVal = 0.0;
                
                bool readingHeader = false;
                bool readingFileList = false;
                
                while(!lineReader.endOfFile())
                {
                    line = lineReader.readLine();
                    boost::algorithm::trim(line);
                    
                    if((line != "") && (!txtUtils.lineStartWithHash(line)))
                    {
                        if(line == "[START HEADER]")
                        {
                            readingHeader = true;
                        }
                        else if(line == "[END HEADER]")
                        {
                            readingHeader = false;
                        }
                        else if(line == "[FILE LIST START]")
                        {
                            readingHeader = false;
                            readingFileList = true;
                        }
                        else if(line == "[FILE LIST END]")
                        {
                            readingFileList = false;
                        }
                        
                        if(readingHeader & (line != "[START HEADER]"))
                        {
                            tokens->clear();
                            txtUtils.tokenizeString(line, '=', tokens, true);
                            
                            if(tokens->size() >= 2)
                            {
                                if(txtUtils.lineContainsChar(tokens->at(1), '#'))
                                {
                                    std::string temp1Val = tokens->at(0);
                                    std::string temp2Val = tokens->at(1);
                                    tokens->clear();
                                    txtUtils.tokenizeString(temp2Val, '#', tokens, true);
                                    temp2Val = tokens->at(0);
                                    tokens->clear();
                                    tokens->push_back(temp1Val);
                                    tokens->push_back(temp2Val);
                                }
                            }

                            if(tokens->at(0) == "maxR")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'maxR\'.");
                                }
                                hdrParams->maxR = txtUtils.strtofloat(tokens->at(1));
                            }
                            else if(tokens->at(0) == "nAz")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'nAz\'.");
                                }
                                hdrParams->nAz = txtUtils.strto16bitUInt(tokens->at(1));
                            }
                            else if(tokens->at(0) == "nZen")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'nZen\'.");
                                }
                                hdrParams->nZen = txtUtils.strto16bitUInt(tokens->at(1));
                            }
                            else if(tokens->at(0) == "azStep")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'azStep\'.");
                                }
                                hdrParams->azStep = txtUtils.strtofloat(tokens->at(1));
                            }
                            else if(tokens->at(0) == "zStep")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'zStep\'.");
                                }
                                hdrParams->zStep = txtUtils.strtofloat(tokens->at(1));
                            }
                            else if(tokens->at(0) == "maxZen")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'maxZen\'.");
                                }
                                hdrParams->maxZen = txtUtils.strtofloat(tokens->at(1));
                            }
                            else if(tokens->at(0) == "azStart")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'azStart\'.");
                                }
                                hdrParams->azStart = txtUtils.strtofloat(tokens->at(1));
                            }
                            else if(tokens->at(0) == "omega")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'omega\'.");
                                }
                                hdrParams->omega = txtUtils.strtofloat(tokens->at(1));
                            }
                            else
                            {
                                std::cout << line << std::endl;
                                
                                throw SPDIOException("Could not parser the header.");
                            }
                            
                        }
                        else if(readingFileList & (line != "[FILE LIST START]"))
                        {
                            tokens->clear();
                            txtUtils.tokenizeString(line, ':', tokens, true);
                            if(tokens->size() != 2)
                            {
                                throw SPDIOException("Failed to parser file list (Structure should be \'azimuth:filename\').");
                            }
                            azimuthVal = txtUtils.strtofloat(tokens->at(0));
                            
                            tmpFilePath = filePath;
                            tmpFilePath /= boost::filesystem::path(tokens->at(1));
                            
                            fileList->push_back(std::pair<float, std::string>(azimuthVal, tmpFilePath.string()));
                        }
                    }
                }
                delete tokens;
                lineReader.closeFile();
                
            }
        }
        catch (SPDIOException &e)
        {
            delete hdrParams;
            throw e;
        }
        catch (std::exception &e)
        {
            delete hdrParams;
            throw SPDIOException(e.what());
        }
        
        return hdrParams;
    }
 
    
    /** Read data into array */
    int* SPDSALCADataBinaryImporter::readData(std::string inFilePath, int i, unsigned int numb, unsigned int nBins, unsigned int *length) throw(SPDIOException)
    {
        int *outData = NULL;
        try
        {
            char *data = NULL;
            FILE *ipoo = NULL;
            
            // Open the input file
            if((ipoo=fopen(inFilePath.c_str(),"rb"))==NULL)
            {
                std::string message = std::string("Could not open file: \'") + inFilePath + std::string("\'");
                throw SPDIOException(message);
            }
            
            // Determine the file length
            if(fseek(ipoo,(long)0,SEEK_END))
            {
                throw SPDIOException("Could not determine the length of the file.");
            }
            *length=ftell(ipoo);
            
            if((nBins*numb)>(*length))
            {
                std::string message = std::string("File size mismatch: \'") + inFilePath + std::string("\'");
                std::cout << "(nBins*numb) = " << nBins*numb << std::endl;
                std::cout << "*length = " << *length << std::endl;
                throw SPDIOException(message);
            }
            data = new char[*length];
            
            // Now we know how long, read the file
            if(fseek(ipoo,(long)0,SEEK_SET))
            {
                throw SPDIOException("Could not restart the seek read to the beginning of the file.");
            }
            
            // Read the data into the data array
            if(fread(&(data[0]),sizeof(char),*length,ipoo)!=*length)
            {
                throw SPDIOException("Failed to read the data - reason unknown.");
            }
            
            if(ipoo)
            {
                fclose(ipoo);
                ipoo=NULL;
            }
            
            outData = new int[*length];
            for(int i = 0; i < (*length); ++i)
            {
                outData[i] = ((int)data[i])+128;
            }
            delete[] data;
            
        }
        catch(std::exception &e)
        {
            throw SPDIOException(e.what());
        }
        catch(SPDIOException &e)
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDIOException(e.what());
        }
    
        return(outData);
    }

    void SPDSALCADataBinaryImporter::findWaveformsBinIdxes(int *data, unsigned int dataLen, unsigned int maxRNBins, unsigned int prevWl2End, unsigned int *wl1StartIdxTrans, unsigned int *wl2StartIdxTrans, unsigned int *wl1EndIdxTrans, unsigned int *wl2EndIdxTrans, unsigned int *wl1StartIdxRec, unsigned int *wl2StartIdxRec, unsigned int *wl1EndIdxRec, unsigned int *wl2EndIdxRec) throw(SPDIOException)
    {
        try
        {
            bool foundWl1Trans = false;
            bool foundWl2Trans = false;
            bool foundNextWl1Trans = false;
            unsigned int nextWl1TransIdx = 0;
            for(unsigned int i = prevWl2End; i < dataLen; ++i)
            {
                // Check for zero crossing... (i.e. a peak).
                if(data[i] > 230)
                {
                    if(i == 0)
                    {
                        if(this->zeroCrossing(data, 0, 2, i))
                        {
                            foundWl1Trans = true;
                            *wl1StartIdxTrans = i;
                        }
                    }
                    else
                    {
                        bool zeroCross = false;
                        
                        if(i == 1)
                        {
                            zeroCross = this->zeroCrossing(data, i-1, i+2, i);
                        }
                        else if(i == dataLen-1)
                        {
                            zeroCross = this->zeroCrossing(data, i-1, i, i);
                        }
                        else if(i == dataLen-2)
                        {
                            zeroCross = this->zeroCrossing(data, i-2, i+1, i);
                        }
                        else
                        {
                            zeroCross = this->zeroCrossing(data, i-2, i+2, i);
                        }
                        
                        if(zeroCross)
                        {
                            if(!foundWl1Trans)
                            {
                                if((i - prevWl2End) > 40)
                                {
                                    foundWl1Trans = true;
                                    *wl1StartIdxTrans = prevWl2End;
                                    foundWl2Trans = true;
                                    *wl2StartIdxTrans = i-1;
                                }
                                else
                                {
                                    foundWl1Trans = true;
                                    *wl1StartIdxTrans = i-1;
                                }
                            }
                            else if(!foundWl2Trans)
                            {
                                if(((i-1) > (*wl1StartIdxTrans)) & (((i-1)) - (*wl1StartIdxTrans) > maxRNBins))
                                {
                                    foundWl2Trans = true;
                                    *wl2StartIdxTrans = i-1;
                                }
                            }
                            else if(!foundNextWl1Trans)
                            {
                                if((((i-1)) > (*wl2StartIdxTrans)) & (((i-1)) - (*wl2StartIdxTrans) > maxRNBins))
                                {
                                    foundNextWl1Trans = true;
                                    nextWl1TransIdx = i-1;
                                    break;
                                }
                            }
                            else
                            {
                                break;
                            }
                        }
                    }
                }
            }
            
            std::cout << "Start W1 Trans Idx = " << *wl1StartIdxTrans << std::endl;
            std::cout << "Start W2 Trans Idx = " << *wl2StartIdxTrans << std::endl;
            std::cout << "Start Next Trans Idx = " << nextWl1TransIdx << std::endl;
            
            
            if((foundWl1Trans & foundWl2Trans) & (!foundNextWl1Trans))
            {
                nextWl1TransIdx = dataLen;
            }
            else if((!foundWl1Trans) | (!foundWl2Trans) | (!foundNextWl1Trans))
            {
                throw SPDIOException("Did not find all the required transmitted waveforms...");
            }
            
            *wl1EndIdxTrans = (*wl1StartIdxTrans) + 2;
            *wl1StartIdxRec = (*wl1EndIdxTrans) + 1;
            *wl1EndIdxRec = (*wl2StartIdxTrans) - 1;
            if((*wl1EndIdxRec) >= dataLen)
            {
                (*wl1EndIdxRec) = dataLen;
            }
            else if(((*wl1EndIdxRec) - (*wl1StartIdxRec)) > 1400)
            {
                std::cout << "(*wl1StartIdxRec) = " << (*wl1StartIdxRec) << std::endl;
                std::cout << "(*wl1EndIdxRec) = " << (*wl1EndIdxRec) << std::endl;
                
                for(unsigned int i = (*wl1StartIdxRec); i < (*wl1EndIdxRec)+10; ++i)
                {
                    if(i == (*wl1StartIdxRec))
                    {
                        std::cout << data[i];
                    }
                    else
                    {
                        std::cout << ", " << data[i];
                    }
                }
                std::cout << std::endl;
                
                throw SPDIOException("Number of bins too long for a single pulse error occurred.");
                //(*wl1EndIdxRec) = (*wl1StartIdxRec) + maxRNBins;
            }
            
            *wl2EndIdxTrans = (*wl2StartIdxTrans) + 2;
            *wl2StartIdxRec = (*wl2EndIdxTrans) + 1;
            *wl2EndIdxRec = nextWl1TransIdx-1;
            if((*wl2EndIdxRec) >= dataLen)
            {
                (*wl2EndIdxRec) = dataLen;
            }
            else if(((*wl2EndIdxRec) - (*wl2StartIdxRec)) > 1400)
            {
                std::cout << "(*wl2StartIdxRec) = " << (*wl2StartIdxRec) << std::endl;
                std::cout << "(*wl2EndIdxRec) = " << (*wl2EndIdxRec) << std::endl;
                
                for(unsigned int i = (*wl2StartIdxRec)-15; i < (*wl2EndIdxRec)+10; ++i)
                {
                    if(i == (*wl2StartIdxRec))
                    {
                        std::cout << data[i];
                    }
                    else
                    {
                        std::cout << ", " << data[i];
                    }
                }
                std::cout << std::endl;
                
                throw SPDIOException("Number of bins too long for a single pulse error occurred.");
                //(*wl2EndIdxRec) = (*wl2StartIdxRec) + maxRNBins;
            }
            
        }
        catch(std::exception &e)
        {
            throw SPDIOException(e.what());
        }
        catch(SPDIOException &e)
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDIOException(e.what());
        }
    }
    
    bool SPDSALCADataBinaryImporter::zeroCrossing(int *data, unsigned int startIdx, unsigned int endIdx, unsigned int idx) throw(SPDIOException)
    {
        //std::cout << "Idx = " << idx << " [" << startIdx << ", " << endIdx << "]\n";
        
        bool foundCrossing = false;
        try
        {
            if((endIdx - startIdx) == 2) // 3
            {
                if((startIdx+1) != idx)
                {
                    throw SPDIOException("For a length of 3 the index needs to be the centre");
                }
                
                if((data[idx] > data[startIdx]) & (data[idx] > data[endIdx]))
                {
                    foundCrossing = true;
                }
                else
                {
                    foundCrossing = false;
                }
            }
            else if((endIdx - startIdx) == 1) //2
            {
                if(idx == endIdx)
                {
                    if(data[idx] > data[startIdx])
                    {
                        foundCrossing = true;
                    }
                    else
                    {
                        foundCrossing = false;
                    }
                }
                else if(idx == startIdx)
                {
                    if(data[idx] > data[endIdx])
                    {
                        foundCrossing = true;
                    }
                    else
                    {
                        foundCrossing = false;
                    }
                }
                else
                {
                    throw SPDIOException("For a length of 2 the index needs to either be equal to the start or the end index.");
                }
            }
            else if((endIdx - startIdx) == 3) //4
            {
                if((idx == (startIdx+1)) | (idx == (endIdx-1)))
                {
                    if((data[idx] > data[idx-1]) & (data[idx] > data[idx+1]))
                    {
                        foundCrossing = true;
                    }
                    else if((data[startIdx+1] > data[startIdx]) & (data[startIdx+1] == data[startIdx+2]) & (data[startIdx+2] > data[startIdx+3]))
                    {
                        foundCrossing = true;
                    }
                    else
                    {
                        foundCrossing = false;
                    }
                }
                else
                {
                    throw SPDIOException("For a length of 4 the index needs to either of the middle two bins.");
                }
            }
            else if((endIdx - startIdx) == 4) //5
            {
                if(idx != startIdx+2)
                {
                    throw SPDIOException("For a length of 5 the index needs to be the middle bin.");
                }
                else
                {
                    if((data[idx] > data[idx-1]) & (data[idx] > data[idx+1]))
                    {
                        foundCrossing = true;
                    }
                    else if((data[idx] > data[idx-2]) & (data[idx] > data[idx+2]))
                    {
                        foundCrossing = true;
                    }
                    else
                    {
                        foundCrossing = false;
                    }
                }
            }
            else
            {
                throw SPDIOException("Can only find the zero crossing for regions of 2, 3, 4, or 5 bins.");
            }
        }
        catch(std::exception &e)
        {
            throw SPDIOException(e.what());
        }
        catch(SPDIOException &e)
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDIOException(e.what());
        }
        return foundCrossing;
    }
    
    SPDSALCADataBinaryImporter::~SPDSALCADataBinaryImporter()
    {
        
    }

}


