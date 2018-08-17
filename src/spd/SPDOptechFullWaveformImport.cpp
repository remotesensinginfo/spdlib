/*
 *  SPDOptechFullWaveformImport.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 28/04/2014.
 *  Copyright 2014 SPDLib. All rights reserved.
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

#include "spd/SPDOptechFullWaveformImport.h"


namespace spdlib
{
	
    SPDOptechFullWaveformASCIIImport::SPDOptechFullWaveformASCIIImport(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold):SPDDataImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold)
    {

    }

    SPDDataImporter* SPDOptechFullWaveformASCIIImport::getInstance(bool convertCoords, std::string outputProjWKT,std::string schema,boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold)
    {
        return new SPDOptechFullWaveformASCIIImport(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
    }

    std::list<SPDPulse*>* SPDOptechFullWaveformASCIIImport::readAllDataToList(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)
    {
        std::list<SPDPulse*> *pulses = new std::list<SPDPulse*>();

        try
        {
            std::string sensorFile = "";
            std::string waveformsFile = "";

            this->readSPDOPTHeader(inputFile, spdFile, &sensorFile, &waveformsFile);



        }
        catch(SPDIOException &e)
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDIOException(e.what());
        }
        catch(std::exception &e)
        {
            throw SPDIOException(e.what());
        }

        return pulses;
    }

    std::vector<SPDPulse*>* SPDOptechFullWaveformASCIIImport::readAllDataToVector(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)
    {
        std::vector<SPDPulse*> *pulses = new std::vector<SPDPulse*>();

        try
        {
            SPDTextFileUtilities txtUtils;
            SPDPulseUtils plsUtils;
            std::string sensorFile = "";
            std::string waveformsFile = "";

            this->readSPDOPTHeader(inputFile, spdFile, &sensorFile, &waveformsFile);

            if(boost::filesystem::exists(sensorFile) & boost::filesystem::exists(waveformsFile))
            {
                SPDTextFileLineReader lineSensorReader;
                SPDTextFileLineReader lineWavesReader;

                lineSensorReader.openFile(sensorFile);
                lineWavesReader.openFile(waveformsFile);

                size_t maxNumOfWaveforms = txtUtils.countLines(waveformsFile);
                pulses->reserve(maxNumOfWaveforms/2); // There is most likely be a transmitted and recieved waveform for each pulse.

                SPDPulse *pulse = NULL;
                std::string waveLine = "";
                std::string waveLineTiming = "";
                std::string waveLineBins = "";
                std::string waveformTime = "";
                std::string waveformTorR = "";
                std::string waveformDist2Start = "";
                std::string sensorLine = "";
                std::string sensorTime = "";
                double sensorRoll = 0.0;
                double sensorPitch = 0.0;
                double sensorHeading = 0.0;
                double shotAngle = 0.0;
                double shotDirect = 0.0;

                std::vector<std::string> *tokens = new std::vector<std::string>();
                bool transmitted = false;
                bool received = false;
                bool parseSensor = false;
                bool foundSensorLine = false;
                size_t plCount = 0;
                size_t idx = 0;
                for(size_t i = 0; i < maxNumOfWaveforms; ++i)
                {
                    if(!lineWavesReader.endOfFile())
                    {
                        waveLine = lineWavesReader.readLine();
                        boost::algorithm::trim(waveLine);
                        tokens->clear();
                        txtUtils.tokenizeString(waveLine, '[', tokens, true);
                        if(tokens->size() != 2)
                        {
                            std::cout << "LINE: " << waveLine << std::endl;
                            throw SPDIOException("Could not parse the waveform line.");
                        }

                        waveLineTiming = tokens->at(0);
                        boost::algorithm::trim(waveLineTiming);
                        waveLineBins = tokens->at(1);
                        waveLineBins = txtUtils.removeChar(waveLineBins, ']');
                        boost::algorithm::trim(waveLineBins);

                        tokens->clear();
                        txtUtils.tokenizeString(waveLineTiming, ' ', tokens, true);
                        if(tokens->size() != 3)
                        {
                            std::cout << "LINE: " << waveLine << std::endl;
                            throw SPDIOException("Could not parse the waveform line: There should be 3 values before the waveform bins.");
                        }
                        waveformTime = tokens->at(0);
                        boost::algorithm::trim(waveformTime);
                        waveformTorR = tokens->at(1);
                        boost::algorithm::trim(waveformTorR);
                        waveformDist2Start = tokens->at(2);
                        boost::algorithm::trim(waveformDist2Start);

                        if(waveformTorR == "0")
                        {
                            if(transmitted & parseSensor)
                            {
                                pulses->push_back(pulse);
                            }
                            else if(transmitted)
                            {
                                plsUtils.deleteSPDPulse(pulse);
                                std::cout << "WARNING: There was a pulse (ID: " << plCount-1 << ") for which the sensor and recieved waveform were not available\n";
                            }
                            transmitted = true;
                            received = false;
                            foundSensorLine = false;
                            parseSensor = false;
                            pulse = new SPDPulse();
                            plsUtils.initSPDPulse(pulse);

                            pulse->pulseID = plCount++;
                            pulse->gpsTime = static_cast<boost::uint_fast64_t>(txtUtils.strtodouble(waveformTime) * 1000000);
                            pulse->receiveWaveNoiseThreshold = receivedWaveformThershold;
                            pulse->receiveWaveOffset = receivedWaveformOffset;
                            pulse->receiveWaveGain = receivedWaveformGain;
                            pulse->transWaveNoiseThres = transWaveformThershold;
                            pulse->transWaveOffset = transWaveformOffset;
                            pulse->transWaveGain = transWaveformGain;
                            pulse->wavelength = laserWavelength;

                            tokens->clear();
                            txtUtils.tokenizeString(waveLineBins, ' ', tokens, true);
                            pulse->numOfTransmittedBins = tokens->size();
                            pulse->transmitted = new boost::uint_fast32_t[pulse->numOfTransmittedBins];
                            idx = 0;
                            for(std::vector<std::string>::iterator iterTokens = tokens->begin(); iterTokens != tokens->end(); ++iterTokens)
                            {
                                pulse->transmitted[idx++] = txtUtils.strto32bitUInt(*iterTokens);
                                //std::cout << pulse->transmitted[idx-1] << ", ";
                            }
                            //std::cout << std::endl;

                            while(!lineSensorReader.endOfFile())
                            {
                                sensorLine = lineSensorReader.readLine();
                                tokens->clear();
                                txtUtils.tokenizeString(sensorLine, ' ', tokens, true);
                                if(tokens->size() != 10)
                                {
                                    std::cout << "There are " << tokens->size() << " tokens in the line.\n";
                                    std::cout << "LINE: " << sensorLine << std::endl;
                                    throw SPDIOException("Could not parse the sensor line, should have 10 fields.");
                                }
                                sensorTime = tokens->at(7);
                                boost::algorithm::trim(sensorTime);
                                if(sensorTime == waveformTime)
                                {
                                    foundSensorLine = true;
                                    break;
                                }
                            }
                            if(!foundSensorLine)
                            {
                                throw SPDIOException("Could not find the sensor line.");
                            }

                            pulse->x0 = txtUtils.strtodouble(tokens->at(0));
                            pulse->y0 = txtUtils.strtodouble(tokens->at(1));
                            pulse->z0 = txtUtils.strtodouble(tokens->at(2));

                            sensorRoll = txtUtils.strtodouble(tokens->at(3));
                            sensorPitch = txtUtils.strtodouble(tokens->at(4));
                            sensorHeading = txtUtils.strtodouble(tokens->at(5));
                            shotAngle = txtUtils.strtodouble(tokens->at(8));
                            shotDirect = txtUtils.strtodouble(tokens->at(9));

                            std::cout << "Roll: " << sensorRoll << std::endl;
                            std::cout << "Pitch: " << sensorPitch << std::endl;
                            std::cout << "Heading: " << sensorHeading << std::endl;
                            std::cout << "Shot Angle: " << shotAngle << std::endl;
                            std::cout << "Shot Direction: " << shotDirect << std::endl;

                            pulse->zenith = 0.0;
                            pulse->azimuth = 0.0;

                            parseSensor = true;
                        }
                        else if(waveformTorR == "1")
                        {
                            if(!transmitted)
                            {
                                std::cout << "LINE: " << waveLine << std::endl;
                                throw SPDIOException("Recieved waveform was not preceeded by a Transmitted waveform");
                            }
                            received = true;

                            pulse->rangeToWaveformStart = txtUtils.strtofloat(waveformDist2Start)/100.0;

                            tokens->clear();
                            txtUtils.tokenizeString(waveLineBins, ' ', tokens, true);
                            pulse->numOfReceivedBins = tokens->size();
                            pulse->received = new boost::uint_fast32_t[pulse->numOfReceivedBins];
                            idx = 0;
                            for(std::vector<std::string>::iterator iterTokens = tokens->begin(); iterTokens != tokens->end(); ++iterTokens)
                            {
                                pulse->received[idx++] = txtUtils.strto32bitUInt(*iterTokens);
                                //std::cout << pulse->transmitted[idx-1] << ", ";
                            }
                        }
                    }
                    else
                    {
                        // All waveforms have been processed.
                        break;
                    }
                }





            }
            else
            {
                std::cout << "Sensor: " << sensorFile << std::endl;
                std::cout << "Waveforms: " << waveformsFile << std::endl;
                throw SPDIOException("Either the sensor or waveforms file could not be opened - please check the file paths.\nNOTE: path is relative to the header.");
            }
        }
        catch(SPDIOException &e)
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDIOException(e.what());
        }
        catch(std::exception &e)
        {
            throw SPDIOException(e.what());
        }

        return pulses;
    }
		
    void SPDOptechFullWaveformASCIIImport::readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) throw(SPDIOException)
    {
        try
        {
            std::string sensorFile = "";
            std::string waveformsFile = "";

            this->readSPDOPTHeader(inputFile, spdFile, &sensorFile, &waveformsFile);



        }
        catch(SPDIOException &e)
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDIOException(e.what());
        }
        catch(std::exception &e)
        {
            throw SPDIOException(e.what());
        }
    }

    bool SPDOptechFullWaveformASCIIImport::isFileType(std::string fileType)
    {
        if(fileType == "OPTECH_WFASCII")
		{
			return true;
		}
		return false;
    }

    void SPDOptechFullWaveformASCIIImport::readHeaderInfo(std::string inputFile, SPDFile *spdFile) throw(SPDIOException)
    {
        try
        {
            std::string sensorFile = "";
            std::string waveformsFile = "";

            this->readSPDOPTHeader(inputFile, spdFile, &sensorFile, &waveformsFile);
        }
        catch(SPDIOException &e)
        {
            throw e;
        }
    }

    void SPDOptechFullWaveformASCIIImport::readSPDOPTHeader(std::string inputHDRFile, SPDFile *spdFile, std::string *sensorFile, std::string *waveformsFile)throw(SPDIOException)
    {
        try
        {
            if(boost::filesystem::exists(inputHDRFile))
            {
                boost::filesystem::path filePath = boost::filesystem::path(inputHDRFile);
                filePath = boost::filesystem::absolute(filePath).parent_path();
                boost::filesystem::path tmpFilePath;


                SPDTextFileLineReader lineReader;
                SPDTextFileUtilities txtUtils;
                std::vector<std::string> *tokens = new std::vector<std::string>();
                lineReader.openFile(inputHDRFile);
                std::string line = "";
                std::string filePathStr = "";
                bool haveSensorFile = false;
                bool haveWaveformFile = false;
                bool haveTimeInterval = false;
                bool haveWavelengths = false;
                bool haveBandWidths = false;
                bool haveTThreshold = false;
                bool haveTGain = false;
                bool haveTOffset = false;
                bool haveRThreshold = false;
                bool haveRGain = false;
                bool haveROffset = false;
                bool haveAquDate = false;
                bool haveAquTime = false;


                while(!lineReader.endOfFile())
                {
                    line = lineReader.readLine();
                    boost::algorithm::trim(line);

                    if((line != "") && (!txtUtils.lineStartWithHash(line)))
                    {
                        tokens->clear();
                        txtUtils.tokenizeString(line, '=', tokens, true);
                        if(tokens->at(0) == "sensor")
                        {
                            if(tokens->size() != 2)
                            {
                                throw SPDIOException("Failed to parser header line \'sensor\'.");
                            }
                            tmpFilePath = filePath;
                            tmpFilePath /= boost::filesystem::path(tokens->at(1));

                            *sensorFile = tmpFilePath.string();
                            haveSensorFile = true;
                        }
                        else if(tokens->at(0) == "waveforms")
                        {
                            if(tokens->size() != 2)
                            {
                                throw SPDIOException("Failed to parser header line \'waveforms\'.");
                            }
                            tmpFilePath = filePath;
                            tmpFilePath /= boost::filesystem::path(tokens->at(1));

                            *waveformsFile = tmpFilePath.string();
                            haveWaveformFile = true;
                        }
                        else if(tokens->at(0) == "waveform_time_interval(ns)")
                        {
                            if(tokens->size() != 2)
                            {
                                throw SPDIOException("Failed to parser header line \'waveform_time_interval(ns)\'.");
                            }
                            spdFile->setTemporalBinSpacing(txtUtils.strtodouble(tokens->at(1)));
                            haveTimeInterval = true;
                        }
                        else if(tokens->at(0) == "laser_wavelength(nm)")
                        {
                            if(tokens->size() != 2)
                            {
                                throw SPDIOException("Failed to parser header line \'laser_wavelength(ns)\'.");
                            }

                            laserWavelength = txtUtils.strtofloat(tokens->at(1));
                            std::vector<float> wavelengths;
                            wavelengths.push_back(laserWavelength);
                            spdFile->setWavelengths(wavelengths);
                            spdFile->setNumOfWavelengths(1);
                            haveWavelengths = true;
                        }
                        else if(tokens->at(0) == "laser_bandwidth(nm)")
                        {
                            if(tokens->size() != 2)
                            {
                                throw SPDIOException("Failed to parser header line \'laser_bandwidth(ns)\'.");
                            }

                            std::vector<float> bandWidths;
                            bandWidths.push_back(txtUtils.strtofloat(tokens->at(1)));
                            spdFile->setBandwidths(bandWidths);
                            haveBandWidths = true;
                        }
                        else if(tokens->at(0) == "trans_waveform_thershold")
                        {
                            if(tokens->size() != 2)
                            {
                                throw SPDIOException("Failed to parser header line \'trans_waveform_thershold\'.");
                            }
                            transWaveformThershold = txtUtils.strtofloat(tokens->at(1));
                            haveTThreshold = true;
                        }
                        else if(tokens->at(0) == "trans_waveform_gain")
                        {
                            if(tokens->size() != 2)
                            {
                                throw SPDIOException("Failed to parser header line \'trans_waveform_gain\'.");
                            }
                            transWaveformGain = txtUtils.strtofloat(tokens->at(1));
                            haveTGain = true;
                        }
                        else if(tokens->at(0) == "trans_waveform_offset")
                        {
                            if(tokens->size() != 2)
                            {
                                throw SPDIOException("Failed to parser header line \'trans_waveform_offset\'.");
                            }
                            transWaveformOffset = txtUtils.strtofloat(tokens->at(1));
                            haveTOffset = true;
                        }
                        else if(tokens->at(0) == "recieve_waveform_thershold")
                        {
                            if(tokens->size() != 2)
                            {
                                throw SPDIOException("Failed to parser header line \'waveform_thershold\'.");
                            }
                            receivedWaveformThershold = txtUtils.strtofloat(tokens->at(1));
                            haveRThreshold = true;
                        }
                        else if(tokens->at(0) == "recieve_waveform_gain")
                        {
                            if(tokens->size() != 2)
                            {
                                throw SPDIOException("Failed to parser header line \'waveform_gain\'.");
                            }
                            receivedWaveformGain = txtUtils.strtofloat(tokens->at(1));
                            haveRGain = true;
                        }
                        else if(tokens->at(0) == "recieve_waveform_offset")
                        {
                            if(tokens->size() != 2)
                            {
                                throw SPDIOException("Failed to parser header line \'waveform_offset\'.");
                            }
                            receivedWaveformOffset = txtUtils.strtofloat(tokens->at(1));
                            haveROffset = true;
                        }
                        else if(tokens->at(0) == "aquasition_date")
                        {
                            if(tokens->size() != 2)
                            {
                                throw SPDIOException("Failed to parser header line \'aquasition_date\'.");
                            }

                            std::string dateStr = tokens->at(1);

                            tokens->clear();
                            txtUtils.tokenizeString(dateStr, '-', tokens, true);

                            if(tokens->size() != 3)
                            {
                                std::cout << dateStr << std::endl;
                                throw SPDIOException("Failed to parser header line \'aquasition_date\', must have format YYYY-MM-DD.");
                            }

                            spdFile->setYearOfCapture(txtUtils.strto16bitUInt(tokens->at(0)));
                            spdFile->setMonthOfCapture(txtUtils.strto16bitUInt(tokens->at(1)));
                            spdFile->setDayOfCapture(txtUtils.strto16bitUInt(tokens->at(2)));

                            haveAquDate = true;
                        }

                        else if(tokens->at(0) == "aquasition_time")
                        {
                            if(tokens->size() != 2)
                            {
                                throw SPDIOException("Failed to parser header line \'aquasition_time\'.");
                            }
                            std::string timeStr = tokens->at(1);
                            tokens->clear();
                            txtUtils.tokenizeString(timeStr, ':', tokens, true);

                            if(tokens->size() != 3)
                            {
                                std::cout << timeStr << std::endl;
                                throw SPDIOException("Failed to parser header line \'aquasition_time\', must have format HH:MM:SS.");
                            }

                            spdFile->setHourOfCapture(txtUtils.strto16bitUInt(tokens->at(0)));
                            spdFile->setMinuteOfCapture(txtUtils.strto16bitUInt(tokens->at(1)));
                            spdFile->setSecondOfCapture(txtUtils.strto16bitUInt(tokens->at(2)));

                            haveAquTime = true;
                        }
                        else
                        {
                            std::cout << line << std::endl;

                            throw SPDIOException("Could not parser the header.");
                        }
                    }
                }
                delete tokens;
                lineReader.closeFile();

                if(!haveSensorFile)
                {
                    throw SPDIOException("The 'sensor' file must be specified.");
                }
                if(!haveWaveformFile)
                {
                    throw SPDIOException("The 'waveforms' file must be specified.");
                }

                if(!haveTimeInterval)
                {
                    spdFile->setTemporalBinSpacing(1.0);
                }
                if(!haveWavelengths)
                {
                    laserWavelength = 1064.0;
                    std::vector<float> wavelengths;
                    wavelengths.push_back(1064.0);
                    spdFile->setWavelengths(wavelengths);
                    spdFile->setNumOfWavelengths(1);
                }
                if(!haveBandWidths)
                {
                    std::vector<float> bandWidths;
                    bandWidths.push_back(1.0);
                    spdFile->setBandwidths(bandWidths);
                }
                if(!haveTThreshold)
                {
                    transWaveformThershold = this->waveNoiseThreshold;
                }
                if(!haveTGain)
                {
                    transWaveformGain = 1;
                }
                if(!haveTOffset)
                {
                    transWaveformOffset = 0;
                }
                if(!haveRThreshold)
                {
                    receivedWaveformThershold = this->waveNoiseThreshold;
                }
                if(!haveRGain)
                {
                    receivedWaveformGain = 1;
                }
                if(!haveROffset)
                {
                    receivedWaveformOffset = 0;
                }
            }
            else
            {
                throw SPDIOException("Could not find the input header file, please check the file path.");
            }
        }
        catch(SPDIOException &e)
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDIOException(e.what());
        }
        catch(std::exception &e)
        {
            throw SPDIOException(e.what());
        }
    }

    SPDOptechFullWaveformASCIIImport::~SPDOptechFullWaveformASCIIImport()
    {

    }

}

