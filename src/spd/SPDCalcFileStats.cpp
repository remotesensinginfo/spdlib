/*
 *  SPDCalcFileStats.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 28/03/2012.
 *  Copyright 2012 SPDLib. All rights reserved.
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

#include "spd/SPDCalcFileStats.h"


namespace spdlib
{	

    SPDCalcFileStats::SPDCalcFileStats()
    {

    }

    void SPDCalcFileStats::calcImagePulsePointDensity(std::string inputSPDFile, std::string outputImageFile, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize, float processingResolution, std::string gdalFormat) throw(SPDProcessingException)
    {
        try
        {
            SPDFile *spdInFile = new SPDFile(inputSPDFile);
            SPDPulseProcessor *pulseStatsProcessor = new SPDPulseProcessorCalcStats();
            SPDSetupProcessPulses processPulses = SPDSetupProcessPulses(blockXSize, blockYSize, true);
            processPulses.processPulsesWithOutputImage(pulseStatsProcessor, spdInFile, outputImageFile, 2, processingResolution, gdalFormat, false, 0);

            delete spdInFile;
            delete pulseStatsProcessor;
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }

    void SPDCalcFileStats::calcOverallPulsePointDensityStats(std::string inputSPDFile, std::string outputTextFile, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize, float processingResolution) throw(SPDProcessingException)
    {
        try
        {
            SPDFile *spdInFile = new SPDFile(inputSPDFile);
            SPDPulseProcessorCalcStats *pulseStatsProcessor = new SPDPulseProcessorCalcStats();
            SPDSetupProcessPulses processPulses = SPDSetupProcessPulses(blockXSize, blockYSize, true);
            processPulses.processPulses(pulseStatsProcessor, spdInFile, processingResolution, false, 0);

            boost::uint_fast32_t binCount = pulseStatsProcessor->getBinCount();
            float meanPulses = pulseStatsProcessor->getMeanPulses();
            boost::uint_fast32_t minPulses = pulseStatsProcessor->getMinPulses();
            boost::uint_fast32_t maxPulses = pulseStatsProcessor->getMaxPulses();
            float meanPoints = pulseStatsProcessor->getMeanPoints();
            boost::uint_fast32_t minPoints = pulseStatsProcessor->getMinPoints();
            boost::uint_fast32_t maxPoints = pulseStatsProcessor->getMaxPoints();

            pulseStatsProcessor->setCalcStdDev(meanPulses, meanPoints);
            processPulses.processPulses(pulseStatsProcessor, spdInFile, processingResolution, false, 0);
            float stdDevPulses = pulseStatsProcessor->getStdDevPulses();
            float stdDevPoints = pulseStatsProcessor->getStdDevPoints();

            delete spdInFile;
            delete pulseStatsProcessor;

            std::ofstream outTxtFile;
            outTxtFile.open(outputTextFile.c_str(), std::ios::out | std::ios::trunc);
            outTxtFile << "Num Bins: " << binCount << std::endl;
            outTxtFile << "#Pulses" << std::endl;
            outTxtFile << "Min Pulses: " << minPulses << std::endl;
            outTxtFile << "Max Pulses: " << maxPulses << std::endl;
            outTxtFile << "Mean Pulses: " << meanPulses << std::endl;
            outTxtFile << "Std Dev Pulses: " << stdDevPulses << std::endl;
            outTxtFile << "#Points" << std::endl;
            outTxtFile << "Min Points: " << minPoints << std::endl;
            outTxtFile << "Max Points: " << maxPoints << std::endl;
            outTxtFile << "Mean Points: " << meanPoints << std::endl;
            outTxtFile << "Std Dev Points: " << stdDevPoints << std::endl;
            outTxtFile.flush();
            outTxtFile.close();
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }
		
    SPDCalcFileStats::~SPDCalcFileStats()
    {

    }




    SPDPulseProcessorCalcStats::SPDPulseProcessorCalcStats():SPDPulseProcessor()
    {
        calcStdDevVals = false;
        first = true;
        countBins = 0;

        sumPulses = 0;
        minPulses = 0;
        maxPulses = 0;
        meanPulses = 0;
        sqDiffPulses = 0;

        sumPoints = 0;
        minPoints = 0;
        maxPoints = 0;
        meanPoints = 0;
        sqDiffPoints = 0;
    }

    void SPDPulseProcessorCalcStats::processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
    {
        try
        {
            if(numImgBands < 2)
            {
                throw SPDProcessingException("Processing requires at least 2 image bands.");
            }

            imageData[0] = pulses->size();

            boost::uint_fast32_t ptsCount = 0;
            for(std::vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
            {
                ptsCount += (*iterPulses)->numberOfReturns;
            }

            imageData[1] = ptsCount;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
    }
		
    void SPDPulseProcessorCalcStats::processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException)
    {
        try
        {
            if(pulses->size() > 0)
            {
                boost::uint_fast32_t ptsCount = 0;
                for(std::vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
                {
                    ptsCount += (*iterPulses)->numberOfReturns;
                }

                if(!calcStdDevVals)
                {
                    if(first)
                    {
                        minPulses = pulses->size();
                        maxPulses = pulses->size();
                        sumPulses = pulses->size();

                        minPoints = ptsCount;
                        maxPoints = ptsCount;
                        sumPoints = ptsCount;
                        first = false;
                    }
                    else
                    {
                        if(pulses->size() < minPulses)
                        {
                            minPoints = pulses->size();
                        }
                        else if(pulses->size() > maxPulses)
                        {
                            maxPulses = pulses->size();
                        }
                        sumPulses += pulses->size();

                        if(ptsCount < minPoints)
                        {
                            minPoints = ptsCount;
                        }
                        else if(ptsCount > maxPoints)
                        {
                            maxPoints = ptsCount;
                        }
                        sumPoints += ptsCount;
                    }
                    ++countBins;
                }
                else
                {
                    if(first)
                    {
                        sqDiffPulses = (((double)pulses->size())-meanPulses)*(((double)pulses->size())-meanPulses);
                        sqDiffPoints = (((double)ptsCount)-meanPoints)*(((double)ptsCount)-meanPoints);
                        first = false;
                    }
                    else
                    {
                        sqDiffPulses += (((double)pulses->size())-meanPulses)*(((double)pulses->size())-meanPulses);
                        sqDiffPoints += (((double)ptsCount)-meanPoints)*(((double)ptsCount)-meanPoints);
                    }
                }
            }

        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
    }

    std::vector<std::string> SPDPulseProcessorCalcStats::getImageBandDescriptions() throw(SPDProcessingException)
    {
        std::vector<std::string> bandNames;
        bandNames.push_back("Pulse Density");
        bandNames.push_back("Point Density");

        return bandNames;
    }

    void SPDPulseProcessorCalcStats::setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
    {
        // NOTHING TO DO HERE...
    }

    SPDPulseProcessorCalcStats::~SPDPulseProcessorCalcStats()
    {

    }

}




