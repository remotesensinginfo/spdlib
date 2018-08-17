/*
 *  SPDProcessPulses.cpp
 *
 *  Created by Pete Bunting on 27/03/2012.
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

#include "spd/SPDProcessPulses.h"

namespace spdlib
{
    SPDSetupProcessPulses::SPDSetupProcessPulses(boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize, bool printProgress)
    {
        this->blockXSize = blockXSize;
        this->blockYSize = blockYSize;
        this->printProgress = printProgress;
    }

    void SPDSetupProcessPulses::processPulsesWithInputImage(SPDPulseProcessor *pulseProcessor, SPDFile *spdInFile, std::string outFile, std::string imageFilePath, bool usingWindow, boost::uint_fast16_t winHSize) throw(SPDProcessingException)
    {
        try
        {
            SPDProcessPulses *processPulses = new SPDProcessPulses(pulseProcessor, usingWindow, winHSize);
            if(!usingWindow)
            {
                winHSize = 0;
            }
            SPDProcessDataBlocks *processDataBlocks = new SPDProcessDataBlocks(processPulses, winHSize, blockXSize, blockYSize, printProgress);
            processDataBlocks->processDataBlocksGridPulsesInputImage(spdInFile, outFile, imageFilePath);

            delete processDataBlocks;
            delete processPulses;
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }

    void SPDSetupProcessPulses::processPulsesWithOutputImage(SPDPulseProcessor *pulseProcessor, SPDFile *spdInFile, std::string outImagePath, boost::uint_fast16_t numImgBands, float processingResolution, std::string gdalFormat, bool usingWindow, boost::uint_fast16_t winHSize) throw(SPDProcessingException)
    {
        try
        {
            SPDProcessPulses *processPulses = new SPDProcessPulses(pulseProcessor, usingWindow, winHSize);
            if(!usingWindow)
            {
                winHSize = 0;
            }
            SPDProcessDataBlocks *processDataBlocks = new SPDProcessDataBlocks(processPulses, winHSize, blockXSize, blockYSize, printProgress);
            processDataBlocks->processDataBlocksGridPulsesOutputImage(spdInFile, outImagePath, processingResolution, numImgBands, gdalFormat);

            delete processDataBlocks;
            delete processPulses;
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }

    void SPDSetupProcessPulses::processPulsesWithOutputSPD(SPDPulseProcessor *pulseProcessor, SPDFile *spdInFile, std::string outFile, float processingResolution, bool usingWindow, boost::uint_fast16_t winHSize) throw(SPDProcessingException)
    {
        try
        {
            SPDProcessPulses *processPulses = new SPDProcessPulses(pulseProcessor, usingWindow, winHSize);
            if(!usingWindow)
            {
                winHSize = 0;
            }
            SPDProcessDataBlocks *processDataBlocks = new SPDProcessDataBlocks(processPulses, winHSize, blockXSize, blockYSize, printProgress);
            processDataBlocks->processDataBlocksGridPulsesOutputSPD(spdInFile, outFile, processingResolution);

            delete processDataBlocks;
            delete processPulses;
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }

    void SPDSetupProcessPulses::processPulses(SPDPulseProcessor *pulseProcessor, SPDFile *spdInFile, float processingResolution, bool usingWindow, boost::uint_fast16_t winHSize) throw(SPDProcessingException)
    {
        try
        {
            SPDProcessPulses *processPulses = new SPDProcessPulses(pulseProcessor, usingWindow, winHSize);
            if(!usingWindow)
            {
                winHSize = 0;
            }
            SPDProcessDataBlocks *processDataBlocks = new SPDProcessDataBlocks(processPulses, winHSize, blockXSize, blockYSize, printProgress);
            processDataBlocks->processDataBlocksGridPulses(spdInFile, processingResolution);

            delete processDataBlocks;
            delete processPulses;
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }

    SPDSetupProcessPulses::~SPDSetupProcessPulses()
    {

    }






    SPDProcessPulses::SPDProcessPulses(SPDPulseProcessor *pulseProcessor, bool usingWindow, boost::uint_fast16_t winHSize):SPDDataBlockProcessor()
    {
        this->pulseProcessor = pulseProcessor;
        this->usingWindow = usingWindow;
        this->winHSize = winHSize;
    }


    void SPDProcessPulses::processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
    {
        try
        {
            if(!usingWindow)
            {
                for(boost::uint_fast32_t y = 0; y < ySize; ++y)
                {
                    for(boost::uint_fast32_t x = 0; x < xSize; ++x)
                    {
                        pulseProcessor->processDataColumnImage(inSPDFile, pulses[y][x], imageDataBlock[y][x], cenPts[y][x], numImgBands, binSize);
                    }
                }
            }
            else
            {
                boost::uint_fast16_t winSize = (winHSize * 2) + 1;

                bool **validBins = new bool*[winSize];
                float ***winImgDataBlock = new float**[winSize];
                SPDXYPoint ***winCenPts = new SPDXYPoint**[winSize];
                std::vector<SPDPulse*> ***winPulses = new std::vector<SPDPulse*>**[winSize];
                for(boost::uint_fast32_t i = 0; i < winSize; ++i)
                {
                    winPulses[i] = new std::vector<SPDPulse*>*[winSize];
                    winCenPts[i] = new SPDXYPoint*[winSize];
                    winImgDataBlock[i] = new float*[winSize];
                    validBins[i] = new bool[winSize];
                }

                boost::int_fast32_t xMinPos = 0;
                boost::int_fast32_t yMinPos = 0;
                boost::int_fast32_t xMaxPos = 0;
                boost::int_fast32_t yMaxPos = 0;

                boost::uint_fast32_t idxIOff = 0; // i index (i.e., Y)
                boost::uint_fast32_t idxJOff = 0; // j index (i.e., X)

                for(boost::uint_fast32_t y = 0; y < ySize; ++y)
                {
                    for(boost::uint_fast32_t x = 0; x < xSize; ++x)
                    {
                        // Reset the arrays.
                        for(boost::uint_fast32_t i = 0; i < winSize; ++i)
                        {
                            for(boost::uint_fast32_t j = 0; j < winSize; ++j)
                            {
                                winPulses[i][j] = NULL;
                                winCenPts[i][j] = NULL;
                                winImgDataBlock[i][j] = NULL;
                                validBins[i][j] = false;
                            }
                        }
                        xMinPos = x - winHSize;
                        yMinPos = y - winHSize;
                        xMaxPos = x + winHSize;
                        yMaxPos = y + winHSize;

                        if(xMinPos < 0)
                        {
                            xMinPos = 0;
                            idxJOff = (xMinPos * (-1));
                        }
                        if(yMinPos < 0)
                        {
                            yMinPos = 0;
                            idxIOff = (yMinPos * (-1));
                        }
                        if(xMaxPos >= xSize)
                        {
                            xMaxPos = xSize-1;
                        }
                        if(yMaxPos >= ySize)
                        {
                            yMaxPos = ySize-1;
                        }

                        for(boost::uint_fast32_t yIdx = yMinPos, i = idxIOff; yIdx < yMaxPos; ++yIdx, ++i)
                        {
                            for(boost::uint_fast32_t xIdx = xMinPos, j = idxJOff; xIdx < xMaxPos; ++xIdx, ++j)
                            {
                                winPulses[i][j] = pulses[yIdx][xIdx];
                                winCenPts[i][j] = cenPts[yIdx][xIdx];
                                winImgDataBlock[i][j] = imageDataBlock[yIdx][xIdx];
                                validBins[i][j] = true;
                            }
                        }


                        pulseProcessor->processDataWindowImage(inSPDFile, validBins, winPulses, winImgDataBlock, winCenPts, numImgBands, binSize, winSize);
                    }
                }


                for(boost::uint_fast32_t i = 0; i < winSize; ++i)
                {
                    delete[] winPulses[i];
                    delete[] winCenPts[i];
                    delete[] winImgDataBlock[i];
                    delete[] validBins[i];
                }
                delete[] winImgDataBlock;
                delete[] winCenPts;
                delete[] winPulses;
                delete[] validBins;
            }
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }

    void SPDProcessPulses::processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException)
    {
        try
        {
            if(!usingWindow)
            {
                for(boost::uint_fast32_t y = 0; y < ySize; ++y)
                {
                    for(boost::uint_fast32_t x = 0; x < xSize; ++x)
                    {
                        pulseProcessor->processDataColumn(inSPDFile, pulses[y][x], cenPts[y][x]);
                    }
                }
            }
            else
            {
                boost::int_fast16_t winSize = (winHSize * 2) + 1;

                bool **validBins = new bool*[winSize];
                SPDXYPoint ***winCenPts = new SPDXYPoint**[winSize];
                std::vector<SPDPulse*> ***winPulses = new std::vector<SPDPulse*>**[winSize];
                for(boost::uint_fast32_t i = 0; i < winSize; ++i)
                {
                    winPulses[i] = new std::vector<SPDPulse*>*[winSize];
                    winCenPts[i] = new SPDXYPoint*[winSize];
                    validBins[i] = new bool[winSize];
                }

                boost::int_fast32_t xMinPos = 0;
                boost::int_fast32_t yMinPos = 0;
                boost::int_fast32_t xMaxPos = 0;
                boost::int_fast32_t yMaxPos = 0;

                boost::uint_fast32_t idxIOff = 0; // i index (i.e., Y)
                boost::uint_fast32_t idxJOff = 0; // j index (i.e., X)

                for(boost::int_fast32_t y = 0; y < ySize; ++y)
                {
                    for(boost::int_fast32_t x = 0; x < xSize; ++x)
                    {
                        // Reset the arrays.
                        for(boost::uint_fast32_t i = 0; i < winSize; ++i)
                        {
                            for(boost::uint_fast32_t j = 0; j < winSize; ++j)
                            {
                                winPulses[i][j] = NULL;
                                winCenPts[i][j] = NULL;
                                validBins[i][j] = false;
                            }
                        }
                        //std::cout << "x = " << x << " y = " << y << std::endl;
                        xMinPos = x - winHSize;
                        yMinPos = y - winHSize;
                        xMaxPos = x + winHSize;
                        yMaxPos = y + winHSize;
                        idxIOff = 0;
                        idxJOff = 0;
                        //std::cout << "(B4) Min Pos: [" << xMinPos << " , " << yMinPos << "]\n";
                        //std::cout << "(B4) Max Pos: [" << xMaxPos << " , " << yMaxPos << "]\n";
                        if(xMinPos < 0)
                        {
                            idxJOff = (xMinPos * (-1));
                            xMinPos = 0;
                        }
                        if(yMinPos < 0)
                        {
                            idxIOff = (yMinPos * (-1));
                            yMinPos = 0;
                        }
                        if(xMaxPos >= xSize)
                        {
                            xMaxPos = xSize;
                        }
                        if(yMaxPos >= ySize)
                        {
                            yMaxPos = ySize;
                        }

                        //std::cout << "Array Off: [" << idxJOff << " , " << idxIOff << "]\n";
                        //std::cout << "Min Pos: [" << xMinPos << " , " << yMinPos << "]\n";
                        //std::cout << "Max Pos: [" << xMaxPos << " , " << yMaxPos << "]\n";

                        for(boost::uint_fast32_t yIdx = yMinPos, i = idxIOff; yIdx < yMaxPos; ++yIdx, ++i)
                        {
                            for(boost::uint_fast32_t xIdx = xMinPos, j = idxJOff; xIdx < xMaxPos; ++xIdx, ++j)
                            {
                                winPulses[i][j] = pulses[yIdx][xIdx];
                                winCenPts[i][j] = cenPts[yIdx][xIdx];
                                validBins[i][j] = true;
                            }
                        }

                        pulseProcessor->processDataWindow(inSPDFile, validBins, winPulses, winCenPts, winSize);
                    }
                }


                for(boost::uint_fast32_t i = 0; i < winSize; ++i)
                {
                    delete[] winPulses[i];
                    delete[] winCenPts[i];
                    delete[] validBins[i];
                }
                delete[] winCenPts;
                delete[] winPulses;
                delete[] validBins;
            }
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }

    void SPDProcessPulses::processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) throw(SPDProcessingException)
    {
        throw SPDProcessingException("processDataBlockImage without pulses grid is not implemented.");
    }

    void SPDProcessPulses::processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses) throw(SPDProcessingException)
    {
        throw SPDProcessingException("processDataBlock without pulses grid is not implemented.");
    }

    std::vector<std::string> SPDProcessPulses::getImageBandDescriptions() throw(SPDProcessingException)
    {
        try
        {
            return pulseProcessor->getImageBandDescriptions();
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }

    void SPDProcessPulses::setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
    {
        try
        {
            pulseProcessor->setHeaderValues(spdFile);
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }

    SPDProcessPulses::~SPDProcessPulses()
    {

    }

}



