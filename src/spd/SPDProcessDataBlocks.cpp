/*
 *  SPDProcessDataBlocks.cpp
 *
 *  Created by Pete Bunting on 11/03/2012.
 *  Copyright 2012 RSGISLib. All rights reserved.
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

#include "spd/SPDProcessDataBlocks.h"


namespace spdlib
{

    SPDProcessDataBlocks::SPDProcessDataBlocks(SPDDataBlockProcessor *dataBlockProcessor, boost::uint_fast32_t overlap, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize, bool printProgress, bool keepMinExtent): dataBlockProcessor(NULL), overlap(0), blockXSize(0), blockYSize(0), printProgress(true)

    {
        this->dataBlockProcessor = dataBlockProcessor;
        this->overlap = overlap;
        this->blockXSize = blockXSize;
        this->blockYSize = blockYSize;
        this->printProgress = printProgress;
        this->keepMinExtent = keepMinExtent;
    }

    SPDProcessDataBlocks::SPDProcessDataBlocks(const SPDProcessDataBlocks &processDataBlock): dataBlockProcessor(NULL), overlap(0), blockXSize(0), blockYSize(0), printProgress(true)
    {
        this->dataBlockProcessor = processDataBlock.dataBlockProcessor;
        this->overlap = processDataBlock.overlap;
        this->blockXSize = processDataBlock.blockXSize;
        this->blockYSize = processDataBlock.blockYSize;
        this->printProgress = processDataBlock.printProgress;
    }

    SPDProcessDataBlocks& SPDProcessDataBlocks::operator=(const SPDProcessDataBlocks& processDataBlock)
    {
        this->dataBlockProcessor = processDataBlock.dataBlockProcessor;
        this->overlap = processDataBlock.overlap;
        this->blockXSize = processDataBlock.blockXSize;
        this->blockYSize = processDataBlock.blockYSize;
        this->printProgress = processDataBlock.printProgress;
		return *this;
    }

    void SPDProcessDataBlocks::processDataBlocksGridPulsesInputImage(SPDFile *spdInFile, std::string outFile, std::string imageFilePath) throw(SPDProcessingException)
    {
        try
        {
            GDALAllRegister();
            SPDFileReader reader;
            reader.readHeaderInfo(spdInFile->getFilePath(), spdInFile);
            
            if(spdInFile->getIndexType() == SPD_UPD_TYPE)
            {
                throw SPDProcessingException("The SPD file must have a spatial index. Use spdtranslate.");
            }
            
            if(this->blockXSize == 0)
            {
                this->blockXSize = spdInFile->getNumberBinsX();
            }
            
            if((this->overlap > this->blockXSize) | (this->overlap > this->blockYSize))
            {
                throw SPDProcessingException("The overlap must be smaller than the block size in both axis\'");
            }
            
            GDALDataset *gdalDataset = (GDALDataset *) GDALOpenShared(imageFilePath.c_str(), GA_ReadOnly);
            if(gdalDataset == NULL)
            {
                std::string message = std::string("Could not open image ") + imageFilePath;
                throw SPDProcessingException(message);
            }
            double *geoTrans = new double[6];
            gdalDataset->GetGeoTransform(geoTrans);

            double imgXOrigin = geoTrans[0];
            double imgYOrigin = geoTrans[3];
            float processingResolution = geoTrans[1];
            delete[] geoTrans;
            boost::uint_fast16_t numImgBands = gdalDataset->GetRasterCount();
            GDALRasterBand **imgBands = new GDALRasterBand*[numImgBands];
            for(boost::uint_fast16_t i = 0; i < numImgBands; ++i)
            {
                imgBands[i] = gdalDataset->GetRasterBand(i+1);
            }

            SPDGridData gridData;
            
            SPDFile *spdOutFile = new SPDFile(outFile);
            spdOutFile->copyAttributesFrom(spdInFile);

            boost::uint_fast64_t nativeXBins = spdInFile->getNumberBinsX();
            boost::uint_fast64_t nativeYBins = spdInFile->getNumberBinsY();

            boost::uint_fast64_t procResXBins = 0;
            boost::uint_fast64_t procResYBins = 0;

            double geoWidth = 0;
            double geoHeight = 0;

            if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
            {
                geoWidth = spdInFile->getXMax() - spdInFile->getXMin();
                geoHeight = spdInFile->getYMax() - spdInFile->getYMin();
            }
            else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
            {
                geoWidth = spdInFile->getAzimuthMax() - spdInFile->getAzimuthMin();
                geoHeight = spdInFile->getZenithMax() - spdInFile->getZenithMin();
            }
            else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
            {
                geoWidth = spdInFile->getScanlineIdxMax() - spdInFile->getScanlineIdxMin();
                geoHeight = spdInFile->getScanlineMax() - spdInFile->getScanlineMin();
            }
            //std::cout << "Geo: [" << geoWidth << "," << geoHeight << "]\n";

            bool usingNativeRes = false;
            bool scaleDown = true;
            boost::uint_fast32_t binScaling = 0;
            if(processingResolution == spdInFile->getBinSize())
            {
                usingNativeRes = true;
                procResXBins = spdInFile->getNumberBinsX();
                procResYBins = spdInFile->getNumberBinsY();
            }
            else if(processingResolution > spdInFile->getBinSize())
            {
                if(fmod(processingResolution, spdInFile->getBinSize()) != 0)
                {
                    std::cerr << "Native Res: " << spdInFile->getBinSize() << std::endl;
                    std::cerr << "Process Res: " << processingResolution << std::endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resoltion.");
                }
                usingNativeRes = false;
                procResXBins = boost::numeric_cast<boost::uint_fast64_t>(geoWidth / processingResolution)+1;
                procResYBins = boost::numeric_cast<boost::uint_fast64_t>(geoHeight / processingResolution)+1;

                scaleDown = false;
                binScaling = boost::numeric_cast<boost::uint_fast32_t>(processingResolution/spdInFile->getBinSize());
            }
            else
            {
                float tmpNumOutBins = spdInFile->getBinSize()/processingResolution;
                if(fmod(tmpNumOutBins, ((float)1.0)) != 0)
                {
                    std::cerr << "Native Res: " << spdInFile->getBinSize() << std::endl;
                    std::cerr << "Process Res: " << processingResolution << std::endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resolution.");
                }
                usingNativeRes = false;
                procResXBins = boost::numeric_cast<boost::uint_fast64_t>(geoWidth / processingResolution)+1;
                procResYBins = boost::numeric_cast<boost::uint_fast64_t>(geoHeight / processingResolution)+1;

                scaleDown = true;
                binScaling = boost::numeric_cast<boost::uint_fast32_t>(spdInFile->getBinSize()/processingResolution);
            }

            //std::cout << "Bin Scaling: " << binScaling << std::endl;
            //std::cout << "Process Bins: [" << procResXBins << "," << procResYBins << "]\n";

            if(usingNativeRes)
            {
                std::cout << "Using native resolution for processing\n";
            }
            else
            {
                std::cout << "Resampling native resolution\n";
                if(scaleDown)
                {
                    std::cout << "Scaling down\n";
                }
                else
                {
                    std::cout << "Scaling up\n";
                }
            }

            boost::uint_fast32_t numXFullBlocks = floor(((double)nativeXBins)/this->blockXSize);
            boost::uint_fast32_t numYFullBlocks = floor(((double)nativeYBins)/this->blockYSize);

            //std::cout << "Number of full blocks: [" << numXFullBlocks << "," << numYFullBlocks << "]\n";

            boost::uint_fast32_t remainingCols = nativeXBins - (numXFullBlocks * this->blockXSize);
            boost::uint_fast32_t remainingRows = nativeYBins - (numYFullBlocks * this->blockYSize);

            //std::cout << "Remainder: [" << remainingCols << "," << remainingRows << "]\n";

            double blockMinX = 0;
            double blockMaxX = 0;
            double blockMinY = 0;
            double blockMaxY = 0;

            double blockWidth = blockXSize * spdInFile->getBinSize();
            double blockHeight = blockYSize * spdInFile->getBinSize();

            boost::uint_fast32_t procResXBlockSize = 0;
            boost::uint_fast32_t procResYBlockSize = 0;

            if(binScaling == 0)
            {
                procResXBlockSize = blockXSize;
                procResYBlockSize = blockYSize;
            }
            else if(scaleDown)
            {
                procResXBlockSize = ceil(((double)blockXSize) * binScaling);
                procResYBlockSize = ceil(((double)blockYSize) * binScaling);
            }
            else
            {
                procResXBlockSize = ceil(((double)blockXSize) / binScaling);
                procResYBlockSize = ceil(((double)blockYSize) / binScaling);
            }

            //std::cout << "Native block size: [" << this->blockXSize << "," << this->blockXSize << "]\n";
            //std::cout << "Process block size: [" << procResXBlockSize << "," << procResYBlockSize << "]\n";

            //std::cout << "Block Size: [" << blockWidth << "," << blockHeight << "]\n";

            boost::uint_fast32_t numBlocks = numYFullBlocks * numXFullBlocks;
            if(remainingCols > 0)
            {
                numBlocks += numYFullBlocks;
            }
            if(remainingRows > 0)
            {
                numBlocks += numXFullBlocks;
                if(remainingCols > 0)
                {
                    numBlocks += 1;
                }
            }
            boost::uint_fast32_t cBlocksIdx = 1;

            blockMinY = 0;
            blockMaxY = blockYSize;

            SPDFileIncrementalReader incReader;
            incReader.open(spdInFile);

            SPDDataExporter *fileWriter = NULL;

            if(this->blockXSize == spdInFile->getNumberBinsX())
            {
                fileWriter = new SPDSeqFileWriter();
            }
            else
            {
                fileWriter = new SPDNonSeqFileWriter();
            }
            fileWriter->open(spdOutFile, outFile);
            fileWriter->setKeepMinExtent(keepMinExtent);
            
            boost::uint_fast32_t pulsesBlockSizeX = this->blockXSize + (2 * this->overlap);
            boost::uint_fast32_t pulsesBlockSizeY = this->blockYSize + (2 * this->overlap);

            boost::uint_fast32_t scaledOverlap = 0;
            if(binScaling != 0)
            {
                if(scaleDown)
                {
                    scaledOverlap = ceil(((double)this->overlap) * binScaling);
                }
                else
                {
                    scaledOverlap = ceil(((double)this->overlap) / binScaling);
                }
            }

            boost::uint_fast32_t pulsesScaledBlockSizeX = procResXBlockSize + (2 * scaledOverlap);
            boost::uint_fast32_t pulsesScaledBlockSizeY = procResYBlockSize + (2 * scaledOverlap);

            std::vector<SPDPulse*> ***pulses = new std::vector<SPDPulse*>**[pulsesBlockSizeY];
            for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
            {
                pulses[i] = new std::vector<SPDPulse*>*[pulsesBlockSizeX];
                for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                {
                    pulses[i][j] = new std::vector<SPDPulse*>();
                }
            }


            SPDXYPoint ***cenPts = NULL;
            std::vector<SPDPulse*> ***pulseScaled = NULL;
            float ***imageBlockVals = NULL;
            if(binScaling != 0)
            {
                pulseScaled = new std::vector<SPDPulse*>**[pulsesScaledBlockSizeY];
                cenPts = new SPDXYPoint**[pulsesScaledBlockSizeY];
                imageBlockVals = new float**[pulsesScaledBlockSizeY];
                for(boost::uint_fast32_t i = 0; i < pulsesScaledBlockSizeY; ++i)
                {
                    pulseScaled[i] = new std::vector<SPDPulse*>*[pulsesScaledBlockSizeX];
                    cenPts[i] = new SPDXYPoint*[pulsesScaledBlockSizeX];
                    imageBlockVals[i] = new float*[pulsesScaledBlockSizeX];
                    for(boost::uint_fast32_t j = 0; j < pulsesScaledBlockSizeX; ++j)
                    {
                        pulseScaled[i][j] = new std::vector<SPDPulse*>();
                        cenPts[i][j] = new SPDXYPoint();
                        imageBlockVals[i][j] = new float[numImgBands];
                    }
                }
            }
            else
            {
                cenPts = new SPDXYPoint**[pulsesBlockSizeY];
                imageBlockVals = new float**[pulsesBlockSizeY];
                for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
                {
                    cenPts[i] = new SPDXYPoint*[pulsesBlockSizeX];
                    imageBlockVals[i] = new float*[pulsesBlockSizeX];
                    for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                    {
                        cenPts[i][j] = new SPDXYPoint();
                        imageBlockVals[i][j] = new float[numImgBands];
                    }
                }
            }

            boost::uint_fast32_t *bbox = new boost::uint_fast32_t[4];

            boost::uint_fast32_t xOffset = 0;
            boost::uint_fast32_t yOffset = 0;

            double blockXOrigin = 0;
            double blockYOrigin = 0;

            if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
            {
                blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getYMax() + (overlap * spdInFile->getBinSize());
            }
            else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
            {
                blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getZenithMin() - (overlap * spdInFile->getBinSize());
            }
            else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
            {
                blockXOrigin = spdInFile->getScanlineIdxMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getScanlineMin() - (overlap * spdInFile->getBinSize());
            }
            
            for(boost::uint_fast32_t i = 0; i < numYFullBlocks; ++i)
            {
                blockMinX = 0;
                blockMaxX = blockXSize;

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
                {
                    blockXOrigin = spdInFile->getScanlineIdxMin() - (overlap * spdInFile->getBinSize());
                }

                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }

                    //std::cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //std::cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin <<"]\n";

                    //std::cout << "Block Size [" << blockXSize << "," << blockYSize << "]\n";

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);

                    if(binScaling == 0)
                    {
                        this->populateFromImage(imageBlockVals, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands, imgBands, imgXOrigin, imgYOrigin, processingResolution, blockXOrigin, blockYOrigin);
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulses, imageBlockVals, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, this->blockXSize, this->blockYSize, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateFromImage(imageBlockVals, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands, imgBands, imgXOrigin, imgYOrigin, processingResolution, blockXOrigin, blockYOrigin);
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulseScaled, imageBlockVals, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, this->blockXSize, this->blockYSize, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }

                    blockXOrigin += blockWidth;

                    blockMinX += blockXSize;
                    blockMaxX += blockXSize;
                }
                if(remainingCols > 0)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    blockMaxX -= (blockXSize-remainingCols);
                    //std::cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //std::cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin <<"]\n";

                    //std::cout << "Block Size [" << blockXSize << "," << blockYSize << "]\n";

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);

                    if(binScaling == 0)
                    {
                        this->populateFromImage(imageBlockVals, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands, imgBands, imgXOrigin, imgYOrigin, processingResolution, blockXOrigin, blockYOrigin);
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulses, imageBlockVals, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, remainingCols, this->blockYSize, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateFromImage(imageBlockVals, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands, imgBands, imgXOrigin, imgYOrigin, processingResolution, blockXOrigin, blockYOrigin);
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulseScaled, imageBlockVals, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, remainingCols, this->blockYSize, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }
                }

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockYOrigin -= blockHeight;
                }
                else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    blockYOrigin += blockHeight;
                }
                else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
                {
                    blockYOrigin += blockHeight;
                }
                
                blockMinY += blockYSize;
                blockMaxY += blockYSize;
            }

            if(remainingRows > 0)
            {
                blockMaxY -= (blockYSize-remainingRows);

                blockMinX = 0;
                blockMaxX = blockXSize;

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
                {
                    blockXOrigin = spdInFile->getScanlineIdxMin() - (overlap * spdInFile->getBinSize());
                }
                
                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }

                    //std::cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //std::cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin << "]\n";

                    //std::cout << "Block Size [" << blockXSize << "," << blockYSize << "]\n";

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                          if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);

                    if(binScaling == 0)
                    {
                        this->populateFromImage(imageBlockVals, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands, imgBands, imgXOrigin, imgYOrigin, processingResolution, blockXOrigin, blockYOrigin);
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulses, imageBlockVals, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, this->blockXSize, remainingRows, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateFromImage(imageBlockVals, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands, imgBands, imgXOrigin, imgYOrigin, processingResolution, blockXOrigin, blockYOrigin);
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulseScaled, imageBlockVals, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, this->blockXSize, remainingRows, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }

                    blockMinX += blockXSize;
                    blockMaxX += blockXSize;
                    blockXOrigin += blockWidth;
                }
                if(remainingCols > 0)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    blockMaxX -= (blockXSize-remainingCols);

                    //std::cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //std::cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin <<"]\n";

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);

                    if(binScaling == 0)
                    {
                        this->populateFromImage(imageBlockVals, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands, imgBands, imgXOrigin, imgYOrigin, processingResolution, blockXOrigin, blockYOrigin);
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulses, imageBlockVals, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, remainingCols, remainingRows, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateFromImage(imageBlockVals, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands, imgBands, imgXOrigin, imgYOrigin, processingResolution, blockXOrigin, blockYOrigin);
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulseScaled, imageBlockVals, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, remainingCols, remainingRows, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }
                }
            }
            std::cout << "Complete\n";
            
            dataBlockProcessor->setHeaderValues(spdOutFile);

            for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                {
                    delete pulses[i][j];
                }
                delete[] pulses[i];
            }
            delete[] pulses;

            if(binScaling != 0)
            {
                for(boost::uint_fast32_t i = 0; i < pulsesScaledBlockSizeY; ++i)
                {
                    for(boost::uint_fast32_t j = 0; j < pulsesScaledBlockSizeX; ++j)
                    {
                        delete pulseScaled[i][j];
                        delete cenPts[i][j];
                        delete[] imageBlockVals[i][j];
                    }
                    delete[] pulseScaled[i];
                    delete[] cenPts[i];
                    delete[] imageBlockVals[i];
                }
                delete[] pulseScaled;
                delete[] cenPts;
                delete[] imageBlockVals;
            }
            else
            {
                for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
                {
                    for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                    {
                        delete cenPts[i][j];
                        delete[] imageBlockVals[i][j];
                    }
                    delete[] cenPts[i];
                    delete[] imageBlockVals[i];
                }
                delete[] cenPts;
                delete[] imageBlockVals;
            }

            delete[] bbox;
            incReader.close();
            fileWriter->finaliseClose();
            delete fileWriter;
            delete[] imgBands;
            GDALClose(gdalDataset);
        }
		catch(boost::numeric::negative_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::positive_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::bad_numeric_cast& e)
		{
			throw SPDProcessingException(e.what());
		}
        catch(SPDIOException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }

    void SPDProcessDataBlocks::processDataBlocksGridPulsesOutputImage(SPDFile *spdInFile, std::string outImagePath, float processingResolution, boost::uint_fast16_t numImgBands, std::string gdalFormat) throw(SPDProcessingException)
    {
        try
        {
            GDALAllRegister();
            SPDGridData gridData;
            SPDFileReader reader;
            reader.readHeaderInfo(spdInFile->getFilePath(), spdInFile);
            
            if(spdInFile->getIndexType() == SPD_UPD_TYPE)
            {
                throw SPDProcessingException("The SPD file must have a spatial index. Use spdtranslate.");
            }
            
            if(this->blockXSize == 0)
            {
                this->blockXSize = spdInFile->getNumberBinsX();
            }
            
            if((this->overlap > this->blockXSize) | (this->overlap > this->blockYSize))
            {
                throw SPDProcessingException("The overlap must be smaller than the block size in both axis\'");
            }

            if(processingResolution <= 0)
            {
                processingResolution = spdInFile->getBinSize();
            }
            
            boost::uint_fast64_t nativeXBins = spdInFile->getNumberBinsX();
            boost::uint_fast64_t nativeYBins = spdInFile->getNumberBinsY();
            
            boost::uint_fast64_t procResXBins = 0;
            boost::uint_fast64_t procResYBins = 0;
            
            double geoWidth = 0;
            double geoHeight = 0;
            
            if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
            {
                geoWidth = spdInFile->getXMax() - spdInFile->getXMin();
                geoHeight = spdInFile->getYMax() - spdInFile->getYMin();
            }
            else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
            {
                geoWidth = spdInFile->getAzimuthMax() - spdInFile->getAzimuthMin();
                geoHeight = spdInFile->getZenithMax() - spdInFile->getZenithMin();
            }
            else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
            {
                geoWidth = spdInFile->getScanlineIdxMax() - spdInFile->getScanlineIdxMin();
                geoHeight = spdInFile->getScanlineMax() - spdInFile->getScanlineMin();
            }
                      
            //std::cout << "Geo: [" << geoWidth << "," << geoHeight << "]\n";
            
            bool usingNativeRes = false;
            bool scaleDown = true;
            boost::uint_fast32_t binScaling = 0;
            if(processingResolution == spdInFile->getBinSize())
            {
                usingNativeRes = true;
                procResXBins = spdInFile->getNumberBinsX();
                procResYBins = spdInFile->getNumberBinsY();
            }
            else if(processingResolution > spdInFile->getBinSize())
            {
                if(fmod(processingResolution, spdInFile->getBinSize()) != 0)
                {
                    std::cerr << "Native Res: " << spdInFile->getBinSize() << std::endl;
                    std::cerr << "Process Res: " << processingResolution << std::endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resoltion.");
                }
                usingNativeRes = false;
                procResXBins = ceil(((double)geoWidth / processingResolution))+1;
                procResYBins = ceil(((double)geoHeight / processingResolution))+1;
                
                scaleDown = false;
                binScaling = boost::numeric_cast<boost::uint_fast32_t>(processingResolution/spdInFile->getBinSize());
            }
            else
            {
                float tmpNumOutBins = spdInFile->getBinSize()/processingResolution;
                if(fmod(tmpNumOutBins, ((float)1.0)) != 0)
                {
                    std::cerr << "Native Res: " << spdInFile->getBinSize() << std::endl;
                    std::cerr << "Process Res: " << processingResolution << std::endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resoltion.");
                }
                usingNativeRes = false;
                procResXBins = ceil(((double)geoWidth / processingResolution))+1;
                procResYBins = ceil(((double)geoHeight / processingResolution))+1;
                
                scaleDown = true;
                binScaling = boost::numeric_cast<boost::uint_fast32_t>(spdInFile->getBinSize()/processingResolution);
            }
            
            //std::cout << "Bin Scaling: " << binScaling << std::endl;
            //std::cout << "Process Bins: [" << procResXBins << "," << procResYBins << "]\n";
                        
            if(usingNativeRes)
            {
                std::cout << "Using native resolution for processing\n";
            }
            else
            {
                std::cout << "Resampling native resolution\n";
                if(scaleDown)
                {
                    std::cout << "Scaling down\n";
                }
                else
                {
                    std::cout << "Scaling up\n";
                }
            }
            
            boost::uint_fast32_t numXFullBlocks = floor(((double)nativeXBins)/this->blockXSize);
            boost::uint_fast32_t numYFullBlocks = floor(((double)nativeYBins)/this->blockYSize);
            
            //std::cout << "Number of full blocks: [" << numXFullBlocks << "," << numYFullBlocks << "]\n";
            
            boost::uint_fast32_t remainingCols = nativeXBins - (numXFullBlocks * this->blockXSize);
            boost::uint_fast32_t remainingRows = nativeYBins - (numYFullBlocks * this->blockYSize);
            
            //std::cout << "Remainder: [" << remainingCols << "," << remainingRows << "]\n";
            
            double blockMinX = 0;
            double blockMaxX = 0;
            double blockMinY = 0;
            double blockMaxY = 0;
            
            double blockWidth = blockXSize * spdInFile->getBinSize();
            double blockHeight = blockYSize * spdInFile->getBinSize();
            
            boost::uint_fast32_t procResXBlockSize = 0;
            boost::uint_fast32_t procResYBlockSize = 0;
            
            if(binScaling == 0)
            {
                procResXBlockSize = blockXSize;
                procResYBlockSize = blockYSize;
            }
            else if(scaleDown)
            {
                procResXBlockSize = ceil(((double)blockXSize) * binScaling);
                procResYBlockSize = ceil(((double)blockYSize) * binScaling);
            }
            else
            {
                procResXBlockSize = ceil(((double)blockXSize) / binScaling);
                procResYBlockSize = ceil(((double)blockYSize) / binScaling);
            }
            
            //std::cout << "Native block size: [" << this->blockXSize << "," << this->blockYSize << "]\n";
            //std::cout << "Process block size: [" << procResXBlockSize << "," << procResYBlockSize << "]\n";
            //std::cout << "Block Size: [" << blockWidth << "," << blockHeight << "]\n";
            
            boost::uint_fast32_t numBlocks = numYFullBlocks * numXFullBlocks;
            if(remainingCols > 0)
            {
                numBlocks += numYFullBlocks;
            }
            if(remainingRows > 0)
            {
                numBlocks += numXFullBlocks;
                if(remainingCols > 0)
                {
                    numBlocks += 1;
                }
            }
            boost::uint_fast32_t cBlocksIdx = 1;
            
            blockMinY = 0;
            blockMaxY = blockYSize;
            
            boost::uint_fast32_t pulsesBlockSizeX = this->blockXSize + (2 * this->overlap);
            boost::uint_fast32_t pulsesBlockSizeY = this->blockYSize + (2 * this->overlap);
            boost::uint_fast32_t remainingColsScaled = remainingCols;
            boost::uint_fast32_t remainingRowsScaled = remainingRows;
            
            boost::uint_fast32_t scaledOverlap = 0;
            if(binScaling != 0)
            {
                if(scaleDown)
                {
                    scaledOverlap = this->overlap * binScaling;
                    remainingColsScaled = ceil(((double)remainingCols * binScaling));
                    remainingRowsScaled = ceil(((double)remainingRows * binScaling));
                }
                else
                {
                    scaledOverlap = this->overlap / binScaling;
                    remainingColsScaled = ceil(((double)remainingCols / binScaling));
                    remainingRowsScaled = ceil(((double)remainingRows / binScaling));
                }
            }
            
            boost::uint_fast32_t pulsesScaledBlockSizeX = procResXBlockSize + (2 * scaledOverlap);
            boost::uint_fast32_t pulsesScaledBlockSizeY = procResYBlockSize + (2 * scaledOverlap);
            
            //std::cout << "Pulses Block Size: [" << pulsesBlockSizeX << "," << pulsesBlockSizeY << "]\n";
            //std::cout << "Processing Pulses Block Size: [" << procResXBlockSize << "," << procResYBlockSize << "]\n";
            //std::cout << "Remaining: [" << remainingColsScaled << "," << remainingRowsScaled << "]\n";
            
            boost::uint_fast32_t imageXSize = (procResXBlockSize * numXFullBlocks) + remainingColsScaled;
            boost::uint_fast32_t imageYSize = (procResYBlockSize * numYFullBlocks) + remainingRowsScaled;
            
            /****** CREATE A NEW GDALDATASET *******/
            GDALDriver *gdalDriver = GetGDALDriverManager()->GetDriverByName(gdalFormat.c_str());
			if(gdalDriver == NULL)
			{
                std::string message = gdalFormat + std::string(" GDAL driver cannot be found.");
				throw SPDProcessingException(message);
			}
            char **papszMetadata;
            papszMetadata = gdalDriver->GetMetadata();
            if( !CSLFetchBoolean( papszMetadata, GDAL_DCAP_CREATE, FALSE ) )
            {
                std::string message = gdalFormat + std::string(" does not support create method. Select a GDAL driver which does (see http://www.gdal.org/formats_list.html).");
                throw SPDProcessingException(message);
            }

            GDALDataset *outImage = gdalDriver->Create(outImagePath.c_str(), imageXSize, imageYSize, numImgBands, GDT_Float32, NULL);
            
            if(outImage == NULL)
            {
                std::string message = std::string("Failed to create image ") + outImagePath;
				throw SPDProcessingException(message);
            }
            
            //std::cout << spdInFile << std::endl;
            
            double *gdalTransform = new double[6];
            gdalTransform[0] = spdInFile->getXMin();
            gdalTransform[1] = processingResolution;
            gdalTransform[2] = 0;
            gdalTransform[3] = spdInFile->getYMax();
            gdalTransform[4] = 0;
            gdalTransform[5] = processingResolution*(-1);
            outImage->SetGeoTransform(gdalTransform);
            
            if(spdInFile->getSpatialReference() != "")
            {
                outImage->SetProjection(spdInFile->getSpatialReference().c_str());
            }
            std::vector<std::string> bandNames = this->dataBlockProcessor->getImageBandDescriptions();
            
            GDALRasterBand **imageBands = new GDALRasterBand*[numImgBands];
            for(boost::uint_fast16_t n = 0; n < numImgBands; ++n)
            {
                imageBands[n] = outImage->GetRasterBand(n+1);
            }
            
            if(bandNames.size() == numImgBands)
            {
                for(boost::uint_fast16_t n = 0; n < numImgBands; ++n)
                {
                    imageBands[n]->SetDescription(bandNames[n].c_str());
                }
            }
            /****** CREATED A NEW GDALDATASET *******/
            
            SPDFileIncrementalReader incReader;
            incReader.open(spdInFile);
                        
            std::vector<SPDPulse*> ***pulses = new std::vector<SPDPulse*>**[pulsesBlockSizeY];
            for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
            {
                pulses[i] = new std::vector<SPDPulse*>*[pulsesBlockSizeX];
                for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                {
                    pulses[i][j] = new std::vector<SPDPulse*>();
                }
            }
            
            SPDXYPoint ***cenPts = NULL;
            std::vector<SPDPulse*> ***pulseScaled = NULL;
            float ***imageBlockVals = NULL;
            if(binScaling != 0)
            {
                pulseScaled = new std::vector<SPDPulse*>**[pulsesScaledBlockSizeY];
                cenPts = new SPDXYPoint**[pulsesScaledBlockSizeY];
                imageBlockVals = new float**[pulsesScaledBlockSizeY];
                for(boost::uint_fast32_t i = 0; i < pulsesScaledBlockSizeY; ++i)
                {
                    pulseScaled[i] = new std::vector<SPDPulse*>*[pulsesScaledBlockSizeX];
                    cenPts[i] = new SPDXYPoint*[pulsesScaledBlockSizeX];
                    imageBlockVals[i] = new float*[pulsesScaledBlockSizeX];
                    for(boost::uint_fast32_t j = 0; j < pulsesScaledBlockSizeX; ++j)
                    {
                        pulseScaled[i][j] = new std::vector<SPDPulse*>();
                        cenPts[i][j] = new SPDXYPoint();
                        imageBlockVals[i][j] = new float[numImgBands];
                        for(boost::uint_fast32_t k = 0; k < numImgBands; ++k)
                        {
                            imageBlockVals[i][j][k] = 0.0;
                        }
                    }
                }
            }
            else
            {
                cenPts = new SPDXYPoint**[pulsesBlockSizeY];
                imageBlockVals = new float**[pulsesBlockSizeY];
                for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
                {
                    cenPts[i] = new SPDXYPoint*[pulsesBlockSizeX];
                    imageBlockVals[i] = new float*[pulsesBlockSizeX];
                    for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                    {
                        cenPts[i][j] = new SPDXYPoint();
                        imageBlockVals[i][j] = new float[numImgBands];
                        for(boost::uint_fast32_t k = 0; k < numImgBands; ++k)
                        {
                            imageBlockVals[i][j][k] = 0.0;
                        }
                    }
                }
            }
            
            boost::uint_fast32_t *bbox = new boost::uint_fast32_t[4];
            
            boost::uint_fast32_t xOffset = 0;
            boost::uint_fast32_t yOffset = 0;
            
            double blockXOrigin = 0;
            double blockYOrigin = 0;
            
            boost::uint_fast32_t blockMinXScaled = 0;
            boost::uint_fast32_t blockMinYScaled = 0;
                        
            if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
            {
                blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getYMax() + (overlap * spdInFile->getBinSize());
            }
            else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
            {
                blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getZenithMin() - (overlap * spdInFile->getBinSize());
            }
            else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
            {
                blockXOrigin = spdInFile->getScanlineIdxMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getScanlineMin() - (overlap * spdInFile->getBinSize());
            }
            
            for(boost::uint_fast32_t i = 0; i < numYFullBlocks; ++i)
            {
                blockMinX = 0;
                blockMaxX = blockXSize;
                
                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
                {
                    blockXOrigin = spdInFile->getScanlineIdxMin() - (overlap * spdInFile->getBinSize());
                }
              
                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    
                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;
                    
                    xOffset = 0;
                    yOffset = 0;
                    
                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }
                    
                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }
                    
                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }
                    
                    //std::cout << "xOffset = " << xOffset << std::endl;
                    //std::cout << "yOffset = " << yOffset << std::endl;
                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);
                    if(binScaling == 0)
                    {
                        this->resetImageBlock2Zeros(imageBlockVals, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands);
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulses, imageBlockVals, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands, processingResolution);
                        this->writeImageData(imageBands, imageBlockVals, this->blockXSize, this->blockYSize, numImgBands, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->resetImageBlock2Zeros(imageBlockVals, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands);
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulseScaled, imageBlockVals, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands, processingResolution);
                        if(scaleDown)
                        {
                            blockMinXScaled = blockMinX * binScaling;
                            blockMinYScaled = blockMinY * binScaling;
                        }
                        else
                        {
                            blockMinXScaled = blockMinX / binScaling;
                            blockMinYScaled = blockMinY / binScaling;
                        }
                        this->writeImageData(imageBands, imageBlockVals, procResXBlockSize, procResYBlockSize, numImgBands, scaledOverlap, scaledOverlap, blockMinXScaled, blockMinYScaled);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }
                    
                    blockXOrigin += blockWidth;
                    
                    blockMinX += blockXSize;
                    blockMaxX += blockXSize;
                }
                if(remainingCols > 0)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    blockMaxX -= (blockXSize-remainingCols);
                    
                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;
                    
                    xOffset = 0;
                    yOffset = 0;
                    
                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }
                    
                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }
                    
                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }
                    
                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);
                    
                    if(binScaling == 0)
                    {
                        this->resetImageBlock2Zeros(imageBlockVals, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands);
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulses, imageBlockVals, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands, processingResolution);
                        this->writeImageData(imageBands, imageBlockVals, remainingCols, this->blockYSize, numImgBands, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->resetImageBlock2Zeros(imageBlockVals, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands);
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulseScaled, imageBlockVals, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands, processingResolution);
                        if(scaleDown)
                        {
                            blockMinXScaled = blockMinX * binScaling;
                            blockMinYScaled = blockMinY * binScaling;
                        }
                        else
                        {
                            blockMinXScaled = blockMinX / binScaling;
                            blockMinYScaled = blockMinY / binScaling;
                        }
                        this->writeImageData(imageBands, imageBlockVals, remainingColsScaled, procResYBlockSize, numImgBands, scaledOverlap, scaledOverlap, blockMinXScaled, blockMinYScaled);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }
                }
                
                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockYOrigin -= blockHeight;
                }
                else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    blockYOrigin += blockHeight;
                }
                else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
                {
                    blockYOrigin += blockHeight;
                }    
                            
                blockMinY += blockYSize;
                blockMaxY += blockYSize;
            }
            
            if(remainingRows > 0)
            {
                blockMaxY -= (blockYSize-remainingRows);
                
                blockMinX = 0;
                blockMaxX = blockXSize;
                
                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
                {
                    blockXOrigin = spdInFile->getScanlineIdxMin() - (overlap * spdInFile->getBinSize());
                }
               
                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    
                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;
                    
                    xOffset = 0;
                    yOffset = 0;
                    
                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }
                    
                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }
                    
                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }
                    
                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);
                    
                    if(binScaling == 0)
                    {
                        this->resetImageBlock2Zeros(imageBlockVals, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands);
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulses, imageBlockVals, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands, processingResolution);
                        this->writeImageData(imageBands, imageBlockVals, this->blockXSize, remainingRows, numImgBands, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->resetImageBlock2Zeros(imageBlockVals, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands);
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulseScaled, imageBlockVals, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands, processingResolution);
                        if(scaleDown)
                        {
                            blockMinXScaled = blockMinX * binScaling;
                            blockMinYScaled = blockMinY * binScaling;
                        }
                        else
                        {
                            blockMinXScaled = blockMinX / binScaling;
                            blockMinYScaled = blockMinY / binScaling;
                        }
                        this->writeImageData(imageBands, imageBlockVals, procResXBlockSize, remainingRowsScaled, numImgBands, scaledOverlap, scaledOverlap, blockMinXScaled, blockMinYScaled);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }
                    
                    blockMinX += blockXSize;
                    blockMaxX += blockXSize;
                    blockXOrigin += blockWidth;
                }
                if(remainingCols > 0)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    
                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;
                    
                    xOffset = 0;
                    yOffset = 0;
                    
                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }
                    
                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }
                    
                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }
                    
                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);
                    
                    if(binScaling == 0)
                    {
                        this->resetImageBlock2Zeros(imageBlockVals, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands);
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulses, imageBlockVals, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, numImgBands, processingResolution);
                        this->writeImageData(imageBands, imageBlockVals, remainingCols, remainingRows, numImgBands, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->resetImageBlock2Zeros(imageBlockVals, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands);
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlockImage(spdInFile, pulseScaled, imageBlockVals, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, numImgBands, processingResolution);
                        if(scaleDown)
                        {
                            blockMinXScaled = blockMinX * binScaling;
                            blockMinYScaled = blockMinY * binScaling;
                        }
                        else
                        {
                            blockMinXScaled = blockMinX / binScaling;
                            blockMinYScaled = blockMinY / binScaling;
                        }
                        this->writeImageData(imageBands, imageBlockVals, remainingColsScaled, remainingRowsScaled, numImgBands, scaledOverlap, scaledOverlap, blockMinXScaled, blockMinYScaled);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }
                }
            }
            std::cout << "Complete\n";
            
            for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                {
                    delete pulses[i][j];
                }
                delete[] pulses[i];
            }
            delete[] pulses;
            
            if(binScaling != 0)
            {
                for(boost::uint_fast32_t i = 0; i < pulsesScaledBlockSizeY; ++i)
                {
                    for(boost::uint_fast32_t j = 0; j < pulsesScaledBlockSizeX; ++j)
                    {
                        delete pulseScaled[i][j];
                        delete cenPts[i][j];
                        delete[] imageBlockVals[i][j];
                    }
                    delete[] pulseScaled[i];
                    delete [] cenPts[i];
                    delete[] imageBlockVals[i];
                }
                delete[] pulseScaled;
                delete[] cenPts;
                delete[] imageBlockVals;
            }
            else
            {
                for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
                {
                    for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                    {
                        delete cenPts[i][j];
                        delete[] imageBlockVals[i][j];
                    }
                    delete[] cenPts[i];
                    delete[] imageBlockVals[i];
                }
                delete[] cenPts;
                delete[] imageBlockVals;
            }
            
            delete[] bbox;
            delete[] imageBands;
            incReader.close();
            GDALClose(outImage);
            delete[] gdalTransform;
        }
		catch(boost::numeric::negative_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::positive_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::bad_numeric_cast& e)
		{
			throw SPDProcessingException(e.what());
		}
        catch(SPDIOException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }

    void SPDProcessDataBlocks::processDataBlocksGridPulsesOutputSPD(SPDFile *spdInFile, std::string outFile, float processingResolution) throw(SPDProcessingException)
    {
        try
        {
            SPDGridData gridData;
            SPDFileReader reader;
            reader.readHeaderInfo(spdInFile->getFilePath(), spdInFile);
            
            if(spdInFile->getIndexType() == SPD_UPD_TYPE)
            {
                throw SPDProcessingException("The SPD file must have a spatial index. Use spdtranslate.");
            }
            
            if(this->blockXSize == 0)
            {
                this->blockXSize = spdInFile->getNumberBinsX();
            }
            
            if((this->overlap > this->blockXSize) | (this->overlap > this->blockYSize))
            {
                throw SPDProcessingException("The overlap must be smaller than the block size in both axis\'");
            }
            
            if(processingResolution <= 0)
            {
                processingResolution = spdInFile->getBinSize();
            }

            SPDFile *spdOutFile = new SPDFile(outFile);
            spdOutFile->copyAttributesFrom(spdInFile);

            boost::uint_fast64_t nativeXBins = spdInFile->getNumberBinsX();
            boost::uint_fast64_t nativeYBins = spdInFile->getNumberBinsY();

            boost::uint_fast64_t procResXBins = 0;
            boost::uint_fast64_t procResYBins = 0;

            double geoWidth = 0;
            double geoHeight = 0;

            if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
            {
                geoWidth = spdInFile->getXMax() - spdInFile->getXMin();
                geoHeight = spdInFile->getYMax() - spdInFile->getYMin();
            }
            else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
            {
                geoWidth = spdInFile->getAzimuthMax() - spdInFile->getAzimuthMin();
                geoHeight = spdInFile->getZenithMax() - spdInFile->getZenithMin();
            }
            else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
            {
                geoWidth = spdInFile->getScanlineIdxMax() - spdInFile->getScanlineIdxMin();
                geoHeight = spdInFile->getScanlineMax() - spdInFile->getScanlineMin();
            }

            //std::cout << "Geo: [" << geoWidth << "," << geoHeight << "]\n";

            bool usingNativeRes = false;
            bool scaleDown = true;
            boost::uint_fast32_t binScaling = 0;
            if(processingResolution == spdInFile->getBinSize())
            {
                usingNativeRes = true;
                procResXBins = spdInFile->getNumberBinsX();
                procResYBins = spdInFile->getNumberBinsY();
            }
            else if(processingResolution > spdInFile->getBinSize())
            {
                if(fmod(processingResolution, spdInFile->getBinSize()) != 0)
                {
                    std::cerr << "Native Res: " << spdInFile->getBinSize() << std::endl;
                    std::cerr << "Process Res: " << processingResolution << std::endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resoltion.");
                }
                usingNativeRes = false;
                procResXBins = boost::numeric_cast<boost::uint_fast64_t>(geoWidth / processingResolution)+1;
                procResYBins = boost::numeric_cast<boost::uint_fast64_t>(geoHeight / processingResolution)+1;

                scaleDown = false;
                binScaling = boost::numeric_cast<boost::uint_fast32_t>(processingResolution/spdInFile->getBinSize());
            }
            else
            {
                float tmpNumOutBins = spdInFile->getBinSize()/processingResolution;
                if(fmod(tmpNumOutBins, ((float)1.0)) != 0)
                {
                    std::cerr << "Native Res: " << spdInFile->getBinSize() << std::endl;
                    std::cerr << "Process Res: " << processingResolution << std::endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resoltion.");
                }
                usingNativeRes = false;
                procResXBins = boost::numeric_cast<boost::uint_fast64_t>(geoWidth / processingResolution)+1;
                procResYBins = boost::numeric_cast<boost::uint_fast64_t>(geoHeight / processingResolution)+1;

                scaleDown = true;
                binScaling = boost::numeric_cast<boost::uint_fast32_t>(spdInFile->getBinSize()/processingResolution);
            }

            //std::cout << "Bin Scaling: " << binScaling << std::endl;
            //std::cout << "Process Bins: [" << procResXBins << "," << procResYBins << "]\n";

            if(usingNativeRes)
            {
                std::cout << "Using native resolution for processing\n";
            }
            else
            {
                std::cout << "Resampling native resolution\n";
                if(scaleDown)
                {
                    std::cout << "Scaling down\n";
                }
                else
                {
                    std::cout << "Scaling up\n";
                }
            }

            boost::uint_fast32_t numXFullBlocks = floor(((double)nativeXBins)/this->blockXSize);
            boost::uint_fast32_t numYFullBlocks = floor(((double)nativeYBins)/this->blockYSize);

            //std::cout << "Number of full blocks: [" << numXFullBlocks << "," << numYFullBlocks << "]\n";

            boost::uint_fast32_t remainingCols = nativeXBins - (numXFullBlocks * this->blockXSize);
            boost::uint_fast32_t remainingRows = nativeYBins - (numYFullBlocks * this->blockYSize);

            //std::cout << "Remainder: [" << remainingCols << "," << remainingRows << "]\n";

            double blockMinX = 0;
            double blockMaxX = 0;
            double blockMinY = 0;
            double blockMaxY = 0;

            double blockWidth = blockXSize * spdInFile->getBinSize();
            double blockHeight = blockYSize * spdInFile->getBinSize();

            boost::uint_fast32_t procResXBlockSize = 0;
            boost::uint_fast32_t procResYBlockSize = 0;

            if(binScaling == 0)
            {
                procResXBlockSize = blockXSize;
                procResYBlockSize = blockYSize;
            }
            else if(scaleDown)
            {
                procResXBlockSize = ceil(((double)blockXSize) * binScaling);
                procResYBlockSize = ceil(((double)blockYSize) * binScaling);
            }
            else
            {
                procResXBlockSize = ceil(((double)blockXSize) / binScaling);
                procResYBlockSize = ceil(((double)blockYSize) / binScaling);
            }

            //std::cout << "Native block size: [" << this->blockXSize << "," << this->blockXSize << "]\n";
            //std::cout << "Process block size: [" << procResXBlockSize << "," << procResYBlockSize << "]\n";

            //std::cout << "Block Size: [" << blockWidth << "," << blockHeight << "]\n";



            boost::uint_fast32_t numBlocks = numYFullBlocks * numXFullBlocks;
            if(remainingCols > 0)
            {
                numBlocks += numYFullBlocks;
            }
            if(remainingRows > 0)
            {
                numBlocks += numXFullBlocks;
                if(remainingCols > 0)
                {
                    numBlocks += 1;
                }
            }
            boost::uint_fast32_t cBlocksIdx = 1;

            blockMinY = 0;
            blockMaxY = blockYSize;

            SPDFileIncrementalReader incReader;
            incReader.open(spdInFile);

            SPDDataExporter *fileWriter = NULL;

            if(this->blockXSize == spdInFile->getNumberBinsX())
            {
                fileWriter = new SPDSeqFileWriter();
            }
            else
            {
                fileWriter = new SPDNonSeqFileWriter();
            }
            fileWriter->open(spdOutFile, outFile);
            fileWriter->setKeepMinExtent(keepMinExtent);

            boost::uint_fast32_t pulsesBlockSizeX = this->blockXSize + (2 * this->overlap);
            boost::uint_fast32_t pulsesBlockSizeY = this->blockYSize + (2 * this->overlap);

            boost::uint_fast32_t scaledOverlap = 0;
            if(binScaling != 0)
            {
                if(scaleDown)
                {
                    scaledOverlap = ceil(((double)this->overlap) * binScaling);
                }
                else
                {
                    scaledOverlap = ceil(((double)this->overlap) / binScaling);
                }
            }

            boost::uint_fast32_t pulsesScaledBlockSizeX = procResXBlockSize + (2 * scaledOverlap);
            boost::uint_fast32_t pulsesScaledBlockSizeY = procResYBlockSize + (2 * scaledOverlap);

            std::vector<SPDPulse*> ***pulses = new std::vector<SPDPulse*>**[pulsesBlockSizeY];
            for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
            {
                pulses[i] = new std::vector<SPDPulse*>*[pulsesBlockSizeX];
                for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                {
                    pulses[i][j] = new std::vector<SPDPulse*>();
                }
            }


            SPDXYPoint ***cenPts = NULL;
            std::vector<SPDPulse*> ***pulseScaled = NULL;
            if(binScaling != 0)
            {
                pulseScaled = new std::vector<SPDPulse*>**[pulsesScaledBlockSizeY];
                cenPts = new SPDXYPoint**[pulsesScaledBlockSizeY];
                for(boost::uint_fast32_t i = 0; i < pulsesScaledBlockSizeY; ++i)
                {
                    pulseScaled[i] = new std::vector<SPDPulse*>*[pulsesScaledBlockSizeX];
                    cenPts[i] = new SPDXYPoint*[pulsesScaledBlockSizeX];
                    for(boost::uint_fast32_t j = 0; j < pulsesScaledBlockSizeX; ++j)
                    {
                        pulseScaled[i][j] = new std::vector<SPDPulse*>();
                        cenPts[i][j] = new SPDXYPoint();
                    }
                }
            }
            else
            {
                cenPts = new SPDXYPoint**[pulsesBlockSizeY];
                for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
                {
                    cenPts[i] = new SPDXYPoint*[pulsesBlockSizeX];
                    for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                    {
                        cenPts[i][j] = new SPDXYPoint();
                    }
                }
            }

            boost::uint_fast32_t *bbox = new boost::uint_fast32_t[4];

            boost::uint_fast32_t xOffset = 0;
            boost::uint_fast32_t yOffset = 0;

            double blockXOrigin = 0;
            double blockYOrigin = 0;

            if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
            {
                blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getYMax() + (overlap * spdInFile->getBinSize());
            }
            else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
            {
                blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getZenithMin() - (overlap * spdInFile->getBinSize());
            }
            else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
            {
                blockXOrigin = spdInFile->getScanlineIdxMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getScanlineMin() - (overlap * spdInFile->getBinSize());
            }

            for(boost::uint_fast32_t i = 0; i < numYFullBlocks; ++i)
            {
                blockMinX = 0;
                blockMaxX = blockXSize;

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
                {
                    blockXOrigin = spdInFile->getScanlineIdxMin() - (overlap * spdInFile->getBinSize());
                }
                
                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    
                    //std::cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //std::cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin <<"]\n";

                    //std::cout << "Block Size [" << blockXSize << "," << blockYSize << "]\n";

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);

                    if(binScaling == 0)
                    {
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, this->blockXSize, this->blockYSize, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, this->blockXSize, this->blockYSize, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }

                    blockXOrigin += blockWidth;

                    blockMinX += blockXSize;
                    blockMaxX += blockXSize;
                    
                }
                if(remainingCols > 0)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    blockMaxX -= (blockXSize-remainingCols);
                    //std::cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //std::cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin <<"]\n";

                    //std::cout << "Block Size [" << blockXSize << "," << blockYSize << "]\n";

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);

                    if(binScaling == 0)
                    {
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, remainingCols, this->blockYSize, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, remainingCols, this->blockYSize, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }
                }

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockYOrigin -= blockHeight;
                }
                else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    blockYOrigin += blockHeight;
                }
                else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
                {
                    blockYOrigin += blockHeight;
                }
                
                blockMinY += blockYSize;
                blockMaxY += blockYSize;
            }

            if(remainingRows > 0)
            {
                blockMaxY -= (blockYSize-remainingRows);

                blockMinX = 0;
                blockMaxX = blockXSize;

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
                {
                    blockXOrigin = spdInFile->getScanlineIdxMin() - (overlap * spdInFile->getBinSize());
                }

                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }

                    //std::cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //std::cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin << "]\n";

                    //std::cout << "Block Size [" << blockXSize << "," << blockYSize << "]\n";

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);

                    if(binScaling == 0)
                    {
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, this->blockXSize, remainingRows, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, this->blockXSize, remainingRows, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }

                    blockMinX += blockXSize;
                    blockMaxX += blockXSize;
                    blockXOrigin += blockWidth;
                }
                if(remainingCols > 0)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    blockMaxX -= (blockXSize-remainingCols);

                    //std::cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //std::cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin <<"]\n";

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);

                    if(binScaling == 0)
                    {
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, remainingCols, remainingRows, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, processingResolution);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, remainingCols, remainingRows, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }
                }
            }
            std::cout << "Complete\n";
            
            dataBlockProcessor->setHeaderValues(spdOutFile);

            for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                {
                    delete pulses[i][j];
                }
                delete[] pulses[i];
            }
            delete[] pulses;

            if(binScaling != 0)
            {
                for(boost::uint_fast32_t i = 0; i < pulsesScaledBlockSizeY; ++i)
                {
                    for(boost::uint_fast32_t j = 0; j < pulsesScaledBlockSizeX; ++j)
                    {
                        delete pulseScaled[i][j];
                        delete cenPts[i][j];
                    }
                    delete[] pulseScaled[i];
                    delete [] cenPts[i];
                }
                delete[] pulseScaled;
                delete[] cenPts;
            }
            else
            {
                for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
                {
                    for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                    {
                        delete cenPts[i][j];
                    }
                    delete [] cenPts[i];
                }
                delete[] cenPts;
            }

            delete[] bbox;
            incReader.close();
            fileWriter->finaliseClose();
            delete fileWriter;
        }
		catch(boost::numeric::negative_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::positive_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::bad_numeric_cast& e)
		{
			throw SPDProcessingException(e.what());
		}
        catch(SPDIOException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }

    void SPDProcessDataBlocks::processDataBlocksGridPulses(SPDFile *spdInFile, float processingResolution) throw(SPDProcessingException)
    {
        try
        {
            SPDGridData gridData;
            SPDFileReader reader;
            reader.readHeaderInfo(spdInFile->getFilePath(), spdInFile);
            
            if(spdInFile->getIndexType() == SPD_UPD_TYPE)
            {
                throw SPDProcessingException("The SPD file must have a spatial index. Use spdtranslate.");
            }
            
            if(this->blockXSize == 0)
            {
                this->blockXSize = spdInFile->getNumberBinsX();
            }
            
            if((this->overlap > this->blockXSize) | (this->overlap > this->blockYSize))
            {
                throw SPDProcessingException("The overlap must be smaller than the block size in both axis\'");
            }
            
            if(processingResolution <= 0)
            {
                processingResolution = spdInFile->getBinSize();
            }

            boost::uint_fast64_t nativeXBins = spdInFile->getNumberBinsX();
            boost::uint_fast64_t nativeYBins = spdInFile->getNumberBinsY();

            boost::uint_fast64_t procResXBins = 0;
            boost::uint_fast64_t procResYBins = 0;

            double geoWidth = 0;
            double geoHeight = 0;

            if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
            {
                geoWidth = spdInFile->getXMax() - spdInFile->getXMin();
                geoHeight = spdInFile->getYMax() - spdInFile->getYMin();
            }
            else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
            {
                geoWidth = spdInFile->getAzimuthMax() - spdInFile->getAzimuthMin();
                geoHeight = spdInFile->getZenithMax() - spdInFile->getZenithMin();
            }
            else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
            {
                geoWidth = spdInFile->getScanlineIdxMax() - spdInFile->getScanlineIdxMin();
                geoHeight = spdInFile->getScanlineMax() - spdInFile->getScanlineMin();
            }
            
            bool usingNativeRes = false;
            bool scaleDown = true;
            boost::uint_fast32_t binScaling = 0;
            if(processingResolution == spdInFile->getBinSize())
            {
                usingNativeRes = true;
                procResXBins = spdInFile->getNumberBinsX();
                procResYBins = spdInFile->getNumberBinsY();
            }
            else if(processingResolution > spdInFile->getBinSize())
            {
                if(fmod(processingResolution, spdInFile->getBinSize()) != 0)
                {
                    std::cerr << "Native Res: " << spdInFile->getBinSize() << std::endl;
                    std::cerr << "Process Res: " << processingResolution << std::endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resoltion.");
                }
                usingNativeRes = false;
                procResXBins = boost::numeric_cast<boost::uint_fast64_t>(geoWidth / processingResolution)+1;
                procResYBins = boost::numeric_cast<boost::uint_fast64_t>(geoHeight / processingResolution)+1;

                scaleDown = false;
                binScaling = boost::numeric_cast<boost::uint_fast32_t>(processingResolution/spdInFile->getBinSize());
            }
            else
            {
                float tmpNumOutBins = spdInFile->getBinSize()/processingResolution;
                if(fmod(tmpNumOutBins, ((float)1.0)) != 0)
                {
                    std::cerr << "Native Res: " << spdInFile->getBinSize() << std::endl;
                    std::cerr << "Process Res: " << processingResolution << std::endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resoltion.");
                }
                usingNativeRes = false;
                procResXBins = boost::numeric_cast<boost::uint_fast64_t>(geoWidth / processingResolution)+1;
                procResYBins = boost::numeric_cast<boost::uint_fast64_t>(geoHeight / processingResolution)+1;

                scaleDown = true;
                binScaling = boost::numeric_cast<boost::uint_fast32_t>(spdInFile->getBinSize()/processingResolution);
            }

            if(usingNativeRes)
            {
                std::cout << "Using native resolution for processing\n";
            }
            else
            {
                std::cout << "Resampling native resolution\n";
                if(scaleDown)
                {
                    std::cout << "Scaling down\n";
                }
                else
                {
                    std::cout << "Scaling up\n";
                }
            }

            boost::uint_fast32_t numXFullBlocks = floor(((double)nativeXBins)/this->blockXSize);
            boost::uint_fast32_t numYFullBlocks = floor(((double)nativeYBins)/this->blockYSize);

            boost::uint_fast32_t remainingCols = nativeXBins - (numXFullBlocks * this->blockXSize);
            boost::uint_fast32_t remainingRows = nativeYBins - (numYFullBlocks * this->blockYSize);

            double blockMinX = 0;
            double blockMaxX = 0;
            double blockMinY = 0;
            double blockMaxY = 0;

            double blockWidth = blockXSize * spdInFile->getBinSize();
            double blockHeight = blockYSize * spdInFile->getBinSize();

            boost::uint_fast32_t procResXBlockSize = 0;
            boost::uint_fast32_t procResYBlockSize = 0;

            if(binScaling == 0)
            {
                procResXBlockSize = blockXSize;
                procResYBlockSize = blockYSize;
            }
            else if(scaleDown)
            {
                procResXBlockSize = ceil(((double)blockXSize) * binScaling);
                procResYBlockSize = ceil(((double)blockYSize) * binScaling);
            }
            else
            {
                procResXBlockSize = ceil(((double)blockXSize) / binScaling);
                procResYBlockSize = ceil(((double)blockYSize) / binScaling);
            }

            boost::uint_fast32_t numBlocks = numYFullBlocks * numXFullBlocks;
            if(remainingCols > 0)
            {
                numBlocks += numYFullBlocks;
            }
            if(remainingRows > 0)
            {
                numBlocks += numXFullBlocks;
                if(remainingCols > 0)
                {
                    numBlocks += 1;
                }
            }
            boost::uint_fast32_t cBlocksIdx = 1;

            blockMinY = 0;
            blockMaxY = blockYSize;

            SPDFileIncrementalReader incReader;
            incReader.open(spdInFile);

            boost::uint_fast32_t pulsesBlockSizeX = this->blockXSize + (2 * this->overlap);
            boost::uint_fast32_t pulsesBlockSizeY = this->blockYSize + (2 * this->overlap);

            boost::uint_fast32_t scaledOverlap = 0;
            if(binScaling != 0)
            {
                if(scaleDown)
                {
                    scaledOverlap = ceil(((double)this->overlap) * binScaling);
                }
                else
                {
                    scaledOverlap = ceil(((double)this->overlap) / binScaling);
                }
            }

            boost::uint_fast32_t pulsesScaledBlockSizeX = procResXBlockSize + (2 * scaledOverlap);
            boost::uint_fast32_t pulsesScaledBlockSizeY = procResYBlockSize + (2 * scaledOverlap);

            std::vector<SPDPulse*> ***pulses = new std::vector<SPDPulse*>**[pulsesBlockSizeY];
            for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
            {
                pulses[i] = new std::vector<SPDPulse*>*[pulsesBlockSizeX];
                for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                {
                    pulses[i][j] = new std::vector<SPDPulse*>();
                }
            }


            SPDXYPoint ***cenPts = NULL;
            std::vector<SPDPulse*> ***pulseScaled = NULL;
            if(binScaling != 0)
            {
                pulseScaled = new std::vector<SPDPulse*>**[pulsesScaledBlockSizeY];
                cenPts = new SPDXYPoint**[pulsesScaledBlockSizeY];
                for(boost::uint_fast32_t i = 0; i < pulsesScaledBlockSizeY; ++i)
                {
                    pulseScaled[i] = new std::vector<SPDPulse*>*[pulsesScaledBlockSizeX];
                    cenPts[i] = new SPDXYPoint*[pulsesScaledBlockSizeX];
                    for(boost::uint_fast32_t j = 0; j < pulsesScaledBlockSizeX; ++j)
                    {
                        pulseScaled[i][j] = new std::vector<SPDPulse*>();
                        cenPts[i][j] = new SPDXYPoint();
                    }
                }
            }
            else
            {
                cenPts = new SPDXYPoint**[pulsesBlockSizeY];
                for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
                {
                    cenPts[i] = new SPDXYPoint*[pulsesBlockSizeX];
                    for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                    {
                        cenPts[i][j] = new SPDXYPoint();
                    }
                }
            }

            boost::uint_fast32_t *bbox = new boost::uint_fast32_t[4];

            boost::uint_fast32_t xOffset = 0;
            boost::uint_fast32_t yOffset = 0;

            double blockXOrigin = 0;
            double blockYOrigin = 0;

            if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
            {
                blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getYMax() + (overlap * spdInFile->getBinSize());
            }
            else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
            {
                blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getZenithMin() - (overlap * spdInFile->getBinSize());
            }
            else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
            {
                blockXOrigin = spdInFile->getScanlineIdxMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getScanlineMin() - (overlap * spdInFile->getBinSize());
            }
            
            for(boost::uint_fast32_t i = 0; i < numYFullBlocks; ++i)
            {
                blockMinX = 0;
                blockMaxX = blockXSize;

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
                {
                    blockXOrigin = spdInFile->getScanlineIdxMin() - (overlap * spdInFile->getBinSize());
                }
                
                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);

                    if(binScaling == 0)
                    {
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, processingResolution);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, processingResolution);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }

                    blockXOrigin += blockWidth;

                    blockMinX += blockXSize;
                    blockMaxX += blockXSize;
                }
                if(remainingCols > 0)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    blockMaxX -= (blockXSize-remainingCols);

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);

                    if(binScaling == 0)
                    {
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, processingResolution);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, processingResolution);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }
                }

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockYOrigin -= blockHeight;
                }
                else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    blockYOrigin += blockHeight;
                }
                else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
                {
                    blockYOrigin += blockHeight;
                }
                
                blockMinY += blockYSize;
                blockMaxY += blockYSize;
            }

            if(remainingRows > 0)
            {
                blockMaxY -= (blockYSize-remainingRows);

                blockMinX = 0;
                blockMaxX = blockXSize;

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
                {
                    blockXOrigin = spdInFile->getScanlineIdxMin() - (overlap * spdInFile->getBinSize());
                }
                
                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);

                    if(binScaling == 0)
                    {
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, processingResolution);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, processingResolution);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }

                    blockMinX += blockXSize;
                    blockMaxX += blockXSize;
                    blockXOrigin += blockWidth;
                }
                if(remainingCols > 0)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    blockMaxX -= (blockXSize-remainingCols);

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox, xOffset, yOffset);

                    if(binScaling == 0)
                    {
                        this->populateCentrePoints(cenPts, pulsesBlockSizeX, pulsesBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY, processingResolution);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, processingResolution);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }
                }
            }
            std::cout << "Complete\n";

            for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                {
                    delete pulses[i][j];
                }
                delete[] pulses[i];
            }
            delete[] pulses;

            if(binScaling != 0)
            {
                for(boost::uint_fast32_t i = 0; i < pulsesScaledBlockSizeY; ++i)
                {
                    for(boost::uint_fast32_t j = 0; j < pulsesScaledBlockSizeX; ++j)
                    {
                        delete pulseScaled[i][j];
                        delete cenPts[i][j];
                    }
                    delete[] pulseScaled[i];
                    delete [] cenPts[i];
                }
                delete[] pulseScaled;
                delete[] cenPts;
            }
            else
            {
                for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
                {
                    for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                    {
                        delete cenPts[i][j];
                    }
                    delete [] cenPts[i];
                }
                delete[] cenPts;
            }

            delete[] bbox;
            incReader.close();
        }
		catch(boost::numeric::negative_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::positive_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::bad_numeric_cast& e)
		{
			throw SPDProcessingException(e.what());
		}
        catch(SPDIOException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }

    void SPDProcessDataBlocks::processDataBlocksOutputImage(SPDFile *spdInFile, std::string outImagePath, float processingResolution, boost::uint_fast16_t numImgBands, std::string gdalFormat) throw(SPDProcessingException)
    {
        //GDALAllRegister();
    }

    void SPDProcessDataBlocks::processDataBlocks(SPDFile *spdInFile) throw(SPDProcessingException)
    {
        try
        {
            SPDFileReader reader;
            reader.readHeaderInfo(spdInFile->getFilePath(), spdInFile);
            
            if(spdInFile->getIndexType() == SPD_UPD_TYPE)
            {
                throw SPDProcessingException("The SPD file must have a spatial index. Use spdtranslate.");
            }
            
            if(this->blockXSize == 0)
            {
                this->blockXSize = spdInFile->getNumberBinsX();
            }
            
            if((this->overlap > this->blockXSize) | (this->overlap > this->blockYSize))
            {
                throw SPDProcessingException("The overlap must be smaller than the block size in both axis\'");
            }

            boost::uint_fast64_t nativeXBins = spdInFile->getNumberBinsX();
            boost::uint_fast64_t nativeYBins = spdInFile->getNumberBinsY();

            boost::uint_fast64_t procResXBins = 0;
            boost::uint_fast64_t procResYBins = 0;

            double geoWidth = 0;
            double geoHeight = 0;

            if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
            {
                geoWidth = spdInFile->getXMax() - spdInFile->getXMin();
                geoHeight = spdInFile->getYMax() - spdInFile->getYMin();
            }
            else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
            {
                geoWidth = spdInFile->getAzimuthMax() - spdInFile->getAzimuthMin();
                geoHeight = spdInFile->getZenithMax() - spdInFile->getZenithMin();
            }
            else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
            {
                geoWidth = spdInFile->getScanlineIdxMax() - spdInFile->getScanlineIdxMin();
                geoHeight = spdInFile->getScanlineMax() - spdInFile->getScanlineMin();
            }
            
            procResXBins = spdInFile->getNumberBinsX();
            procResYBins = spdInFile->getNumberBinsY();

            boost::uint_fast32_t numXFullBlocks = floor(((double)nativeXBins)/this->blockXSize);
            boost::uint_fast32_t numYFullBlocks = floor(((double)nativeYBins)/this->blockYSize);

            boost::uint_fast32_t remainingCols = nativeXBins - (numXFullBlocks * this->blockXSize);
            boost::uint_fast32_t remainingRows = nativeYBins - (numYFullBlocks * this->blockYSize);

            double blockMinX = 0;
            double blockMaxX = 0;
            double blockMinY = 0;
            double blockMaxY = 0;

            double blockWidth = blockXSize * spdInFile->getBinSize();
            double blockHeight = blockYSize * spdInFile->getBinSize();


            boost::uint_fast32_t numBlocks = numYFullBlocks * numXFullBlocks;
            if(remainingCols > 0)
            {
                numBlocks += numYFullBlocks;
            }
            if(remainingRows > 0)
            {
                numBlocks += numXFullBlocks;
                if(remainingCols > 0)
                {
                    numBlocks += 1;
                }
            }
            boost::uint_fast32_t cBlocksIdx = 1;

            blockMinY = 0;
            blockMaxY = blockYSize;

            SPDFileIncrementalReader incReader;
            incReader.open(spdInFile);

            std::vector<SPDPulse*> *pulses = new std::vector<SPDPulse*>();

            boost::uint_fast32_t *bbox = new boost::uint_fast32_t[4];

            boost::uint_fast32_t xOffset = 0;
            boost::uint_fast32_t yOffset = 0;

            double blockXOrigin = 0;
            double blockYOrigin = 0;

            if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
            {
                blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getYMax() + (overlap * spdInFile->getBinSize());
            }
            else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
            {
                blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getZenithMin() - (overlap * spdInFile->getBinSize());
            }
            else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
            {
                blockXOrigin = spdInFile->getScanlineIdxMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getScanlineMin() - (overlap * spdInFile->getBinSize());
            }
            
            for(boost::uint_fast32_t i = 0; i < numYFullBlocks; ++i)
            {
                blockMinX = 0;
                blockMaxX = blockXSize;

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
                {
                    blockXOrigin = spdInFile->getScanlineIdxMin() - (overlap * spdInFile->getBinSize());
                }
                
                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox);

                    dataBlockProcessor->processDataBlock(spdInFile, pulses);
                    this->clearPulses(pulses);

                    blockXOrigin += blockWidth;

                    blockMinX += blockXSize;
                    blockMaxX += blockXSize;
                }
                if(remainingCols > 0)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    blockMaxX -= (blockXSize-remainingCols);

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox);

                    dataBlockProcessor->processDataBlock(spdInFile, pulses);
                    this->clearPulses(pulses);
                }

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockYOrigin -= blockHeight;
                }
                else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    blockYOrigin += blockHeight;
                }
                else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
                {
                    blockYOrigin += blockHeight;
                }
                
                blockMinY += blockYSize;
                blockMaxY += blockYSize;
            }

            if(remainingRows > 0)
            {
                blockMaxY -= (blockYSize-remainingRows);

                blockMinX = 0;
                blockMaxX = blockXSize;

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_SCAN_IDX)
                {
                    blockXOrigin = spdInFile->getScanlineIdxMin() - (overlap * spdInFile->getBinSize());
                }
                
                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox);

                    dataBlockProcessor->processDataBlock(spdInFile, pulses);
                    this->clearPulses(pulses);

                    blockMinX += blockXSize;
                    blockMaxX += blockXSize;
                    blockXOrigin += blockWidth;
                }
                if(remainingCols > 0)
                {
                    if(this->printProgress)
                    {
                        std::cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    blockMaxX -= (blockXSize-remainingCols);

                    bbox[0] = blockMinX;
                    bbox[1] = blockMinY;
                    bbox[2] = blockMaxX;
                    bbox[3] = blockMaxY;

                    xOffset = 0;
                    yOffset = 0;

                    if((((long)blockMinX)-((int)overlap)) < 0)
                    {
                        bbox[0] = 0;
                        xOffset = (((long)blockMinX)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[0] = blockMinX-overlap;
                        xOffset = 0;
                    }
                    if((((long)blockMinY)-((int)overlap)) < 0)
                    {
                        bbox[1] = 0;
                        yOffset = (((long)blockMinY)-((int)overlap)) * (-1);
                    }
                    else
                    {
                        bbox[1] = blockMinY-overlap;
                        yOffset = 0;
                    }

                    if((blockMaxX + overlap) > nativeXBins)
                    {
                        bbox[2] = nativeXBins;
                    }
                    else
                    {
                        bbox[2] = blockMaxX + overlap;
                    }

                    if((blockMaxY + overlap) > nativeYBins)
                    {
                        bbox[3] = nativeYBins;
                    }
                    else
                    {
                        bbox[3] = blockMaxY + overlap;
                    }

                    incReader.readPulseDataBlock(pulses, bbox);

                    dataBlockProcessor->processDataBlock(spdInFile, pulses);
                    this->clearPulses(pulses);
                }
            }
            std::cout << "Complete\n";

            delete pulses;

            delete[] bbox;
            incReader.close();
        }
		catch(boost::numeric::negative_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::positive_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::bad_numeric_cast& e)
		{
			throw SPDProcessingException(e.what());
		}
        catch(SPDIOException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }

    void SPDProcessDataBlocks::removeNullPulses(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize)
    {
        for(boost::uint_fast32_t i = 0; i < ySize; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < xSize; ++j)
            {
                if(pulses[i][j]->size() > 0)
                {
                    for(std::vector<SPDPulse*>::iterator iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); )
                    {
                        if((*iterPulses) ==  NULL)
                        {
                            iterPulses = pulses[i][j]->erase(iterPulses);
                        }
                        else
                        {
                            ++iterPulses;
                        }
                    }
                }
            }
        }
    }

    void SPDProcessDataBlocks::clearPulses(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize)
    {
        SPDPulseUtils pulseUtils;
        for(boost::uint_fast32_t i = 0; i < ySize; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < xSize; ++j)
            {
                if(pulses[i][j]->size() > 0)
                {
                    for(std::vector<SPDPulse*>::iterator iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                    {
                        pulseUtils.deleteSPDPulse(*iterPulses);
                    }
                    pulses[i][j]->clear();
                }
            }
        }
    }

    void SPDProcessDataBlocks::clearPulses(std::vector<SPDPulse*> *pulses)
    {
        SPDPulseUtils pulseUtils;
        for(std::vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
        {
            pulseUtils.deleteSPDPulse(*iterPulses);
        }
        pulses->clear();
    }

    void SPDProcessDataBlocks::clearPulsesNoDelete(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize)
    {
        SPDPulseUtils pulseUtils;
        for(boost::uint_fast32_t i = 0; i < ySize; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < xSize; ++j)
            {
                pulses[i][j]->clear();
            }
        }
    }

    void SPDProcessDataBlocks::populateCentrePoints(SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, double xOrigin, double yOrigin, float binRes)
    {
        double xVal = xOrigin + (binRes/2);
        double yVal = yOrigin - (binRes/2);

        for(boost::uint_fast32_t i = 0; i < ySize; ++i)
        {
            xVal = xOrigin + (binRes/2);
            for(boost::uint_fast32_t j = 0; j < xSize; ++j)
            {
                cenPts[i][j]->x = xVal;
                cenPts[i][j]->y = yVal;
                
                xVal += binRes;
            }
            yVal -= binRes;
        }
    }

    void SPDProcessDataBlocks::populateFromImage(float ***imageDataBlock, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t numImgBands, GDALRasterBand **imgBands, double imgXOrigin, double imgYOrigin, float imgRes, double blockXOrigin, double blockYOrigin) throw(SPDProcessingException)
    {
        std::cout.precision(12);
        /*
        std::cout << "xSize = " << xSize << std::endl;
        std::cout << "ySize = " << ySize << std::endl;
        std::cout << "Num Image Bands = " << numImgBands << std::endl;
        std::cout << "Image Resolution = " << imgRes << std::endl;
        */
        try
        {
            // Set image data block to zero.
            for(boost::uint_fast32_t i = 0; i < ySize; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < xSize; ++j)
                {
                    for(boost::uint_fast16_t n = 0; n < numImgBands; ++n)
                    {
                        imageDataBlock[i][j][n] = 0.0;
                    }
                }
            }
            
            
            // Define the bounds of the block and image
            boost::uint_fast32_t imgXSize = imgBands[0]->GetXSize();
            boost::uint_fast32_t imgYSize = imgBands[0]->GetYSize();

            double blockBRX = blockXOrigin + (xSize * imgRes);
            double blockBRY = blockYOrigin - (ySize * imgRes);
            /*
            std::cout << "blockXOrigin = " << blockXOrigin << std::endl;
            std::cout << "blockYOrigin = " << blockYOrigin << std::endl;
            std::cout << "blockBRX = " << blockBRX << std::endl;
            std::cout << "blockBRY = " << blockBRY << std::endl;
            */
            double imgBRX = imgXOrigin + (imgXSize * imgRes);
            double imgBRY = imgYOrigin - (imgYSize * imgRes);
            /*
            std::cout << "imgXOrigin = " << imgXOrigin << std::endl;
            std::cout << "imgYOrigin = " << imgYOrigin << std::endl;
            std::cout << "imgBRX = " << imgBRX << std::endl;
            std::cout << "imgBRY = " << imgBRY << std::endl;
            */
            // Block and Image offsets - initialised.
            
            boost::uint_fast32_t blockOffsetXTL = 0;
            boost::uint_fast32_t blockOffsetYTL = 0;
            boost::uint_fast32_t blockOffsetXBR = 0;
            boost::uint_fast32_t blockOffsetYBR = 0;
            
            boost::uint_fast32_t imgOffsetXTL = 0;
            boost::uint_fast32_t imgOffsetYTL = 0;
            boost::uint_fast32_t imgOffsetXBR = 0;
            boost::uint_fast32_t imgOffsetYBR = 0;
                        
            // Calc overlap between block and image.
            double tlXDiff = 0;
            double tlYDiff = 0;
            double brXDiff = 0;
            double brYDiff = 0;
            
            if(blockXOrigin > imgXOrigin)
            {
                tlXDiff = blockXOrigin - imgXOrigin;
                blockOffsetXTL = 0;
                imgOffsetXTL = boost::numeric_cast<boost::uint_fast32_t>(tlXDiff/imgRes);
            }
            else
            {
                tlXDiff = imgXOrigin - blockXOrigin;
                blockOffsetXTL = boost::numeric_cast<boost::uint_fast32_t>(tlXDiff/imgRes);
                imgOffsetXTL = 0;
            }
            
            if(imgYOrigin > blockYOrigin)
            {
                tlYDiff = imgYOrigin - blockYOrigin;
                blockOffsetYTL = 0;
                imgOffsetYTL = boost::numeric_cast<boost::uint_fast32_t>(tlYDiff/imgRes);
            }
            else
            {
                tlYDiff = blockYOrigin - imgYOrigin;
                blockOffsetYTL = boost::numeric_cast<boost::uint_fast32_t>(tlYDiff/imgRes);
                imgOffsetYTL = 0;
            }
            
            /*
            std::cout << "blockOffsetXTL = " << blockOffsetXTL << std::endl;
            std::cout << "blockOffsetYTL = " << blockOffsetYTL << std::endl;
            
            std::cout << "imgOffsetXTL = " << imgOffsetXTL << std::endl;
            std::cout << "imgOffsetYTL = " << imgOffsetYTL << std::endl;
            */
            
            if(imgBRX > blockBRX)
            {
                brXDiff = imgBRX - blockBRX;
                boost::uint_fast32_t diff = boost::numeric_cast<boost::uint_fast32_t>(brXDiff/imgRes);
                blockOffsetXBR = xSize;
                if(diff > imgXSize)
                {
                    imgOffsetXBR = 0;
                }
                else 
                {
                    imgOffsetXBR = imgXSize - diff;
                }
            }
            else
            {
                brXDiff = blockBRX - imgBRX;
                boost::uint_fast32_t diff = boost::numeric_cast<boost::uint_fast32_t>(brXDiff/imgRes);
                if(diff > xSize)
                {
                    blockOffsetXBR = 0;
                }
                else 
                {
                    blockOffsetXBR = xSize - diff;
                }
                imgOffsetXBR = imgXSize;
            }
            
            if(blockBRY > imgBRY)
            {
                brYDiff = blockBRY - imgBRY;
                boost::uint_fast32_t diff = boost::numeric_cast<boost::uint_fast32_t>(brYDiff/imgRes);
                blockOffsetYBR = ySize;
                if(diff > imgYSize)
                {
                    imgOffsetYBR = 0;
                }
                else 
                {
                    imgOffsetYBR = imgYSize - diff;
                }
            }
            else
            {
                brYDiff = imgBRY - blockBRY;
                boost::uint_fast32_t diff = boost::numeric_cast<boost::uint_fast32_t>(brYDiff/imgRes);
                if(diff > ySize)
                {
                    blockOffsetYBR = 0;
                }
                else 
                {
                    blockOffsetYBR = ySize - diff;
                }
                imgOffsetYBR = imgYSize;
            }
            
            /*
            std::cout << "blockOffsetXBR = " << blockOffsetXBR << std::endl;
            std::cout << "blockOffsetYBR = " << blockOffsetYBR << std::endl;
            
            std::cout << "imgOffsetXBR = " << imgOffsetXBR << std::endl;
            std::cout << "imgOffsetYBR = " << imgOffsetYBR << std::endl;
            */
            
            if(blockOffsetXTL > blockOffsetXBR)
            {
                return;
            }
            
            if(blockOffsetYTL > blockOffsetYBR)
            {
                return;
            }
            
            if(imgOffsetXTL > imgOffsetXBR)
            {
                return;
            }
            
            if(imgOffsetYTL > imgOffsetYBR)
            {
                return;
            }
            
            boost::uint_fast32_t blockWidth = blockOffsetXBR - blockOffsetXTL;
            boost::uint_fast32_t blockHeight = blockOffsetYBR - blockOffsetYTL;
            
            boost::uint_fast32_t imgWidth = imgOffsetXBR - imgOffsetXTL;
            boost::uint_fast32_t imgHeight = imgOffsetYBR - imgOffsetYTL;
            
            /*
            std::cout << "blockWidth = " << blockWidth << std::endl;
            std::cout << "blockHeight = " << blockHeight << std::endl;
            
            std::cout << "imgWidth = " << imgWidth << std::endl;
            std::cout << "imgHeight = " << imgHeight << std::endl;
            */
            
            if(blockWidth > imgWidth)
            {
                blockWidth = imgWidth;
            }
            else if(imgWidth > blockWidth)
            {
                imgWidth = blockWidth;
            }
            
            if(blockHeight > imgHeight)
            {
                blockHeight = imgHeight;
            }
            else if(imgHeight > blockHeight)
            {
                imgHeight = blockHeight;
            }
            
            /*
            std::cout << "blockWidth = " << blockWidth << std::endl;
            std::cout << "blockHeight = " << blockHeight << std::endl;
            
            std::cout << "imgWidth = " << imgWidth << std::endl;
            std::cout << "imgHeight = " << imgHeight << std::endl;
            */
            
            float *data = new float[blockWidth];
            
            for(boost::uint_fast32_t i = 0, y = blockOffsetYTL; i < blockHeight; ++i, ++y)
            {
                for(boost::uint_fast16_t n = 0; n < numImgBands; ++n)
                {
                    imgBands[n]->RasterIO(GF_Read, imgOffsetXTL, (imgOffsetYTL+i), imgWidth, 1, data, imgWidth, 1, GDT_Float32, 0, 0);
                    
                    for(boost::uint_fast32_t j = 0, x = blockOffsetXTL; j < blockWidth; ++j, ++x)
                    {
                        /*
                        std::cout << "y = " << y << std::endl;
                        std::cout << "x = " << x << std::endl;
                        std::cout << "n = " << n << std::endl;
                        std::cout << "j = " << j << std::endl << std::endl;
                        */
                        imageDataBlock[y][x][n] = data[j];
                    }
                }
            }
            delete[] data;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        catch(boost::numeric::negative_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::positive_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::bad_numeric_cast& e)
		{
			throw SPDProcessingException(e.what());
		}
    }
    
    void SPDProcessDataBlocks::resetImageBlock2Zeros(float ***imageDataBlock, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t numImgBands)
    {
        for(boost::uint_fast32_t i = 0; i < ySize; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < xSize; ++j)
            {
                for(boost::uint_fast16_t n = 0; n < numImgBands; ++n)
                {
                    imageDataBlock[i][j][n] = 0.0;
                }
            }
        }
    }
    
    void SPDProcessDataBlocks::writeImageData(GDALRasterBand **imageBands, float ***imageDataBlock, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t numImgBands, boost::uint_fast32_t startBinX, boost::uint_fast32_t startBinY, boost::uint_fast32_t startIdxX, boost::uint_fast32_t startIdxY)throw(SPDProcessingException)
    {
        try
        {
            float *imgRowData = new float[xSize];
            
            for(boost::uint_fast32_t i = 0, y = startBinY, imgY = startIdxY; i < ySize; ++i, ++y, ++imgY)
            {
                for(boost::uint_fast16_t n = 0; n < numImgBands; ++n)
                {
                    for(boost::uint_fast32_t j = 0, x = startBinX; j < xSize; ++j, ++x)
                    {
                        imgRowData[j] = imageDataBlock[y][x][n];
                    }
                    imageBands[n]->RasterIO(GF_Write, startIdxX, imgY, xSize, 1, imgRowData, xSize, 1, GDT_Float32, 0, 0);
                } 
            }

        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
    }
    
    SPDProcessDataBlocks::~SPDProcessDataBlocks()
    {

    }


}
