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

    SPDProcessDataBlocks::SPDProcessDataBlocks(SPDDataBlockProcessor *dataBlockProcessor, boost::uint_fast32_t overlap, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize, bool printProgress): dataBlockProcessor(NULL), overlap(0), blockXSize(0), blockYSize(0), printProgress(true)

    {
        this->dataBlockProcessor = dataBlockProcessor;
        this->overlap = overlap;
        this->blockXSize = blockXSize;
        this->blockYSize = blockYSize;
        this->printProgress = printProgress;
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

    void SPDProcessDataBlocks::processDataBlocksGridPulsesInputImage(SPDFile *spdInFile, string outFile, string imageFilePath) throw(SPDProcessingException)
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
                string message = string("Could not open image ") + imageFilePath;
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

            //cout << "Geo: [" << geoWidth << "," << geoHeight << "]\n";

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
                    cerr << "Native Res: " << spdInFile->getBinSize() << endl;
                    cerr << "Process Res: " << processingResolution << endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resoltion.");
                }
                usingNativeRes = false;
                procResXBins = numeric_cast<boost::uint_fast64_t>(geoWidth / processingResolution)+1;
                procResYBins = numeric_cast<boost::uint_fast64_t>(geoHeight / processingResolution)+1;

                scaleDown = false;
                binScaling = numeric_cast<boost::uint_fast32_t>(processingResolution/spdInFile->getBinSize());
            }
            else
            {
                if(fmod(spdInFile->getBinSize(), processingResolution) != 0)
                {
                    cerr << "Native Res: " << spdInFile->getBinSize() << endl;
                    cerr << "Process Res: " << processingResolution << endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resolution.");
                }
                usingNativeRes = false;
                procResXBins = numeric_cast<boost::uint_fast64_t>(geoWidth / processingResolution)+1;
                procResYBins = numeric_cast<boost::uint_fast64_t>(geoHeight / processingResolution)+1;

                scaleDown = true;
                binScaling = numeric_cast<boost::uint_fast32_t>(spdInFile->getBinSize()/processingResolution);
            }

            //cout << "Bin Scaling: " << binScaling << endl;
            //cout << "Process Bins: [" << procResXBins << "," << procResYBins << "]\n";

            if(usingNativeRes)
            {
                cout << "Using native resolution for processing\n";
            }
            else
            {
                cout << "Resampling native resolution\n";
                if(scaleDown)
                {
                    cout << "Scaling down\n";
                }
                else
                {
                    cout << "Scaling up\n";
                }
            }

            boost::uint_fast32_t numXFullBlocks = floor(((double)nativeXBins)/this->blockXSize);
            boost::uint_fast32_t numYFullBlocks = floor(((double)nativeYBins)/this->blockYSize);

            //cout << "Number of full blocks: [" << numXFullBlocks << "," << numYFullBlocks << "]\n";

            boost::uint_fast32_t remainingCols = nativeXBins - (numXFullBlocks * this->blockXSize);
            boost::uint_fast32_t remainingRows = nativeYBins - (numYFullBlocks * this->blockYSize);

            //cout << "Remainder: [" << remainingCols << "," << remainingRows << "]\n";

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

            //cout << "Native block size: [" << this->blockXSize << "," << this->blockXSize << "]\n";
            //cout << "Process block size: [" << procResXBlockSize << "," << procResYBlockSize << "]\n";

            //cout << "Block Size: [" << blockWidth << "," << blockHeight << "]\n";

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

            vector<SPDPulse*> ***pulses = new vector<SPDPulse*>**[pulsesBlockSizeY];
            for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
            {
                pulses[i] = new vector<SPDPulse*>*[pulsesBlockSizeX];
                for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                {
                    pulses[i][j] = new vector<SPDPulse*>();
                }
            }


            SPDXYPoint ***cenPts = NULL;
            vector<SPDPulse*> ***pulseScaled = NULL;
            float ***imageBlockVals = NULL;
            if(binScaling != 0)
            {
                pulseScaled = new vector<SPDPulse*>**[pulsesScaledBlockSizeY];
                cenPts = new SPDXYPoint**[pulsesScaledBlockSizeY];
                imageBlockVals = new float**[pulsesScaledBlockSizeY];
                for(boost::uint_fast32_t i = 0; i < pulsesScaledBlockSizeY; ++i)
                {
                    pulseScaled[i] = new vector<SPDPulse*>*[pulsesScaledBlockSizeX];
                    cenPts[i] = new SPDXYPoint*[pulsesScaledBlockSizeX];
                    imageBlockVals[i] = new float*[pulsesScaledBlockSizeX];
                    for(boost::uint_fast32_t j = 0; j < pulsesScaledBlockSizeX; ++j)
                    {
                        pulseScaled[i][j] = new vector<SPDPulse*>();
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
            else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
            {
                blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getZenithMin() - (overlap * spdInFile->getBinSize());
            }

            for(boost::uint_fast32_t i = 0; i < numYFullBlocks; ++i)
            {
                blockMinX = 0;
                blockMaxX = blockXSize;

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }

                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }

                    //cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin <<"]\n";

                    //cout << "Block Size [" << blockXSize << "," << blockYSize << "]\n";

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
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    blockMaxX -= (blockXSize-remainingCols);
                    //cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin <<"]\n";

                    //cout << "Block Size [" << blockXSize << "," << blockYSize << "]\n";

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
                else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
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
                else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }

                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }

                    //cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin << "]\n";

                    //cout << "Block Size [" << blockXSize << "," << blockYSize << "]\n";

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
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    blockMaxX -= (blockXSize-remainingCols);

                    //cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin <<"]\n";

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
            cout << "Complete\n";
            
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
		catch(negative_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(positive_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(bad_numeric_cast& e)
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

    void SPDProcessDataBlocks::processDataBlocksGridPulsesOutputImage(SPDFile *spdInFile, string outImagePath, float processingResolution, boost::uint_fast16_t numImgBands, string gdalFormat) throw(SPDProcessingException)
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
            
            //cout << "Geo: [" << geoWidth << "," << geoHeight << "]\n";
            
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
                    cerr << "Native Res: " << spdInFile->getBinSize() << endl;
                    cerr << "Process Res: " << processingResolution << endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resoltion.");
                }
                usingNativeRes = false;
                procResXBins = ceil(geoWidth / processingResolution)+1;
                procResYBins = ceil(geoHeight / processingResolution)+1;
                
                scaleDown = false;
                binScaling = numeric_cast<boost::uint_fast32_t>(processingResolution/spdInFile->getBinSize());
            }
            else
            {
                if(fmod(spdInFile->getBinSize(), processingResolution) != 0)
                {
                    cerr << "Native Res: " << spdInFile->getBinSize() << endl;
                    cerr << "Process Res: " << processingResolution << endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resoltion.");
                }
                usingNativeRes = false;
                procResXBins = ceil(geoWidth / processingResolution)+1;
                procResYBins = ceil(geoHeight / processingResolution)+1;
                
                scaleDown = true;
                binScaling = numeric_cast<boost::uint_fast32_t>(spdInFile->getBinSize()/processingResolution);
            }
            
            //cout << "Bin Scaling: " << binScaling << endl;
            //cout << "Process Bins: [" << procResXBins << "," << procResYBins << "]\n";
                        
            if(usingNativeRes)
            {
                cout << "Using native resolution for processing\n";
            }
            else
            {
                cout << "Resampling native resolution\n";
                if(scaleDown)
                {
                    cout << "Scaling down\n";
                }
                else
                {
                    cout << "Scaling up\n";
                }
            }
            
            boost::uint_fast32_t numXFullBlocks = floor(((double)nativeXBins)/this->blockXSize);
            boost::uint_fast32_t numYFullBlocks = floor(((double)nativeYBins)/this->blockYSize);
            
            //cout << "Number of full blocks: [" << numXFullBlocks << "," << numYFullBlocks << "]\n";
            
            boost::uint_fast32_t remainingCols = nativeXBins - (numXFullBlocks * this->blockXSize);
            boost::uint_fast32_t remainingRows = nativeYBins - (numYFullBlocks * this->blockYSize);
            
            //cout << "Remainder: [" << remainingCols << "," << remainingRows << "]\n";
            
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
            
            //cout << "Native block size: [" << this->blockXSize << "," << this->blockYSize << "]\n";
            //cout << "Process block size: [" << procResXBlockSize << "," << procResYBlockSize << "]\n";
            //cout << "Block Size: [" << blockWidth << "," << blockHeight << "]\n";
            
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
                    remainingColsScaled = ceil(remainingCols * binScaling);
                    remainingRowsScaled = ceil(remainingRows * binScaling);
                }
                else
                {
                    scaledOverlap = this->overlap / binScaling;
                    remainingColsScaled = ceil(remainingCols / binScaling);
                    remainingRowsScaled = ceil(remainingRows / binScaling);
                }
            }
            
            boost::uint_fast32_t pulsesScaledBlockSizeX = procResXBlockSize + (2 * scaledOverlap);
            boost::uint_fast32_t pulsesScaledBlockSizeY = procResYBlockSize + (2 * scaledOverlap);
            
            //cout << "Pulses Block Size: [" << pulsesBlockSizeX << "," << pulsesBlockSizeY << "]\n";
            //cout << "Processing Pulses Block Size: [" << procResXBlockSize << "," << procResYBlockSize << "]\n";
            //cout << "Remaining: [" << remainingColsScaled << "," << remainingRowsScaled << "]\n";
            
            boost::uint_fast32_t imageXSize = (procResXBlockSize * numXFullBlocks) + remainingColsScaled;
            boost::uint_fast32_t imageYSize = (procResYBlockSize * numYFullBlocks) + remainingRowsScaled;
            
            /****** CREATE A NEW GDALDATASET *******/
            GDALDriver *gdalDriver = GetGDALDriverManager()->GetDriverByName(gdalFormat.c_str());
			if(gdalDriver == NULL)
			{
                string message = gdalFormat + string(" gdal driver cannot be found.");
				throw SPDProcessingException(message);
			}
            char **papszMetadata;
            papszMetadata = gdalDriver->GetMetadata();
            if( CSLFetchBoolean( papszMetadata, GDAL_DCAP_CREATE, FALSE ) )
            {
                string message = gdalFormat + string(" does not support create method. Select a GDAL driver which does (see http://www.gdal.org/formats_list.html).");
            }
            GDALDataset *outImage = gdalDriver->Create(outImagePath.c_str(), imageXSize, imageYSize, numImgBands, GDT_Float32, papszMetadata);
            
            if(outImage == NULL)
            {
                string message = string("Failed to create image ") + outImagePath;
				throw SPDProcessingException(message);
            }
            
            //cout << spdInFile << endl;
            
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
            vector<string> bandNames = this->dataBlockProcessor->getImageBandDescriptions();
            
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
                        
            vector<SPDPulse*> ***pulses = new vector<SPDPulse*>**[pulsesBlockSizeY];
            for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
            {
                pulses[i] = new vector<SPDPulse*>*[pulsesBlockSizeX];
                for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                {
                    pulses[i][j] = new vector<SPDPulse*>();
                }
            }
            
            SPDXYPoint ***cenPts = NULL;
            vector<SPDPulse*> ***pulseScaled = NULL;
            float ***imageBlockVals = NULL;
            if(binScaling != 0)
            {
                pulseScaled = new vector<SPDPulse*>**[pulsesScaledBlockSizeY];
                cenPts = new SPDXYPoint**[pulsesScaledBlockSizeY];
                imageBlockVals = new float**[pulsesScaledBlockSizeY];
                for(boost::uint_fast32_t i = 0; i < pulsesScaledBlockSizeY; ++i)
                {
                    pulseScaled[i] = new vector<SPDPulse*>*[pulsesScaledBlockSizeX];
                    cenPts[i] = new SPDXYPoint*[pulsesScaledBlockSizeX];
                    imageBlockVals[i] = new float*[pulsesScaledBlockSizeX];
                    for(boost::uint_fast32_t j = 0; j < pulsesScaledBlockSizeX; ++j)
                    {
                        pulseScaled[i][j] = new vector<SPDPulse*>();
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
            else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
            {
                blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getZenithMin() - (overlap * spdInFile->getBinSize());
            }
            
            for(boost::uint_fast32_t i = 0; i < numYFullBlocks; ++i)
            {
                blockMinX = 0;
                blockMaxX = blockXSize;
                
                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }
                
                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
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
                    
                    //cout << "xOffset = " << xOffset << endl;
                    //cout << "yOffset = " << yOffset << endl;
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
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
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
                else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
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
                else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }
                
                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
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
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
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
            cout << "Complete\n";
            
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
		catch(negative_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(positive_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(bad_numeric_cast& e)
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

    void SPDProcessDataBlocks::processDataBlocksGridPulsesOutputSPD(SPDFile *spdInFile, string outFile, float processingResolution) throw(SPDProcessingException)
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

            //cout << "Geo: [" << geoWidth << "," << geoHeight << "]\n";

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
                    cerr << "Native Res: " << spdInFile->getBinSize() << endl;
                    cerr << "Process Res: " << processingResolution << endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resoltion.");
                }
                usingNativeRes = false;
                procResXBins = numeric_cast<boost::uint_fast64_t>(geoWidth / processingResolution)+1;
                procResYBins = numeric_cast<boost::uint_fast64_t>(geoHeight / processingResolution)+1;

                scaleDown = false;
                binScaling = numeric_cast<boost::uint_fast32_t>(processingResolution/spdInFile->getBinSize());
            }
            else
            {
                if(fmod(spdInFile->getBinSize(), processingResolution) != 0)
                {
                    cerr << "Native Res: " << spdInFile->getBinSize() << endl;
                    cerr << "Process Res: " << processingResolution << endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resoltion.");
                }
                usingNativeRes = false;
                procResXBins = numeric_cast<boost::uint_fast64_t>(geoWidth / processingResolution)+1;
                procResYBins = numeric_cast<boost::uint_fast64_t>(geoHeight / processingResolution)+1;

                scaleDown = true;
                binScaling = numeric_cast<boost::uint_fast32_t>(spdInFile->getBinSize()/processingResolution);
            }

            //cout << "Bin Scaling: " << binScaling << endl;
            //cout << "Process Bins: [" << procResXBins << "," << procResYBins << "]\n";

            if(usingNativeRes)
            {
                cout << "Using native resolution for processing\n";
            }
            else
            {
                cout << "Resampling native resolution\n";
                if(scaleDown)
                {
                    cout << "Scaling down\n";
                }
                else
                {
                    cout << "Scaling up\n";
                }
            }

            boost::uint_fast32_t numXFullBlocks = floor(((double)nativeXBins)/this->blockXSize);
            boost::uint_fast32_t numYFullBlocks = floor(((double)nativeYBins)/this->blockYSize);

            //cout << "Number of full blocks: [" << numXFullBlocks << "," << numYFullBlocks << "]\n";

            boost::uint_fast32_t remainingCols = nativeXBins - (numXFullBlocks * this->blockXSize);
            boost::uint_fast32_t remainingRows = nativeYBins - (numYFullBlocks * this->blockYSize);

            //cout << "Remainder: [" << remainingCols << "," << remainingRows << "]\n";

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

            //cout << "Native block size: [" << this->blockXSize << "," << this->blockXSize << "]\n";
            //cout << "Process block size: [" << procResXBlockSize << "," << procResYBlockSize << "]\n";

            //cout << "Block Size: [" << blockWidth << "," << blockHeight << "]\n";



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

            vector<SPDPulse*> ***pulses = new vector<SPDPulse*>**[pulsesBlockSizeY];
            for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
            {
                pulses[i] = new vector<SPDPulse*>*[pulsesBlockSizeX];
                for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                {
                    pulses[i][j] = new vector<SPDPulse*>();
                }
            }


            SPDXYPoint ***cenPts = NULL;
            vector<SPDPulse*> ***pulseScaled = NULL;
            if(binScaling != 0)
            {
                pulseScaled = new vector<SPDPulse*>**[pulsesScaledBlockSizeY];
                cenPts = new SPDXYPoint**[pulsesScaledBlockSizeY];
                for(boost::uint_fast32_t i = 0; i < pulsesScaledBlockSizeY; ++i)
                {
                    pulseScaled[i] = new vector<SPDPulse*>*[pulsesScaledBlockSizeX];
                    cenPts[i] = new SPDXYPoint*[pulsesScaledBlockSizeX];
                    for(boost::uint_fast32_t j = 0; j < pulsesScaledBlockSizeX; ++j)
                    {
                        pulseScaled[i][j] = new vector<SPDPulse*>();
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
            else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
            {
                blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getZenithMin() - (overlap * spdInFile->getBinSize());
            }

            for(boost::uint_fast32_t i = 0; i < numYFullBlocks; ++i)
            {
                blockMinX = 0;
                blockMaxX = blockXSize;

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }

                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }

                    //cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin <<"]\n";

                    //cout << "Block Size [" << blockXSize << "," << blockYSize << "]\n";

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
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, this->blockXSize, this->blockYSize, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
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
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    blockMaxX -= (blockXSize-remainingCols);
                    //cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin <<"]\n";

                    //cout << "Block Size [" << blockXSize << "," << blockYSize << "]\n";

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
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, remainingCols, this->blockYSize, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
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
                else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
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
                else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }

                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }

                    //cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin << "]\n";

                    //cout << "Block Size [" << blockXSize << "," << blockYSize << "]\n";

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
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, this->blockXSize, remainingRows, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
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
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
                    }
                    blockMaxX -= (blockXSize-remainingCols);

                    //cout << "Block [" << blockMinX << "," << blockMaxX << "][" << blockMinY << "," << blockMaxY << "]\n";

                    //cout << "Block Origin [" << blockXOrigin << "," << blockYOrigin <<"]\n";

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
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, remainingCols, remainingRows, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                        this->removeNullPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        fileWriter->writeData(pulses, remainingCols, remainingRows, overlap, overlap, blockMinX, blockMinY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }
                }
            }
            cout << "Complete\n";
            
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
		catch(negative_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(positive_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(bad_numeric_cast& e)
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
                    cerr << "Native Res: " << spdInFile->getBinSize() << endl;
                    cerr << "Process Res: " << processingResolution << endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resoltion.");
                }
                usingNativeRes = false;
                procResXBins = numeric_cast<boost::uint_fast64_t>(geoWidth / processingResolution)+1;
                procResYBins = numeric_cast<boost::uint_fast64_t>(geoHeight / processingResolution)+1;

                scaleDown = false;
                binScaling = numeric_cast<boost::uint_fast32_t>(processingResolution/spdInFile->getBinSize());
            }
            else
            {
                if(fmod(spdInFile->getBinSize(), processingResolution) != 0)
                {
                    cerr << "Native Res: " << spdInFile->getBinSize() << endl;
                    cerr << "Process Res: " << processingResolution << endl;
                    throw SPDProcessingException("The processing resolution must be a multiple of the native resoltion.");
                }
                usingNativeRes = false;
                procResXBins = numeric_cast<boost::uint_fast64_t>(geoWidth / processingResolution)+1;
                procResYBins = numeric_cast<boost::uint_fast64_t>(geoHeight / processingResolution)+1;

                scaleDown = true;
                binScaling = numeric_cast<boost::uint_fast32_t>(spdInFile->getBinSize()/processingResolution);
            }

            if(usingNativeRes)
            {
                cout << "Using native resolution for processing\n";
            }
            else
            {
                cout << "Resampling native resolution\n";
                if(scaleDown)
                {
                    cout << "Scaling down\n";
                }
                else
                {
                    cout << "Scaling up\n";
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

            vector<SPDPulse*> ***pulses = new vector<SPDPulse*>**[pulsesBlockSizeY];
            for(boost::uint_fast32_t i = 0; i < pulsesBlockSizeY; ++i)
            {
                pulses[i] = new vector<SPDPulse*>*[pulsesBlockSizeX];
                for(boost::uint_fast32_t j = 0; j < pulsesBlockSizeX; ++j)
                {
                    pulses[i][j] = new vector<SPDPulse*>();
                }
            }


            SPDXYPoint ***cenPts = NULL;
            vector<SPDPulse*> ***pulseScaled = NULL;
            if(binScaling != 0)
            {
                pulseScaled = new vector<SPDPulse*>**[pulsesScaledBlockSizeY];
                cenPts = new SPDXYPoint**[pulsesScaledBlockSizeY];
                for(boost::uint_fast32_t i = 0; i < pulsesScaledBlockSizeY; ++i)
                {
                    pulseScaled[i] = new vector<SPDPulse*>*[pulsesScaledBlockSizeX];
                    cenPts[i] = new SPDXYPoint*[pulsesScaledBlockSizeX];
                    for(boost::uint_fast32_t j = 0; j < pulsesScaledBlockSizeX; ++j)
                    {
                        pulseScaled[i][j] = new vector<SPDPulse*>();
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
            else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
            {
                blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getZenithMin() - (overlap * spdInFile->getBinSize());
            }

            for(boost::uint_fast32_t i = 0; i < numYFullBlocks; ++i)
            {
                blockMinX = 0;
                blockMaxX = blockXSize;

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }

                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
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
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
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
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
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
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }
                }

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockYOrigin -= blockHeight;
                }
                else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
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
                else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }

                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
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
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
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
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
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
                        dataBlockProcessor->processDataBlock(spdInFile, pulses, cenPts, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                    }
                    else
                    {
                        this->populateCentrePoints(cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        gridData.reGridData(spdInFile->getIndexType(), pulses, pulsesBlockSizeX, pulsesBlockSizeY, pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY, blockXOrigin, blockYOrigin, processingResolution);
                        dataBlockProcessor->processDataBlock(spdInFile, pulseScaled, cenPts, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                        this->clearPulses(pulses, pulsesBlockSizeX, pulsesBlockSizeY);
                        this->clearPulsesNoDelete(pulseScaled, pulsesScaledBlockSizeX, pulsesScaledBlockSizeY);
                    }
                }
            }
            cout << "Complete\n";

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
		catch(negative_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(positive_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(bad_numeric_cast& e)
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

    void SPDProcessDataBlocks::processDataBlocksOutputImage(SPDFile *spdInFile, string outImagePath, float processingResolution, boost::uint_fast16_t numImgBands, string gdalFormat) throw(SPDProcessingException)
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

            vector<SPDPulse*> *pulses = new vector<SPDPulse*>();

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
            else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
            {
                blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                blockYOrigin = spdInFile->getZenithMin() - (overlap * spdInFile->getBinSize());
            }

            for(boost::uint_fast32_t i = 0; i < numYFullBlocks; ++i)
            {
                blockMinX = 0;
                blockMaxX = blockXSize;

                if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getXMin() - (overlap * spdInFile->getBinSize());
                }
                else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }

                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
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
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
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
                else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
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
                else if(spdInFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    blockXOrigin = spdInFile->getAzimuthMin() - (overlap * spdInFile->getBinSize());
                }

                for(boost::uint_fast32_t j = 0; j < numXFullBlocks; ++j)
                {
                    if(this->printProgress)
                    {
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
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
                        cout << "Processing block " << cBlocksIdx++ << " of " << numBlocks << " blocks\n";
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
            cout << "Complete\n";

            delete pulses;

            delete[] bbox;
            incReader.close();
        }
		catch(negative_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(positive_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(bad_numeric_cast& e)
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

    void SPDProcessDataBlocks::removeNullPulses(vector<SPDPulse*> ***pulses, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize)
    {
        for(boost::uint_fast32_t i = 0; i < ySize; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < xSize; ++j)
            {
                if(pulses[i][j]->size() > 0)
                {
                    for(vector<SPDPulse*>::iterator iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); )
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

    void SPDProcessDataBlocks::clearPulses(vector<SPDPulse*> ***pulses, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize)
    {
        SPDPulseUtils pulseUtils;
        for(boost::uint_fast32_t i = 0; i < ySize; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < xSize; ++j)
            {
                if(pulses[i][j]->size() > 0)
                {
                    for(vector<SPDPulse*>::iterator iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                    {
                        pulseUtils.deleteSPDPulse(*iterPulses);
                    }
                    pulses[i][j]->clear();
                }
            }
        }
    }

    void SPDProcessDataBlocks::clearPulses(vector<SPDPulse*> *pulses)
    {
        SPDPulseUtils pulseUtils;
        for(vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
        {
            pulseUtils.deleteSPDPulse(*iterPulses);
        }
        pulses->clear();
    }

    void SPDProcessDataBlocks::clearPulsesNoDelete(vector<SPDPulse*> ***pulses, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize)
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

            boost::uint_fast32_t imgXSize = imgBands[0]->GetXSize();
            boost::uint_fast32_t imgYSize = imgBands[0]->GetYSize();

            double blockBRX = blockXOrigin + (xSize * imgRes);
            double blockBRY = blockYOrigin - (ySize * imgRes);

            double imgBRX = imgXOrigin + (imgXSize * imgRes);
            double imgBRY = imgYOrigin - (imgYSize * imgRes);

            bool tlInside = false;
            bool brInside = false;

            if((blockXOrigin > imgXOrigin) & (blockYOrigin < imgYOrigin))
            {
                tlInside = true;
            }

            if((blockBRX < imgBRX) & (blockBRY > imgBRY))
            {
                brInside = true;
            }

            if(tlInside | brInside)
            {
                boost::uint_fast32_t imgOffX = 0;
                boost::uint_fast32_t imgOffY = 0;
                boost::uint_fast32_t dataOffX = 0;
                boost::uint_fast32_t dataOffY = 0;

                boost::uint_fast32_t sampleLenX = 0;
                boost::uint_fast32_t sampleLenY = 0;

                double diffX = blockXOrigin - imgXOrigin;
                double diffY = imgYOrigin - blockYOrigin;

                if(diffX == 0)
                {
                    imgOffX = 0;
                    dataOffX = 0;
                }
                else if(diffX > 0)
                {
                    dataOffX = 0;
                    imgOffX = numeric_cast<boost::uint_fast32_t>(diffX/imgRes);
                }
                else if(diffX < 0)
                {
                    imgOffX = 0;
                    diffX = diffX * (-1);
                    dataOffX = numeric_cast<boost::uint_fast32_t>(diffX/imgRes);
                }

                if(diffY == 0)
                {
                    imgOffY = 0;
                    dataOffY = 0;
                }
                else if(diffY > 0)
                {
                    dataOffY = 0;
                    imgOffY = numeric_cast<boost::uint_fast32_t>(diffY/imgRes);
                }
                else if(diffY < 0)
                {
                    imgOffY = 0;
                    diffY = diffY * (-1);
                    dataOffY = numeric_cast<boost::uint_fast32_t>(diffY/imgRes);
                }

                diffX = imgBRX - blockBRX;
                diffY = blockBRY - imgBRY;

                if(diffX == 0)
                {
                    sampleLenX = xSize - dataOffX;
                }
                else if(diffX > 0)
                {
                    sampleLenX = (xSize - numeric_cast<boost::uint_fast32_t>(diffX/imgRes)) - dataOffX;
                }
                else if(diffX < 0)
                {
                    sampleLenX = (imgXSize - numeric_cast<boost::uint_fast32_t>(diffX/imgRes)) - imgOffX;
                }

                if(diffY == 0)
                {
                    sampleLenY = ySize - dataOffY;
                }
                else if(diffY > 0)
                {
                    sampleLenY = (ySize - numeric_cast<boost::uint_fast32_t>(diffY/imgRes)) - dataOffY;
                }
                else if(diffY < 0)
                {
                    sampleLenY = (imgYSize - numeric_cast<boost::uint_fast32_t>(diffY/imgRes)) - imgOffY;
                }

                if((sampleLenX > 0) & (sampleLenY > 0))
                {
                    float *data = new float[sampleLenX];

                    for(boost::uint_fast32_t i = 0, y = dataOffY; i < sampleLenY; ++i, ++y)
                    {
                        for(boost::uint_fast16_t n = 0; n < numImgBands; ++n)
                        {
                            imgBands[n]->RasterIO(GF_Read, imgOffX, (imgOffY+i), sampleLenX, 1, data, sampleLenX, 1, GDT_Float32, 0, 0);

                            for(boost::uint_fast32_t j = 0, x = dataOffX; j < xSize; ++j, ++x)
                            {
                                imageDataBlock[y][x][n] = data[j];
                            }
                        }
                    }
                }
            }
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        catch(negative_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(positive_overflow& e)
		{
			throw SPDProcessingException(e.what());
		}
		catch(bad_numeric_cast& e)
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
