/*
 *  SPDGenerateTiles.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 18/04/2013.
 *  Copyright 2013 SPDLib. All rights reserved.
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

#include "spd/SPDGenerateTiles.h"

namespace spdlib
{


    SPDTilesUtils::SPDTilesUtils() throw(SPDException)
    {

    }

    void SPDTilesUtils::calcFileExtent(std::vector<std::string> inputFiles, double *xMin, double *xMax, double *yMin, double *yMax) throw(SPDProcessingException)
    {
        try
        {
            SPDFileReader reader;
            SPDFile *inSPDFile = NULL;
            bool first = true;
            for(std::vector<std::string>::iterator iterFiles = inputFiles.begin(); iterFiles != inputFiles.end(); ++iterFiles)
            {
                if((*iterFiles) != "")
                {
                    std::cout << "Processing: \'" << *iterFiles << "\'" << std::endl;

                    // STEP 1: Read file header - Need file extent.
                    inSPDFile = new SPDFile(*iterFiles);
                    reader.readHeaderInfo(*iterFiles, inSPDFile);

                    if(first)
                    {
                        *xMin = inSPDFile->getXMin();
                        *xMax = inSPDFile->getXMax();
                        *yMin = inSPDFile->getYMin();
                        *yMax = inSPDFile->getYMax();
                        first = false;
                    }
                    else
                    {
                        if(inSPDFile->getXMin() < (*xMin))
                        {
                            *xMin = inSPDFile->getXMin();
                        }

                        if(inSPDFile->getXMax() > (*xMax))
                        {
                            *xMax = inSPDFile->getXMax();
                        }

                        if(inSPDFile->getYMin() < (*yMin))
                        {
                            *yMin = inSPDFile->getYMin();
                        }

                        if(inSPDFile->getYMax() > (*yMax))
                        {
                            *yMax = inSPDFile->getYMax();
                        }
                    }

                    delete inSPDFile;
                }
            }
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch (SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }

    }

    std::vector<SPDTile*>* SPDTilesUtils::createTiles(double xSize, double ySize, double overlap, double xMin, double xMax, double yMin, double yMax, boost::uint_fast32_t *rows, boost::uint_fast32_t *cols)throw(SPDProcessingException)
    {
        std::vector<SPDTile*> *tiles = NULL;

        try
        {
            if(xSize <= 0)
            {
                throw SPDProcessingException("X tile size must be greater than zero.");
            }
            else if(ySize <= 0)
            {
                throw SPDProcessingException("Y tile size must be greater than zero.");
            }

            if(xMax <= xMin)
            {
                throw SPDProcessingException("xMax must be larger than xMin.");
            }
            else if(yMax <= yMin)
            {
                throw SPDProcessingException("yMax must be larger than yMin.");
            }

            tiles = new std::vector<SPDTile*>();

            double sceneXSize = xMax - xMin;
            double sceneYSize = yMax - yMin;

            boost::uint_fast32_t numCols = ceil(sceneXSize/xSize);
            boost::uint_fast32_t numRows = ceil(sceneYSize/ySize);

            //std::cout << "Number of Columns = " << numCols << std::endl;
            //std::cout << "Number of Rows = " << numRows << std::endl;

            *cols = numCols;
            *rows = numRows;

            double cXMin = xMin;
            double cXMax = cXMin + xSize;
            double cYMin = yMin;
            double cYMax = cYMin + ySize;

            SPDTile *tile = NULL;

            for(boost::uint_fast32_t i = 0; i < numRows; ++i)
            {
                cXMin = xMin;
                cXMax = cXMin + xSize;

                for(boost::uint_fast32_t j = 0; j < numCols; ++j)
                {
                    tile = new SPDTile();

                    tile->col = j+1;
                    tile->row = i+1;

                    tile->xMinCore = cXMin;
                    tile->xMaxCore = cXMax;
                    tile->yMinCore = cYMin;
                    tile->yMaxCore = cYMax;

                    tile->xMin = tile->xMinCore - overlap;
                    tile->xMax = tile->xMaxCore + overlap;
                    tile->yMin = tile->yMinCore - overlap;
                    tile->yMax = tile->yMaxCore + overlap;

                    //std::cout << "[" << tile->col << " " << tile->row << "] ALL: [" << tile->xMin << "," << tile->xMax << "][" << tile->yMin << "," << tile->yMax << "] CORE: [" << tile->xMinCore << "," << tile->xMaxCore << "][" << tile->yMinCore << "," << tile->yMaxCore << "]" << std::endl;
                    cXMin += xSize;
                    cXMax = cXMin + xSize;

                    tiles->push_back(tile);
                }

                cYMin += ySize;
                cYMax = cYMin + ySize;
            }
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }

        return tiles;
    }

    void SPDTilesUtils::exportTiles2XML(std::string outputFile, std::vector<SPDTile*> *tiles, double xSize, double ySize, double overlap, double xMin, double xMax, double yMin, double yMax, boost::uint_fast32_t rows, boost::uint_fast32_t cols)throw(SPDProcessingException)
    {
        SPDTextFileUtilities txtUtils;

        try
        {
            xercesc::XMLPlatformUtils::Initialize();

            xercesc::DOMImplementation* domImpl = xercesc::DOMImplementationRegistry::getDOMImplementation(xercesc::XMLString::transcode("Core"));

            if(domImpl == NULL)
            {
                throw SPDProcessingException("Couldn't create a DOM implementation.");
            }

            xercesc::DOMDocument *xmlDoc = domImpl->createDocument(0, xercesc::XMLString::transcode("tiles"), 0);

            xercesc::DOMElement *tilesRootElem = xmlDoc->getDocumentElement();

            tilesRootElem->setAttribute(xercesc::XMLString::transcode("rows"), xercesc::XMLString::transcode(txtUtils.uInt32bittostring(rows).c_str()));
            tilesRootElem->setAttribute(xercesc::XMLString::transcode("columns"), xercesc::XMLString::transcode(txtUtils.uInt32bittostring(cols).c_str()));

            tilesRootElem->setAttribute(xercesc::XMLString::transcode("xmin"), xercesc::XMLString::transcode(txtUtils.doubletostring(xMin).c_str()));
            tilesRootElem->setAttribute(xercesc::XMLString::transcode("xmax"), xercesc::XMLString::transcode(txtUtils.doubletostring(xMax).c_str()));
            tilesRootElem->setAttribute(xercesc::XMLString::transcode("ymin"), xercesc::XMLString::transcode(txtUtils.doubletostring(yMin).c_str()));
            tilesRootElem->setAttribute(xercesc::XMLString::transcode("ymax"), xercesc::XMLString::transcode(txtUtils.doubletostring(yMax).c_str()));

            tilesRootElem->setAttribute(xercesc::XMLString::transcode("overlap"), xercesc::XMLString::transcode(txtUtils.doubletostring(overlap).c_str()));

            tilesRootElem->setAttribute(xercesc::XMLString::transcode("xtilesize"), xercesc::XMLString::transcode(txtUtils.doubletostring(xSize).c_str()));
            tilesRootElem->setAttribute(xercesc::XMLString::transcode("ytilesize"), xercesc::XMLString::transcode(txtUtils.doubletostring(ySize).c_str()));


            xercesc::DOMElement *tileElem = NULL;

            for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
            {
                tileElem = xmlDoc->createElement(xercesc::XMLString::transcode("tile"));
                tileElem->setAttribute(xercesc::XMLString::transcode("row"), xercesc::XMLString::transcode(txtUtils.uInt32bittostring((*iterTiles)->row).c_str()));
                tileElem->setAttribute(xercesc::XMLString::transcode("col"), xercesc::XMLString::transcode(txtUtils.uInt32bittostring((*iterTiles)->col).c_str()));

                tileElem->setAttribute(xercesc::XMLString::transcode("xmin"), xercesc::XMLString::transcode(txtUtils.doubletostring((*iterTiles)->xMin).c_str()));
                tileElem->setAttribute(xercesc::XMLString::transcode("xmax"), xercesc::XMLString::transcode(txtUtils.doubletostring((*iterTiles)->xMax).c_str()));
                tileElem->setAttribute(xercesc::XMLString::transcode("ymin"), xercesc::XMLString::transcode(txtUtils.doubletostring((*iterTiles)->yMin).c_str()));
                tileElem->setAttribute(xercesc::XMLString::transcode("ymax"), xercesc::XMLString::transcode(txtUtils.doubletostring((*iterTiles)->yMax).c_str()));

                tileElem->setAttribute(xercesc::XMLString::transcode("corexmin"), xercesc::XMLString::transcode(txtUtils.doubletostring((*iterTiles)->xMinCore).c_str()));
                tileElem->setAttribute(xercesc::XMLString::transcode("corexmax"), xercesc::XMLString::transcode(txtUtils.doubletostring((*iterTiles)->xMaxCore).c_str()));
                tileElem->setAttribute(xercesc::XMLString::transcode("coreymin"), xercesc::XMLString::transcode(txtUtils.doubletostring((*iterTiles)->yMinCore).c_str()));
                tileElem->setAttribute(xercesc::XMLString::transcode("coreymax"), xercesc::XMLString::transcode(txtUtils.doubletostring((*iterTiles)->yMaxCore).c_str()));
                tileElem->setAttribute(xercesc::XMLString::transcode("file"), xercesc::XMLString::transcode((*iterTiles)->outFileName.c_str()));

                tilesRootElem->appendChild(tileElem);
            }


            xercesc::DOMLSSerializer *domSerializer = ((xercesc::DOMImplementationLS*)domImpl)->createLSSerializer();

            if (domSerializer->getDomConfig()->canSetParameter(xercesc::XMLUni::fgDOMWRTDiscardDefaultContent, true))
            {
                domSerializer->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMWRTDiscardDefaultContent, true);
            }

            if (domSerializer->getDomConfig()->canSetParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
            {
                domSerializer->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
            }

            domSerializer->writeToURI(tilesRootElem, xercesc::XMLString::transcode(outputFile.c_str()));

            domSerializer->release();

            xmlDoc->release();

            xercesc::XMLPlatformUtils::Terminate();
        }
        catch (const xercesc::OutOfMemoryException &e)
        {
            throw SPDProcessingException(xercesc::XMLString::transcode(e.getMessage()));
        }
        catch (const xercesc::DOMException &e)
        {
            throw SPDProcessingException(xercesc::XMLString::transcode(e.getMessage()));
        }
        catch(xercesc::XMLException &e)
        {
            throw SPDProcessingException(xercesc::XMLString::transcode(e.getMessage()));
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch(std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
    }

    std::vector<SPDTile*>* SPDTilesUtils::importTilesFromXML(std::string inputFile,  boost::uint_fast32_t *rows, boost::uint_fast32_t *cols, double *xSize, double *ySize, double *overlap, double *xMin, double *xMax, double *yMin, double *yMax)throw(SPDProcessingException)
    {
        SPDTextFileUtilities txtUtils;

        std::vector<SPDTile*> *tiles = NULL;

        try
        {
            xercesc::XMLPlatformUtils::Initialize();

            XMLCh *domImplTypeLS = xercesc::XMLString::transcode("LS");
            XMLCh *tilesElementName = xercesc::XMLString::transcode("tiles");
            XMLCh *tileElementName = xercesc::XMLString::transcode("tile");

			xercesc::DOMImplementation *domImpl = xercesc::DOMImplementationRegistry::getDOMImplementation(domImplTypeLS);
			if(domImpl == NULL)
			{
				throw SPDProcessingException("DOMImplementation is NULL");
			}
			
			// Create Parser
			xercesc::DOMLSParser* domParser = ((xercesc::DOMImplementationLS*)domImpl)->createLSParser(xercesc::DOMImplementationLS::MODE_SYNCHRONOUS, 0);
			xercesc::ErrorHandler *errHandler = (xercesc::ErrorHandler*) new xercesc::HandlerBase();
			domParser->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMErrorHandler, errHandler);
			
			// Open Document
			xercesc::DOMDocument *doc = domParser->parseURI(inputFile.c_str());
			
			// Get the Root element
			xercesc::DOMElement *rootElement = doc->getDocumentElement();
            //std::cout << "Root Element: " << xercesc::XMLString::transcode(rootElement->getTagName()) << std::endl;
			if(!xercesc::XMLString::equals(rootElement->getTagName(), tilesElementName))
			{
				throw SPDProcessingException("Incorrect root element; Root element should be \"tiles\"");
			}

            XMLCh *columnsXMLStr = xercesc::XMLString::transcode("columns");
            if(rootElement->hasAttribute(columnsXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(rootElement->getAttribute(columnsXMLStr));
                *cols = txtUtils.strto32bitUInt(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                throw SPDProcessingException("No \'columns\' attribute was provided.");
            }
            xercesc::XMLString::release(&columnsXMLStr);

            XMLCh *rowsXMLStr = xercesc::XMLString::transcode("rows");
            if(rootElement->hasAttribute(rowsXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(rootElement->getAttribute(rowsXMLStr));
                *rows = txtUtils.strto32bitUInt(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                throw SPDProcessingException("No \'rows\' attribute was provided.");
            }
            xercesc::XMLString::release(&rowsXMLStr);

            XMLCh *overlapXMLStr = xercesc::XMLString::transcode("overlap");
            if(rootElement->hasAttribute(overlapXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(rootElement->getAttribute(overlapXMLStr));
                *overlap = txtUtils.strtodouble(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                throw SPDProcessingException("No \'overlap\' attribute was provided.");
            }
            xercesc::XMLString::release(&overlapXMLStr);

            XMLCh *tileXSizeXMLStr = xercesc::XMLString::transcode("xtilesize");
            if(rootElement->hasAttribute(tileXSizeXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(rootElement->getAttribute(tileXSizeXMLStr));
                *xSize = txtUtils.strtodouble(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                throw SPDProcessingException("No \'xtilesize\' attribute was provided.");
            }
            xercesc::XMLString::release(&tileXSizeXMLStr);

            XMLCh *tileYSizeXMLStr = xercesc::XMLString::transcode("ytilesize");
            if(rootElement->hasAttribute(tileYSizeXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(rootElement->getAttribute(tileYSizeXMLStr));
                *ySize = txtUtils.strtodouble(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                throw SPDProcessingException("No \'ytilesize\' attribute was provided.");
            }
            xercesc::XMLString::release(&tileYSizeXMLStr);

            XMLCh *xMinXMLStr = xercesc::XMLString::transcode("xmin");
            if(rootElement->hasAttribute(xMinXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(rootElement->getAttribute(xMinXMLStr));
                *xMin = txtUtils.strtodouble(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                throw SPDProcessingException("No \'xmin\' attribute was provided.");
            }
            xercesc::XMLString::release(&xMinXMLStr);

            XMLCh *xMaxXMLStr = xercesc::XMLString::transcode("xmax");
            if(rootElement->hasAttribute(xMaxXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(rootElement->getAttribute(xMaxXMLStr));
                *xMax = txtUtils.strtodouble(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                throw SPDProcessingException("No \'xmax\' attribute was provided.");
            }
            xercesc::XMLString::release(&xMaxXMLStr);


            XMLCh *yMinXMLStr = xercesc::XMLString::transcode("ymin");
            if(rootElement->hasAttribute(yMinXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(rootElement->getAttribute(yMinXMLStr));
                *yMin = txtUtils.strtodouble(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                throw SPDProcessingException("No \'ymin\' attribute was provided.");
            }
            xercesc::XMLString::release(&yMinXMLStr);

            XMLCh *yMaxXMLStr = xercesc::XMLString::transcode("ymax");
            if(rootElement->hasAttribute(yMaxXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(rootElement->getAttribute(yMaxXMLStr));
                *yMax = txtUtils.strtodouble(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                throw SPDProcessingException("No \'ymax\' attribute was provided.");
            }
            xercesc::XMLString::release(&yMaxXMLStr);

            xercesc::DOMNodeList *tilesList = rootElement->getElementsByTagName(tileElementName);
            boost::uint_fast32_t numTiles = tilesList->getLength();

            tiles = new std::vector<SPDTile*>();
            tiles->reserve(numTiles);

            SPDTile *tile = NULL;
            xercesc::DOMElement *tileElement = NULL;

            for(boost::uint_fast32_t i = 0; i < numTiles; ++i)
            {
                tileElement = static_cast<xercesc::DOMElement*>(tilesList->item(i));
                tile = new SPDTile();


                XMLCh *colXMLStr = xercesc::XMLString::transcode("col");
                if(tileElement->hasAttribute(colXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(tileElement->getAttribute(colXMLStr));
                    tile->col = txtUtils.strto32bitUInt(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("No \'col\' attribute was provided.");
                }
                xercesc::XMLString::release(&colXMLStr);

                XMLCh *rowXMLStr = xercesc::XMLString::transcode("row");
                if(tileElement->hasAttribute(rowXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(tileElement->getAttribute(rowXMLStr));
                    tile->row = txtUtils.strto32bitUInt(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("No \'row\' attribute was provided.");
                }
                xercesc::XMLString::release(&rowXMLStr);

                XMLCh *xMinTileXMLStr = xercesc::XMLString::transcode("xmin");
                if(tileElement->hasAttribute(xMinTileXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(tileElement->getAttribute(xMinTileXMLStr));
                    tile->xMin = txtUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("No \'xmin\' attribute was provided.");
                }
                xercesc::XMLString::release(&xMinTileXMLStr);

                XMLCh *xMaxTileXMLStr = xercesc::XMLString::transcode("xmax");
                if(tileElement->hasAttribute(xMaxTileXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(tileElement->getAttribute(xMaxTileXMLStr));
                    tile->xMax = txtUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("No \'xmax\' attribute was provided.");
                }
                xercesc::XMLString::release(&xMaxTileXMLStr);


                XMLCh *yMinTileXMLStr = xercesc::XMLString::transcode("ymin");
                if(tileElement->hasAttribute(yMinTileXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(tileElement->getAttribute(yMinTileXMLStr));
                    tile->yMin = txtUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("No \'ymin\' attribute was provided.");
                }
                xercesc::XMLString::release(&yMinTileXMLStr);

                XMLCh *yMaxTileXMLStr = xercesc::XMLString::transcode("ymax");
                if(tileElement->hasAttribute(yMaxTileXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(tileElement->getAttribute(yMaxTileXMLStr));
                    tile->yMax = txtUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("No \'ymax\' attribute was provided.");
                }
                xercesc::XMLString::release(&yMaxTileXMLStr);

                XMLCh *xMinCoreTileXMLStr = xercesc::XMLString::transcode("corexmin");
                if(tileElement->hasAttribute(xMinCoreTileXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(tileElement->getAttribute(xMinCoreTileXMLStr));
                    tile->xMinCore = txtUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("No \'corexmin\' attribute was provided.");
                }
                xercesc::XMLString::release(&xMinCoreTileXMLStr);

                XMLCh *xMaxCoreTileXMLStr = xercesc::XMLString::transcode("corexmax");
                if(tileElement->hasAttribute(xMaxCoreTileXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(tileElement->getAttribute(xMaxCoreTileXMLStr));
                    tile->xMaxCore = txtUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("No \'corexmax\' attribute was provided.");
                }
                xercesc::XMLString::release(&xMaxCoreTileXMLStr);


                XMLCh *yMinCoreTileXMLStr = xercesc::XMLString::transcode("coreymin");
                if(tileElement->hasAttribute(yMinCoreTileXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(tileElement->getAttribute(yMinCoreTileXMLStr));
                    tile->yMinCore = txtUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("No \'coreymin\' attribute was provided.");
                }
                xercesc::XMLString::release(&yMinCoreTileXMLStr);

                XMLCh *yMaxCoreTileXMLStr = xercesc::XMLString::transcode("coreymax");
                if(tileElement->hasAttribute(yMaxCoreTileXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(tileElement->getAttribute(yMaxCoreTileXMLStr));
                    tile->yMaxCore = txtUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("No \'coreymax\' attribute was provided.");
                }
                xercesc::XMLString::release(&yMaxCoreTileXMLStr);

                XMLCh *fileTileXMLStr = xercesc::XMLString::transcode("file");
                if(tileElement->hasAttribute(fileTileXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(tileElement->getAttribute(fileTileXMLStr));
                    tile->outFileName = std::string(charValue);
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("No \'file\' attribute was provided.");
                }
                xercesc::XMLString::release(&fileTileXMLStr);


                tiles->push_back(tile);
            }

            xercesc::XMLString::release(&domImplTypeLS);
            xercesc::XMLString::release(&tilesElementName);
            xercesc::XMLString::release(&tileElementName);

            xercesc::XMLPlatformUtils::Terminate();
        }
        catch (const xercesc::OutOfMemoryException &e)
        {
            throw SPDProcessingException(xercesc::XMLString::transcode(e.getMessage()));
        }
        catch (const xercesc::DOMException &e)
        {
            throw SPDProcessingException(xercesc::XMLString::transcode(e.getMessage()));
        }
        catch(xercesc::XMLException &e)
        {
            throw SPDProcessingException(xercesc::XMLString::transcode(e.getMessage()));
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch(std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }

        return tiles;
    }

    void SPDTilesUtils::findNumRowsColsFromXML(std::string inputFile, boost::uint_fast32_t *rows, boost::uint_fast32_t *cols)throw(SPDProcessingException)
    {
        SPDTextFileUtilities txtUtils;

        try
        {
            xercesc::XMLPlatformUtils::Initialize();

            XMLCh *domImplTypeLS = xercesc::XMLString::transcode("LS");
            XMLCh *tilesElementName = xercesc::XMLString::transcode("tiles");

			xercesc::DOMImplementation *domImpl = xercesc::DOMImplementationRegistry::getDOMImplementation(domImplTypeLS);
			if(domImpl == NULL)
			{
				throw SPDProcessingException("DOMImplementation is NULL");
			}
			
			// Create Parser
			xercesc::DOMLSParser* domParser = ((xercesc::DOMImplementationLS*)domImpl)->createLSParser(xercesc::DOMImplementationLS::MODE_SYNCHRONOUS, 0);
			xercesc::ErrorHandler *errHandler = (xercesc::ErrorHandler*) new xercesc::HandlerBase();
			domParser->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMErrorHandler, errHandler);
			
			// Open Document
			xercesc::DOMDocument *doc = domParser->parseURI(inputFile.c_str());
			
			// Get the Root element
			xercesc::DOMElement *rootElement = doc->getDocumentElement();
            //std::cout << "Root Element: " << xercesc::XMLString::transcode(rootElement->getTagName()) << std::endl;
			if(!xercesc::XMLString::equals(rootElement->getTagName(), tilesElementName))
			{
				throw SPDProcessingException("Incorrect root element; Root element should be \"tiles\"");
			}

            XMLCh *columnsXMLStr = xercesc::XMLString::transcode("columns");
            if(rootElement->hasAttribute(columnsXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(rootElement->getAttribute(columnsXMLStr));
                *cols = txtUtils.strto32bitUInt(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                throw SPDProcessingException("No \'columns\' attribute was provided.");
            }
            xercesc::XMLString::release(&columnsXMLStr);

            XMLCh *rowsXMLStr = xercesc::XMLString::transcode("rows");
            if(rootElement->hasAttribute(rowsXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(rootElement->getAttribute(rowsXMLStr));
                *rows = txtUtils.strto32bitUInt(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                throw SPDProcessingException("No \'rows\' attribute was provided.");
            }
            xercesc::XMLString::release(&rowsXMLStr);

            xercesc::XMLString::release(&domImplTypeLS);
            xercesc::XMLString::release(&tilesElementName);

            xercesc::XMLPlatformUtils::Terminate();
        }
        catch (const xercesc::OutOfMemoryException &e)
        {
            throw SPDProcessingException(xercesc::XMLString::transcode(e.getMessage()));
        }
        catch (const xercesc::DOMException &e)
        {
            throw SPDProcessingException(xercesc::XMLString::transcode(e.getMessage()));
        }
        catch(xercesc::XMLException &e)
        {
            throw SPDProcessingException(xercesc::XMLString::transcode(e.getMessage()));
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch(std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }

    }

    void SPDTilesUtils::deleteTiles(std::vector<SPDTile*> *tiles)
    {
        for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
        {
            delete *iterTiles;
        }
        delete tiles;
    }

    void SPDTilesUtils::printTiles2Console(std::vector<SPDTile*> *tiles)
    {
        std::cout.precision(12);
        for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
        {
            std::cout << "[" << (*iterTiles)->col << " " << (*iterTiles)->row << "] ALL: [" << (*iterTiles)->xMin << "," << (*iterTiles)->xMax << "][" << (*iterTiles)->yMin << "," << (*iterTiles)->yMax << "] CORE: [" << (*iterTiles)->xMinCore << "," << (*iterTiles)->xMaxCore << "][" << (*iterTiles)->yMinCore << "," << (*iterTiles)->yMaxCore << "]" << std::endl;
        }
    }

    void SPDTilesUtils::createTileSPDFiles(std::vector<SPDTile*> *tiles, SPDFile *templateSPDFile, std::string outputBase, double xSize, double ySize, double overlap, double xMin, double xMax, double yMin, double yMax, boost::uint_fast32_t rows, boost::uint_fast32_t cols, bool checkTemplateSPDExtent) throw(SPDProcessingException)
    {
        SPDTextFileUtilities txtUtils;
        SPDMathsUtils mathUtils;
        try
        {
            for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
            {
                if((!checkTemplateSPDExtent) | mathUtils.rectangleIntersection((*iterTiles)->xMin, (*iterTiles)->xMax, (*iterTiles)->yMin, (*iterTiles)->yMax, templateSPDFile->getXMin(), templateSPDFile->getXMax(), templateSPDFile->getYMin(), templateSPDFile->getYMax()))
                {
                    //std::cout << "[" << (*iterTiles)->col << " " << (*iterTiles)->row << "] ALL: [" << (*iterTiles)->xMin << "," << (*iterTiles)->xMax << "][" << (*iterTiles)->yMin << "," << (*iterTiles)->yMax << "] CORE: [" << (*iterTiles)->xMinCore << "," << (*iterTiles)->xMaxCore << "][" << (*iterTiles)->yMinCore << "," << (*iterTiles)->yMaxCore << "]" << std::endl;
                    if((*iterTiles)->outFileName == "")
                    {
                        (*iterTiles)->outFileName = outputBase + std::string("_row") + txtUtils.uInt32bittostring((*iterTiles)->row) + std::string("col") + txtUtils.uInt32bittostring((*iterTiles)->col) + std::string(".spd");
                        std::cout << "Creating File: " << (*iterTiles)->outFileName << std::endl;
                        (*iterTiles)->spdFile = new SPDFile((*iterTiles)->outFileName);
                        (*iterTiles)->spdFile->copyAttributesFromTemplate(templateSPDFile);
                        (*iterTiles)->spdFile->setBoundingBox((*iterTiles)->xMin, (*iterTiles)->xMax, (*iterTiles)->yMin, (*iterTiles)->yMax);
                        (*iterTiles)->writer = new SPDNoIdxFileWriter();
                        (*iterTiles)->writer->open((*iterTiles)->spdFile, (*iterTiles)->outFileName);
                        (*iterTiles)->writer->finaliseClose();
                        delete (*iterTiles)->writer;
                        (*iterTiles)->writerOpen = false;
                    }
                    else
                    {
                        std::cout << "Opening File: " << (*iterTiles)->outFileName << std::endl;
                        (*iterTiles)->spdFile = new SPDFile((*iterTiles)->outFileName);

                        (*iterTiles)->writer = new SPDNoIdxFileWriter();
                        (*iterTiles)->writer->reopen((*iterTiles)->spdFile, (*iterTiles)->outFileName);
                        /*
                        if((*iterTiles)->spdFile->getXMin() != (*iterTiles)->xMin)
                        {
                            std::cout << "Tile xMin = " << (*iterTiles)->xMin << std::endl;
                            std::cout << "SPD File xMin = " << (*iterTiles)->spdFile->getXMin() << std::endl;
                            throw SPDProcessingException("Warning: x Min in SPDFile and tile do not match.");
                        }
                        else if((*iterTiles)->spdFile->getXMax() != (*iterTiles)->xMax)
                        {
                            std::cout << "Tile xMax = " << (*iterTiles)->xMax << std::endl;
                            std::cout << "SPD File xMax = " << (*iterTiles)->spdFile->getXMax() << std::endl;
                            throw SPDProcessingException("Warning: x Max in SPDFile and tile do not match.");
                        }
                        else if((*iterTiles)->spdFile->getYMin() != (*iterTiles)->yMin)
                        {
                            std::cout << "Tile yMin = " << (*iterTiles)->yMin << std::endl;
                            std::cout << "SPD File yMin = " << (*iterTiles)->spdFile->getYMin() << std::endl;
                            throw SPDProcessingException("Warning: y Min in SPDFile and tile do not match.");
                        }
                        else if((*iterTiles)->spdFile->getYMax() != (*iterTiles)->yMax)
                        {
                            std::cout << "Tile yMax = " << (*iterTiles)->yMax << std::endl;
                            std::cout << "SPD File yMax = " << (*iterTiles)->spdFile->getYMax() << std::endl;
                            throw SPDProcessingException("Warning: y Max in SPDFile and tile do not match.");
                        }
                        */
                        (*iterTiles)->writer->finaliseClose();
                        delete (*iterTiles)->writer;
                        (*iterTiles)->writerOpen = false;
                    }
                }
            }
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch(std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }

    }

    void SPDTilesUtils::createTileSPDFilesDirStruct(std::vector<SPDTile*> *tiles, SPDFile *templateSPDFile, std::string outputBase, bool usePrefix, std::string prefix, double xSize, double ySize, double overlap, double xMin, double xMax, double yMin, double yMax, boost::uint_fast32_t rows, boost::uint_fast32_t cols, bool checkTemplateSPDExtent) throw(SPDProcessingException)
    {
        SPDTextFileUtilities txtUtils;
        SPDMathsUtils mathUtils;
        try
        {
            for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
            {
                if((!checkTemplateSPDExtent) | mathUtils.rectangleIntersection((*iterTiles)->xMin, (*iterTiles)->xMax, (*iterTiles)->yMin, (*iterTiles)->yMax, templateSPDFile->getXMin(), templateSPDFile->getXMax(), templateSPDFile->getYMin(), templateSPDFile->getYMax()))
                {
                    //std::cout << "[" << (*iterTiles)->col << " " << (*iterTiles)->row << "] ALL: [" << (*iterTiles)->xMin << "," << (*iterTiles)->xMax << "][" << (*iterTiles)->yMin << "," << (*iterTiles)->yMax << "] CORE: [" << (*iterTiles)->xMinCore << "," << (*iterTiles)->xMaxCore << "][" << (*iterTiles)->yMinCore << "," << (*iterTiles)->yMaxCore << "]" << std::endl;
                    if((*iterTiles)->outFileName == "")
                    {
                        boost::filesystem::path outDir = boost::filesystem::path(outputBase);
                        outDir /= boost::filesystem::path("Rows" + txtUtils.int32bittostring((*iterTiles)->row));
                        outDir /= boost::filesystem::path("Cols" + txtUtils.int32bittostring((*iterTiles)->col));
                        outDir /= boost::filesystem::path("Tiles");

                        if(boost::filesystem::exists(outDir.string()) && boost::filesystem::is_directory(outDir.string()))
                        {
                            if(usePrefix)
                            {
                                outDir /= boost::filesystem::path(prefix + std::string("_row") + txtUtils.uInt32bittostring((*iterTiles)->row) + std::string("col") + txtUtils.uInt32bittostring((*iterTiles)->col) + std::string(".spd"));
                            }
                            else
                            {
                                outDir /= boost::filesystem::path(std::string("_row") + txtUtils.uInt32bittostring((*iterTiles)->row) + std::string("col") + txtUtils.uInt32bittostring((*iterTiles)->col) + std::string(".spd"));

                            }
                            (*iterTiles)->outFileName = outDir.string();
                            std::cout << "Creating File: " << (*iterTiles)->outFileName << std::endl;
                            (*iterTiles)->spdFile = new SPDFile((*iterTiles)->outFileName);
                            (*iterTiles)->spdFile->copyAttributesFromTemplate(templateSPDFile);
                            (*iterTiles)->spdFile->setBoundingBox((*iterTiles)->xMin, (*iterTiles)->xMax, (*iterTiles)->yMin, (*iterTiles)->yMax);
                            (*iterTiles)->writer = new SPDNoIdxFileWriter();
                            (*iterTiles)->writer->open((*iterTiles)->spdFile, (*iterTiles)->outFileName);
                            (*iterTiles)->writer->finaliseClose();
                            delete (*iterTiles)->writer;
                            (*iterTiles)->writerOpen = false;
                        }
                        else
                        {
                            std::string outMess = std::string("Directory \'") + std::string(outDir.string()) + std::string("\' was not present. Did you build the directory structure?");
                            throw SPDProcessingException(outMess);
                        }
                    }
                    else
                    {
                        std::cout << "Opening File: " << (*iterTiles)->outFileName << std::endl;
                        (*iterTiles)->spdFile = new SPDFile((*iterTiles)->outFileName);

                        (*iterTiles)->writer = new SPDNoIdxFileWriter();
                        (*iterTiles)->writer->reopen((*iterTiles)->spdFile, (*iterTiles)->outFileName);
                        /*
                         if((*iterTiles)->spdFile->getXMin() != (*iterTiles)->xMin)
                         {
                         std::cout << "Tile xMin = " << (*iterTiles)->xMin << std::endl;
                         std::cout << "SPD File xMin = " << (*iterTiles)->spdFile->getXMin() << std::endl;
                         throw SPDProcessingException("Warning: x Min in SPDFile and tile do not match.");
                         }
                         else if((*iterTiles)->spdFile->getXMax() != (*iterTiles)->xMax)
                         {
                         std::cout << "Tile xMax = " << (*iterTiles)->xMax << std::endl;
                         std::cout << "SPD File xMax = " << (*iterTiles)->spdFile->getXMax() << std::endl;
                         throw SPDProcessingException("Warning: x Max in SPDFile and tile do not match.");
                         }
                         else if((*iterTiles)->spdFile->getYMin() != (*iterTiles)->yMin)
                         {
                         std::cout << "Tile yMin = " << (*iterTiles)->yMin << std::endl;
                         std::cout << "SPD File yMin = " << (*iterTiles)->spdFile->getYMin() << std::endl;
                         throw SPDProcessingException("Warning: y Min in SPDFile and tile do not match.");
                         }
                         else if((*iterTiles)->spdFile->getYMax() != (*iterTiles)->yMax)
                         {
                         std::cout << "Tile yMax = " << (*iterTiles)->yMax << std::endl;
                         std::cout << "SPD File yMax = " << (*iterTiles)->spdFile->getYMax() << std::endl;
                         throw SPDProcessingException("Warning: y Max in SPDFile and tile do not match.");
                         }
                         */
                        (*iterTiles)->writer->finaliseClose();
                        delete (*iterTiles)->writer;
                        (*iterTiles)->writerOpen = false;
                    }
                }

            }
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch(std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }

    }


    void SPDTilesUtils::populateTilesWithData(std::vector<SPDTile*> *tiles, std::string inputSPDFile) throw(SPDProcessingException)
    {
        try
        {
            SPDFileReader reader;
            SPDMathsUtils mathUtils;
            SPDFile *inSPDFile = NULL;
            std::vector<SPDTile*> *openTiles = new std::vector<SPDTile*>();
            SPDWrite2OverlapTiles *write2Tiles = NULL;

            if(inputSPDFile != "")
            {
                std::cout << "Processing: \'" << inputSPDFile << "\'" << std::endl;

                // STEP 1: Read file header - Need file extent.
                inSPDFile = new SPDFile(inputSPDFile);
                reader.readHeaderInfo(inputSPDFile, inSPDFile);

                // STEP 2: Find the tiles intersecting with the file extent
                for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
                {
                    // If intersect open the writer (reopen) and add to list of tiles for processing...
                    if(mathUtils.rectangleIntersection(inSPDFile->getXMin(), inSPDFile->getXMax(), inSPDFile->getYMin(), inSPDFile->getYMax(), (*iterTiles)->xMin, (*iterTiles)->xMax, (*iterTiles)->yMin, (*iterTiles)->yMax))
                    {
                        //std::cout << "Opening tile [" << (*iterTiles)->col << ", " << (*iterTiles)->row << "]\n";
                        //std::cout << "openTiles->size() = " << openTiles->size() << std::endl;
                        (*iterTiles)->writer = new SPDNoIdxFileWriter();
                        //(*iterTiles)->writer->reopen((*iterTiles)->spdFile, (*iterTiles)->outFileName);
                        (*iterTiles)->writerOpen = false;
                        openTiles->push_back((*iterTiles));
                    }
                }

                // STEP 3: Copy data into the tiles.
                std::cout << "There are " << openTiles->size() << " tiles in active use.\n";
                if(openTiles->size() == 0)
                {
                    throw SPDProcessingException("No intersecting tiles were found.");
                }

                write2Tiles = new SPDWrite2OverlapTiles(openTiles);
                reader.readAndProcessAllData(inputSPDFile, inSPDFile, write2Tiles);

                // STEP 4: Close open tiles
                write2Tiles->completeFileAndClose();
                delete write2Tiles;
                delete inSPDFile;
                openTiles->clear();
            }

            delete openTiles;
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch (SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
    }

    void SPDTilesUtils::populateTilesWithData(std::vector<SPDTile*> *tiles, std::vector<std::string> inputFiles) throw(SPDProcessingException)
    {
        try
        {
            SPDFileReader reader;
            SPDMathsUtils mathUtils;
            SPDFile *inSPDFile = NULL;
            std::vector<SPDTile*> *openTiles = new std::vector<SPDTile*>();
            SPDWrite2OverlapTiles *write2Tiles = NULL;
            for(std::vector<std::string>::iterator iterFiles = inputFiles.begin(); iterFiles != inputFiles.end(); ++iterFiles)
            {
                if((*iterFiles) != "")
                {
                    std::cout << "Processing: \'" << *iterFiles << "\'" << std::endl;

                    // STEP 1: Read file header - Need file extent.
                    inSPDFile = new SPDFile(*iterFiles);
                    reader.readHeaderInfo(*iterFiles, inSPDFile);

                    // STEP 2: Find the tiles intersecting with the file extent
                    for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
                    {
                        // If intersect open the writer (reopen) and add to list of tiles for processing...
                        if(mathUtils.rectangleIntersection(inSPDFile->getXMin(), inSPDFile->getXMax(), inSPDFile->getYMin(), inSPDFile->getYMax(), (*iterTiles)->xMin, (*iterTiles)->xMax, (*iterTiles)->yMin, (*iterTiles)->yMax))
                        {
                            //std::cout << "Opening tile [" << (*iterTiles)->col << ", " << (*iterTiles)->row << "]\n";
                            (*iterTiles)->writer = new SPDNoIdxFileWriter();
                            (*iterTiles)->writer->reopen((*iterTiles)->spdFile, (*iterTiles)->outFileName);
                            (*iterTiles)->writerOpen = true;
                            openTiles->push_back((*iterTiles));
                        }
                    }

                    // STEP 3: Copy data into the tiles.
                    std::cout << "There are " << openTiles->size() << " open\n";
                    if(openTiles->size() == 0)
                    {
                        throw SPDProcessingException("No intersecting tiles were found.");
                    }

                    write2Tiles = new SPDWrite2OverlapTiles(openTiles);
                    reader.readAndProcessAllData((*iterFiles), inSPDFile, write2Tiles);

                    // STEP 4: Close open tiles
                    write2Tiles->completeFileAndClose();
                    delete write2Tiles;
                    delete inSPDFile;
                    openTiles->clear();
                }
            }
            delete openTiles;
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch (SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
    }

    void SPDTilesUtils::populateTileWithData(SPDTile *tile, std::vector<std::string> inputFiles) throw(SPDProcessingException)
    {
        try
        {
            SPDFileReader reader;
            SPDMathsUtils mathUtils;
            SPDFile *inSPDFile = NULL;
            std::vector<SPDTile*> *openTiles = new std::vector<SPDTile*>();
            SPDWrite2OverlapTiles *write2Tiles = NULL;

            tile->writer = new SPDNoIdxFileWriter();
            tile->writer->reopen(tile->spdFile, tile->outFileName);
            tile->writerOpen = true;
            openTiles->push_back(tile);
            write2Tiles = new SPDWrite2OverlapTiles(openTiles);

            for(std::vector<std::string>::iterator iterFiles = inputFiles.begin(); iterFiles != inputFiles.end(); ++iterFiles)
            {
                if((*iterFiles) != "")
                {
                    std::cout << "Processing: \'" << *iterFiles << "\'" << std::endl;

                    // STEP 1: Read file header - Need file extent.
                    inSPDFile = new SPDFile(*iterFiles);
                    reader.readHeaderInfo(*iterFiles, inSPDFile);

                    // If intersect copy data into tile.
                    if(mathUtils.rectangleIntersection(inSPDFile->getXMin(), inSPDFile->getXMax(), inSPDFile->getYMin(), inSPDFile->getYMax(), tile->xMin, tile->xMax, tile->yMin, tile->yMax))
                    {
                        reader.readAndProcessAllData((*iterFiles), inSPDFile, write2Tiles);
                    }

                    // Close tile.
                    delete inSPDFile;
                }
            }
            write2Tiles->completeFileAndClose();
            delete write2Tiles;
            openTiles->clear();
            delete openTiles;
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch (SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
    }

    void SPDTilesUtils::deleteTilesWithNoPulses(std::vector<SPDTile*> *tiles) throw(SPDProcessingException)
    {
        SPDTextFileUtilities txtUtils;
        try
        {
            for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); )
            {
                if((*iterTiles)->outFileName != "")
                {
                    if((*iterTiles)->spdFile->getNumberOfPulses() == 0)
                    {
                        if((*iterTiles)->spdFile != NULL)
                        {
                            delete (*iterTiles)->spdFile;
                        }
                        /*if((*iterTiles)->writer != NULL)
                        {
                            (*iterTiles)->writer->finaliseClose();
                            delete (*iterTiles)->writer;
                        }*/
                        boost::filesystem::path rFilePath((*iterTiles)->outFileName);
                        boost::filesystem::remove(rFilePath);
                        delete *iterTiles;
                        tiles->erase(iterTiles);
                    }
                    else
                    {
                        ++iterTiles;
                    }
                }
                else
                {
                    ++iterTiles;
                }
            }
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch(std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
    }


    void SPDTilesUtils::deleteTileIfNoPulses(std::vector<SPDTile*> *tiles, boost::uint_fast32_t row, boost::uint_fast32_t col) throw(SPDProcessingException)
    {
        SPDTextFileUtilities txtUtils;
        try
        {
            for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); )
            {
                if(((*iterTiles)->row == row) & ((*iterTiles)->col == col))
                {
                    if(((*iterTiles)->spdFile != NULL) && ((*iterTiles)->spdFile->getNumberOfPulses() == 0))
                    {
                        boost::filesystem::path rFilePath((*iterTiles)->outFileName);
                        boost::filesystem::remove(rFilePath);

                        delete (*iterTiles)->spdFile;

                        delete *iterTiles;
                        tiles->erase(iterTiles);
                    }
                    else
                    {
                        ++iterTiles;
                    }
                }
                else
                {
                    ++iterTiles;
                }
            }
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch(SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch(std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
    }

    GDALDataset* SPDTilesUtils::createNewImageFile(std::string imageFile, std::string format, GDALDataType dataType, std::string wktFile, double xRes, double yRes, double tlX, double tlY, boost::uint_fast32_t xImgSize, boost::uint_fast32_t yImgSize, boost::uint_fast32_t numBands, double backgroundVal) throw(SPDProcessingException)
    {
        // Process dataset in memory
        GDALDriver *gdalDriver = GetGDALDriverManager()->GetDriverByName(format.c_str());
        if(gdalDriver == NULL)
        {
            std::string message = std::string("Driver for ") + format + std::string(" does not exist\n");
            throw SPDProcessingException(message.c_str());
        }
        GDALDataset *dataset = gdalDriver->Create(imageFile.c_str(), xImgSize, yImgSize, numBands, dataType, NULL);
        if(dataset == NULL)
        {
            std::string message = std::string("Could not create GDALDataset.");
            throw SPDProcessingException(message);
        }

        double *gdalTranslation = new double[6];
        gdalTranslation[0] = tlX;
        gdalTranslation[1] = xRes;
        gdalTranslation[2] = 0;
        gdalTranslation[3] = tlY;
        gdalTranslation[4] = 0;
        gdalTranslation[5] = yRes;

        dataset->SetGeoTransform(gdalTranslation);
        dataset->SetProjection(wktFile.c_str());
        delete[] gdalTranslation;

        int xBlockSize = 0;
        int yBlockSize = 0;

        dataset->GetRasterBand(1)->GetBlockSize (&xBlockSize, &yBlockSize);

        float **outData = new float*[numBands];
        GDALRasterBand **rasterBands = new GDALRasterBand*[numBands];
        for(int i = 0; i < numBands; i++)
        {
            outData[i] = (float *) CPLMalloc(sizeof(float)*xImgSize*yBlockSize);
            for(unsigned int j = 0; j < (xImgSize*yBlockSize); ++j)
            {
                outData[i][j] = backgroundVal;
            }
            rasterBands[i] = dataset->GetRasterBand(i+1);
        }

        int nYBlocks = yImgSize / yBlockSize;
        int remainRows = yImgSize - (nYBlocks * yBlockSize);
        int rowOffset = 0;

        // Loop images to process data
        for(int i = 0; i < nYBlocks; i++)
        {
            for(int n = 0; n < numBands; n++)
            {
                rowOffset = yBlockSize * i;
                rasterBands[n]->RasterIO(GF_Write, 0, rowOffset, xImgSize, yBlockSize, outData[n], xImgSize, yBlockSize, GDT_Float32, 0, 0);
            }
        }

        if(remainRows > 0)
        {
            for(int n = 0; n < numBands; n++)
            {
                rowOffset = (yBlockSize * nYBlocks);
                rasterBands[n]->RasterIO(GF_Write, 0, rowOffset, xImgSize, remainRows, outData[n], xImgSize, remainRows, GDT_Float32, 0, 0);
            }
        }

        for(int i = 0; i < numBands; i++)
        {
            delete[] outData[i];
        }
        delete[] outData;
        delete[] rasterBands;

        return dataset;
    }

    void SPDTilesUtils::addImageTiles(GDALDataset *image, std::vector<SPDTile*> *tiles, std::vector<std::string> inputImageFiles) throw(SPDProcessingException)
    {
        try
        {
            SPDMathsUtils mathUtils;
            SPDImageUtils imgUtils;
            GDALDataset *tileDataset = NULL;
            double *geoTrans = new double[6];
            double xMin = 0;
            double xMax = 0;
            double yMin = 0;
            double yMax = 0;

            unsigned int xTileSize = 0;
            unsigned int yTileSize = 0;
            double intersection = 0;
            bool first = false;
            double maxInsect;
            SPDTile *maxInsectTile = NULL;
            OGREnvelope *env = new OGREnvelope();
            for(std::vector<std::string>::iterator iterFiles = inputImageFiles.begin(); iterFiles != inputImageFiles.end(); ++iterFiles)
            {
                if((*iterFiles) != "")
                {

                    std::cout << "Processing \'" << (*iterFiles) << "\'\n";

                    /*
                     * OPEN IMAGE FILE TILE.
                     */

                    tileDataset = (GDALDataset *) GDALOpen((*iterFiles).c_str(), GA_ReadOnly);
                    if(tileDataset == NULL)
                    {
                        std::string message = std::string("Could not open image ") + (*iterFiles);
                        throw spdlib::SPDException(message.c_str());
                    }
                    tileDataset->GetGeoTransform(geoTrans);

                    xTileSize = tileDataset->GetRasterXSize();
                    yTileSize = tileDataset->GetRasterYSize();

                    xMin = geoTrans[0];
                    yMax = geoTrans[3];
                    xMax = xMin + (xTileSize * geoTrans[1]);
                    yMin = yMax + (yTileSize * geoTrans[5]);

                    /*
                     * FIND TILE ASSOCIATED WITH FILE.
                     */

                    //std::cout << "Image: [" << xMin << ", " << xMax << "][" << yMin << ", " << yMax << "]\n";
                    first = false;
                    for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
                    {
                        //std::cout << "\tTile: [" << (*iterTiles)->xMinCore << ", " << (*iterTiles)->xMaxCore << "][" << (*iterTiles)->yMinCore << ", " << (*iterTiles)->yMaxCore << "]\n";
                        intersection = mathUtils.calcRectangleIntersection((*iterTiles)->xMinCore, (*iterTiles)->xMaxCore, (*iterTiles)->yMinCore, (*iterTiles)->yMaxCore, xMin,xMax, yMin, yMax);
                        //std::cout << "\t\t" << intersection << std::endl;
                        if(first)
                        {
                            maxInsect = intersection;
                            maxInsectTile = *iterTiles;
                            first = false;
                        }
                        else if(intersection > maxInsect)
                        {
                            maxInsect = intersection;
                            maxInsectTile = *iterTiles;
                        }
                    }

                    //std::cout << "\t Intersect Tile: [" << maxInsectTile->xMinCore << ", " << maxInsectTile->xMaxCore << "][" << maxInsectTile->yMinCore << ", " << maxInsectTile->yMaxCore << "]\n";

                    //std::cout << "\t Intersection Tile: Row = " << maxInsectTile->row << " Column = " << maxInsectTile->col << std::endl;

                    /*
                     * INCLUDE THE TILE WITHIN THE WHOLE IMAGE.
                     */

                    env->MinX = maxInsectTile->xMinCore;
                    env->MaxX = maxInsectTile->xMaxCore;
                    env->MinY = maxInsectTile->yMinCore;
                    env->MaxY = maxInsectTile->yMaxCore;

                    imgUtils.copyInDatasetIntoOutDataset(tileDataset, image, env);

                    GDALClose(tileDataset);
                }
            }
            delete[] geoTrans;
            delete env;
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch (SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
    }

    void SPDTilesUtils::addImageTilesParseFileName(GDALDataset *image, std::vector<SPDTile*> *tiles, std::vector<std::string> inputImageFiles) throw(SPDProcessingException)
    {
        try
        {
            SPDMathsUtils mathUtils;
            SPDImageUtils imgUtils;
            GDALDataset *tileDataset = NULL;
            double *geoTrans = new double[6];
            double xMin = 0;
            double xMax = 0;
            double yMin = 0;
            double yMax = 0;

            boost::uint_fast32_t tileRow = 0;
            boost::uint_fast32_t tileCol = 0;

            unsigned int xTileSize = 0;
            unsigned int yTileSize = 0;
            double intersection = 0;
            double maxInsect;
            SPDTile *maxInsectTile = NULL;
            bool tileFound = false;
            OGREnvelope *env = new OGREnvelope();
            for(std::vector<std::string>::iterator iterFiles = inputImageFiles.begin(); iterFiles != inputImageFiles.end(); ++iterFiles)
            {
                if((*iterFiles) != "")
                {
                    std::cout << "Processing \'" << (*iterFiles) << "\'\n";

                    this->extractRowColFromFileName((*iterFiles), &tileRow, &tileCol);
                    //std::cout << "Row " << tileRow << " Col " << tileCol << std::endl;

                    /*
                     * OPEN IMAGE FILE TILE.
                     */

                    tileDataset = (GDALDataset *) GDALOpen((*iterFiles).c_str(), GA_ReadOnly);
                    if(tileDataset == NULL)
                    {
                        std::string message = std::string("Could not open image ") + (*iterFiles);
                        throw spdlib::SPDException(message.c_str());
                    }
                    tileDataset->GetGeoTransform(geoTrans);

                    xTileSize = tileDataset->GetRasterXSize();
                    yTileSize = tileDataset->GetRasterYSize();

                    //std::cout << "image size = [" << xTileSize << ", " << yTileSize << "]\n";

                    xMin = geoTrans[0];
                    yMax = geoTrans[3];
                    xMax = xMin + (xTileSize * geoTrans[1]);
                    yMin = yMax + (yTileSize * geoTrans[5]);
                    //std::cout << "Image: [" << xMin << ", " << xMax << "][" << yMin << ", " << yMax << "]\n";

                    /*
                     * FIND TILE ASSOCIATED WITH FILE.
                     */
                    tileFound = false;
                    for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
                    {
                        if(((*iterTiles)->row == tileRow) & ((*iterTiles)->col == tileCol))
                        {
                            intersection = mathUtils.calcRectangleIntersection((*iterTiles)->xMinCore, (*iterTiles)->xMaxCore, (*iterTiles)->yMinCore, (*iterTiles)->yMaxCore, xMin,xMax, yMin, yMax);

                            if(intersection > 0)
                            {
                                maxInsect = intersection;
                                maxInsectTile = *iterTiles;
                                tileFound = true;
                            }
                            else
                            {
                                //throw SPDProcessingException("Tile is incorrect as rows and cols does not correspond with intersection of the image and tile core - does the tiles XML file match the tiles?");
                                std::cout << "WARNING: Tile is being skipped [" << tileRow << "," << tileCol << "]\n";
                                std::cout << "\tTile is incorrect as rows and cols does not correspond with intersection of the image and tile core - does the tiles XML file match the tiles?\n";
                                tileFound = false;
                            }
                        }
                    }

                    //std::cout << "\t Intersect Tile: [" << maxInsectTile->xMinCore << ", " << maxInsectTile->xMaxCore << "][" << maxInsectTile->yMinCore << ", " << maxInsectTile->yMaxCore << "]\n";

                    //std::cout << "\t Intersection Tile: Row = " << maxInsectTile->row << " Column = " << maxInsectTile->col << std::endl;

                    /*
                     * INCLUDE THE TILE WITHIN THE WHOLE IMAGE.
                     */
                    if(tileFound)
                    {
                        env->MinX = maxInsectTile->xMinCore;
                        env->MaxX = maxInsectTile->xMaxCore;
                        env->MinY = maxInsectTile->yMinCore;
                        env->MaxY = maxInsectTile->yMaxCore;

                        imgUtils.copyInDatasetIntoOutDataset(tileDataset, image, env);
                    }

                    GDALClose(tileDataset);
                }
            }
            delete[] geoTrans;
            delete env;
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch (SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
    }

    void SPDTilesUtils::addTiles2ClumpImage(GDALDataset *image, std::vector<SPDTile*> *tiles) throw(SPDProcessingException)
    {
        try
        {
            SPDImageUtils imgUtils;
            double *geoTrans = new double[6];
            image->GetGeoTransform(geoTrans);

            double tlX = geoTrans[0];
            double tlY = geoTrans[3];

            double xRes = geoTrans[1];
            double yRes = geoTrans[5];
            double posYRes = yRes;
            if(yRes < 0)
            {
                posYRes = yRes * (-1);
            }

            delete[] geoTrans;

            GDALRasterBand *imgBand = image->GetRasterBand(1);

            //std::cout << "Image Size: X = " << image->GetRasterXSize() << " Y = " << image->GetRasterYSize() << std::endl;

            //GDALRasterAttributeTable *gdalATT = new GDALRasterAttributeTable();
            GDALRasterAttributeTable *gdalATT = new GDALDefaultRasterAttributeTable();
            gdalATT->SetRowCount(tiles->size()+1);

            int xBlock = 0;
            int yBlock = 0;
            boost::uint_fast32_t numBlocks = 0;
            boost::uint_fast32_t remainingLines = 0;
            imgBand->GetBlockSize(&xBlock, &yBlock);

            boost::uint_fast32_t colIdx = imgUtils.findColumnIndexOrCreate(gdalATT, "Col", GFT_Integer);
            boost::uint_fast32_t rowIdx = imgUtils.findColumnIndexOrCreate(gdalATT, "Row", GFT_Integer);

            boost::uint_fast32_t tilePxlX = 0;
            boost::uint_fast32_t tilePxlY = 0;

            boost::uint_fast32_t tileSizeX = 0;
            boost::uint_fast32_t tileSizeY = 0;

            boost::uint_fast32_t rowOffset = 0;

            double tileWidth = 0;
            double tileHeight = 0;

            double xDist = 0;
            double yDist = 0;

            unsigned int *imgData = NULL;

            boost::uint_fast32_t counter = 1;
            for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
            {
                //std::cout << "\nTile: Row = " << (*iterTiles)->row << " Column = " << (*iterTiles)->col << std::endl;
                gdalATT->SetValue(counter, rowIdx, ((int)(*iterTiles)->row));
                gdalATT->SetValue(counter, colIdx, ((int)(*iterTiles)->col));

                tileWidth = (*iterTiles)->xMaxCore - (*iterTiles)->xMinCore;
                tileHeight = (*iterTiles)->yMaxCore - (*iterTiles)->yMinCore;

                //std::cout << "tileWidth = " << tileWidth << std::endl;
                //std::cout << "tileHeight = " << tileHeight << std::endl;

                tileSizeX = floor((tileWidth/xRes)+0.5);
                tileSizeY = floor((tileHeight/posYRes)+0.5);

                //std::cout << "tileSizeX = " << tileSizeX << std::endl;
                //std::cout << "tileSizeY = " << tileSizeY << std::endl;

                xDist = (*iterTiles)->xMinCore - tlX;
                yDist = tlY - (*iterTiles)->yMaxCore;

                //std::cout << "xDist = " << xDist << std::endl;
                //std::cout << "yDist = " << yDist << std::endl;

                if(yDist < 0)
                {
                    yDist *= (-1);
                    tilePxlY = floor((yDist/posYRes)+0.5);
                    tileSizeY -= tilePxlY;

                    yDist = 0;
                }

                if(xDist < 0)
                {
                    xDist *= (-1);
                    tilePxlX = floor((xDist/xRes)+0.5);
                    tileSizeX -= tilePxlX;

                    xDist = 0;
                }


                tilePxlX = floor((xDist/xRes)+0.5);
                tilePxlY = floor((yDist/posYRes)+0.5);

                //std::cout << "tilePxlX = " << tilePxlX << std::endl;
                //std::cout << "tilePxlY = " << tilePxlY << std::endl;

                imgData = (unsigned int *) CPLMalloc(sizeof(unsigned int)*tileSizeX*yBlock);

                for(unsigned int i = 0; i < (tileSizeX*yBlock); ++i)
                {
                    imgData[i] = counter;
                }

                numBlocks = floor(tileSizeY/yBlock);
                remainingLines = tileSizeY - (numBlocks*yBlock);

                //std::cout << "numBlocks = " << numBlocks << std::endl;
                //std::cout << "remainingLines = " << remainingLines << std::endl;

                for(int i = 0; i < numBlocks; i++)
                {
                    rowOffset = tilePxlY + (yBlock * i);
                    //std::cout << "Writing to image band (In block)\n";
                    imgBand->RasterIO(GF_Write, tilePxlX, rowOffset, tileSizeX, yBlock, imgData, tileSizeX, yBlock, GDT_UInt32, 0, 0);
                }

                if(remainingLines > 0)
                {
                    rowOffset = tilePxlY + (yBlock * numBlocks);
                    //std::cout << "Writing to image band (In Remaining)\n";
                    imgBand->RasterIO(GF_Write, tilePxlX, rowOffset, tileSizeX, remainingLines, imgData, tileSizeX, remainingLines, GDT_UInt32, 0, 0);
                }

                delete imgData;
                ++counter;
            }

            imgBand->SetDefaultRAT(gdalATT);
            delete gdalATT;
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch (SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
    }

    void SPDTilesUtils::extractRowColFromFileName(std::string filePathStr, boost::uint_fast32_t *row, boost::uint_fast32_t *col) throw(SPDProcessingException)
    {
        try
        {
            SPDTextFileUtilities txtUtils;
            boost::filesystem::path rFilePath(filePathStr);
            std::string filename = rFilePath.filename().generic_string();

            std::string rowStr = "";
            bool foundRow = false;
            std::string colStr = "";
            bool foundCol = false;

            int lineLength = filename.length();
            for(int i = 0; i < lineLength; i++)
            {
                if(filename.at(i) == 'r')
                {
                    if(i+7 < lineLength)
                    {
                        if((filename.at(i+1) == 'o') & (filename.at(i+2) == 'w'))
                        {
                            foundRow = true;
                            for(int j = i+3; j < lineLength; ++j)
                            {
                                if(txtUtils.isNumber(filename.at(j)))
                                {
                                    rowStr += filename.at(j);
                                }
                                else
                                {
                                    break;
                                }
                            }
                        }
                    }
                }

                if(filename.at(i) == 'c')
                {
                    if(i+3 < lineLength)
                    {
                        if((filename.at(i+1) == 'o') & (filename.at(i+2) == 'l'))
                        {
                            foundCol = true;
                            for(int j = i+3; j < lineLength; ++j)
                            {
                                if(txtUtils.isNumber(filename.at(j)))
                                {
                                    colStr += filename.at(j);
                                }
                                else
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            //std::cout << "ROW = \'" << rowStr << "\'" << std::endl;
            //std::cout << "COL = \'" << colStr << "\'" << std::endl;

            if(!foundRow | (rowStr == ""))
            {
                SPDProcessingException("Row could not be found from the file name.");
            }

            *row = txtUtils.strto32bitUInt(rowStr);

            if(!foundCol | (colStr == ""))
            {
                SPDProcessingException("Col could not be found from the file name.");
            }

            *col = txtUtils.strto32bitUInt(colStr);

        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        catch(std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
    }

    void SPDTilesUtils::extractTileCore(std::string inputSPDFile, std::string outputSPDFile, boost::uint_fast32_t row, boost::uint_fast32_t col, std::vector<SPDTile*> *tiles) throw(SPDProcessingException)
    {
        try
        {
            SPDTile *tile = NULL;
            bool foundTile = false;
            for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
            {
                if(((*iterTiles)->row == row) & ((*iterTiles)->col == col))
                {
                    foundTile = true;
                    tile = *iterTiles;
                    break;
                }
            }

            SPDFile *inSPDFile = new SPDFile(inputSPDFile);
            SPDFileReader reader;
            reader.readHeaderInfo(inputSPDFile, inSPDFile);

            if(tile->spdFile != NULL)
            {
                delete tile->spdFile;
            }
            tile->writer = new SPDNoIdxFileWriter();
            tile->spdFile = new SPDFile(outputSPDFile);
            tile->spdFile->copyAttributesFrom(inSPDFile);
            tile->spdFile->setNumberBinsX(0);
            tile->spdFile->setNumberBinsY(0);
            tile->spdFile->setNumberOfPoints(0);
            tile->spdFile->setNumberOfPulses(0);
            tile->spdFile->setBoundingVolume(tile->xMinCore, tile->xMaxCore, tile->yMinCore, tile->yMaxCore, 0.0, 0.0);

            tile->outFileName = outputSPDFile;
            tile->writer->open(tile->spdFile, tile->outFileName);
            tile->writerOpen = true;
            std::vector<SPDTile*> *openTiles = new std::vector<SPDTile*>();
            openTiles->push_back(tile);

            SPDWrite2TilesCore *write2Tiles = new SPDWrite2TilesCore(openTiles);


            reader.readAndProcessAllData(inputSPDFile, inSPDFile, write2Tiles);

            delete inSPDFile;
            write2Tiles->completeFileAndClose();
            delete write2Tiles;
            openTiles->clear();
            delete openTiles;
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch (SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
    }

    void SPDTilesUtils::buildDirectoryStruct(std::string outputBase, boost::uint_fast32_t rows, boost::uint_fast32_t cols) throw(SPDProcessingException)
    {
        try
        {
            SPDTextFileUtilities txtUtils;
            boost::filesystem::path outDirs;
            // Check the output base exsits.
            if(boost::filesystem::exists(outputBase) && boost::filesystem::is_directory(outputBase))
            {
                // Build the directory structure.
                for(boost::uint_fast32_t i = 0; i < rows; ++i)
                {
                    for(boost::uint_fast32_t j = 0; j < cols; ++j)
                    {
                        outDirs = boost::filesystem::path(outputBase);
                        outDirs /= boost::filesystem::path("Rows" + txtUtils.int32bittostring(i+1));
                        outDirs /= boost::filesystem::path("Cols" + txtUtils.int32bittostring(j+1));
                        outDirs /= boost::filesystem::path("Tiles");
                        if(!boost::filesystem::create_directories(outDirs.string()))
                        {
                            std::string outMess = std::string("Could not create directory \'") + outDirs.string() + std::string("\'");
                            throw SPDProcessingException(outMess);
                        }
                    }
                }
            }
            else
            {
                throw SPDProcessingException("The output base path does not exist or is not a directory.");
            }
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch (SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
    }


    void SPDTilesUtils::removeDirectoriesWithNoTiles(std::vector<SPDTile*> *tiles, std::string outputBase, boost::uint_fast32_t rows, boost::uint_fast32_t cols) throw(SPDProcessingException)
    {
        try
        {
            SPDTextFileUtilities txtUtils;
            boost::filesystem::path outDirs;
            // Check the output base exsits.
            if(boost::filesystem::exists(outputBase) && boost::filesystem::is_directory(outputBase))
            {
                bool notRemoved = false;
                // Build the directory structure.
                for(boost::uint_fast32_t i = 0; i < rows; ++i)
                {
                    notRemoved = false;
                    for(boost::uint_fast32_t j = 0; j < cols; ++j)
                    {
                        outDirs = boost::filesystem::path(outputBase);
                        outDirs /= boost::filesystem::path("Rows" + txtUtils.int32bittostring(i+1));
                        outDirs /= boost::filesystem::path("Cols" + txtUtils.int32bittostring(j+1));
                        outDirs /= boost::filesystem::path("Tiles");
                        if(boost::filesystem::is_directory(outDirs.string()))
                        {
                            if(SPDFileUtilities::getDIRCount(outDirs.string()) == 0)
                            {
                                outDirs = boost::filesystem::path(outputBase);
                                outDirs /= boost::filesystem::path("Rows" + txtUtils.int32bittostring(i+1));
                                outDirs /= boost::filesystem::path("Cols" + txtUtils.int32bittostring(j+1));
                                std::cout << "Remove: " << outDirs.string() << std::endl;
                                boost::filesystem::remove_all(outDirs.string());
                            }
                            else
                            {
                                std::cout << "NOT DELETING: " << outDirs.string() << std::endl;
                                notRemoved = true;
                            }
                        }
                    }

                    if(!notRemoved)
                    {
                        outDirs = boost::filesystem::path(outputBase);
                        outDirs /= boost::filesystem::path("Rows" + txtUtils.int32bittostring(i+1));
                        boost::filesystem::remove_all(outDirs.string());
                    }
                }
            }
            else
            {
                throw SPDProcessingException("The output base path does not exist or is not a directory.");
            }
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch (SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
    }

    void SPDTilesUtils::removeTilesNotInFilesList(std::vector<SPDTile*> *tiles, std::vector<std::string> inputFiles) throw(SPDProcessingException)
    {
        try
        {
            boost::uint_fast32_t row = 0;
            boost::uint_fast32_t col = 0;
            bool foundTile = false;

            for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); )
            {
                //std::cout << "Tile [" << (*iterTiles)->row << ", " << (*iterTiles)->col << "]\n";
                foundTile = false;
                for(std::vector<std::string>::iterator iterTileFiles = inputFiles.begin(); iterTileFiles != inputFiles.end(); ++iterTileFiles)
                {
                    if((*iterTileFiles) != "")
                    {
                        this->extractRowColFromFileName((*iterTileFiles), &row, &col);

                        if(((*iterTiles)->row == row) && ((*iterTiles)->col == col))
                        {
                            //std::cout << "Found ( Row = " << row << " Col = " << col << ")\n";
                            foundTile = true;
                            break;
                        }
                    }
                }

                if(foundTile)
                {
                    ++iterTiles;
                }
                else
                {
                    //std::cout << "deleting ( Row = " << (*iterTiles)->row << " Col = " << (*iterTiles)->col << ")\n";
                    delete *iterTiles;
                    tiles->erase(iterTiles);
                }

            }
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch (SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
    }

    void SPDTilesUtils::generateTileCoresShpFile(std::vector<SPDTile*> *tiles, std::string shpFile, std::string projWKTStr, bool deleteShp) throw(SPDProcessingException)
    {
        try
        {
            OGRRegisterAll();
            SPDVectorUtils vecUtils;

            std::string shpLayerName = vecUtils.getLayerName(shpFile);

            if(boost::filesystem::exists(shpFile))
            {
                // Shapefile Exists - delete it!
                if(deleteShp)
                {
                    std::cout << "Deleting shapefile\n";
                    boost::filesystem::path dirPath = boost::filesystem::path(shpFile).parent_path();
                    boost::filesystem::path file;

                    file = dirPath;
                    file /= boost::filesystem::path(shpLayerName);
                    std::string fileStr = file.string()+".shp";
                    if(boost::filesystem::exists(fileStr))
                    {
                        boost::filesystem::remove(fileStr);
                    }

                    fileStr = file.string()+".shx";
                    if(boost::filesystem::exists(fileStr))
                    {
                        boost::filesystem::remove(fileStr);
                    }

                    fileStr = file.string()+".sbx";
                    if(boost::filesystem::exists(fileStr))
                    {
                        boost::filesystem::remove(fileStr);
                    }

                    fileStr = file.string()+".sbn";
                    if(boost::filesystem::exists(fileStr))
                    {
                        boost::filesystem::remove(fileStr);
                    }

                    fileStr = file.string()+".dbf";
                    if(boost::filesystem::exists(fileStr))
                    {
                        boost::filesystem::remove(fileStr);
                    }

                    fileStr = file.string()+".prj";
                    if(boost::filesystem::exists(fileStr))
                    {
                        boost::filesystem::remove(fileStr);
                    }
                }
                else
                {
                    throw SPDProcessingException("The shapefile already exists delete it and then rerun.");
                }
            }


            /////////////////////////////////////
            //
            // Create Output Shapfile.
            //
            /////////////////////////////////////
            const char *pszDriverName = "ESRI Shapefile";
            GDALDriver *shpFiledriver = GetGDALDriverManager()->GetDriverByName(pszDriverName );
            if( shpFiledriver == NULL )
            {
                throw SPDProcessingException("SHP driver not available.");
            }
            GDALDataset *outputSHPDS = shpFiledriver->Create(shpFile.c_str(), 0, 0, 0, GDT_Unknown, NULL);
            if( outputSHPDS == NULL )
            {
                std::string message = std::string("Could not create vector file ") + shpFile;
                throw SPDProcessingException(message.c_str());
            }

            OGRSpatialReference* spatialRef = new OGRSpatialReference(projWKTStr.c_str());

            OGRLayer *outputSHPLayer = outputSHPDS->CreateLayer(shpLayerName.c_str(), spatialRef, wkbPolygon, NULL );
            if( outputSHPLayer == NULL )
            {
                std::string message = std::string("Could not create vector layer ") + shpLayerName;
                throw SPDProcessingException(message.c_str());
            }

            OGRFieldDefn shpRowField("Row", OFTInteger);
			if(outputSHPLayer->CreateField( &shpRowField ) != OGRERR_NONE )
			{
				throw SPDProcessingException("Creating shapefile \'Row\' field cluster has failed");
			}

            OGRFieldDefn shpColField("Col", OFTInteger);
			if(outputSHPLayer->CreateField( &shpColField ) != OGRERR_NONE )
			{
				throw SPDProcessingException("Creating shapefile \'Col\' field cluster has failed");
			}


            OGRFeatureDefn *outputDefn = outputSHPLayer->GetLayerDefn();
			OGRFeature *featureOutput = NULL;

            std::cout << "Adding Polygons to shapefile\n";
            for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
            {
                //std::cout << "Row: " << (*iterTiles)->row << std::endl;
                //std::cout << "Col: " << (*iterTiles)->col << std::endl;

                featureOutput = OGRFeature::CreateFeature(outputDefn);
                featureOutput->SetField(outputDefn->GetFieldIndex("Row"), ((int)(*iterTiles)->row));
                featureOutput->SetField(outputDefn->GetFieldIndex("Col"), ((int)(*iterTiles)->col));
                featureOutput->SetGeometryDirectly(vecUtils.createPolygon((*iterTiles)->xMinCore, (*iterTiles)->xMaxCore, (*iterTiles)->yMinCore, (*iterTiles)->yMaxCore));

                if( outputSHPLayer->CreateFeature(featureOutput) != OGRERR_NONE )
                {
                    throw SPDProcessingException("Failed to write feature to the output shapefile.");
                }
                OGRFeature::DestroyFeature(featureOutput);

                //std::cout << std::endl;
            }
            std::cout << "Complete\n";

            GDALClose(outputSHPDS);
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch (SPDException &e)
        {
            throw SPDProcessingException(e.what());
        }
        catch (std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
    }
		
    SPDTilesUtils::~SPDTilesUtils()
    {
	
    }



    SPDWrite2OverlapTiles::SPDWrite2OverlapTiles(std::vector<SPDTile*> *tiles) throw(SPDException):SPDImporterProcessor()
    {
        this->tiles = tiles;
        this->pls = new std::vector<SPDPulse*>();
        this->pls->reserve(1);
        this->numOpenFiles = 0;
    }

    void SPDWrite2OverlapTiles::processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException)
    {
        SPDPulseUtils plsUtils;

        try
        {
            //std::cout << "Pulse = " << pulse->pulseID << std::endl;
            for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
            {
                if(this->ptWithinTile(pulse->xIdx, pulse->yIdx, (*iterTiles)->xMin, (*iterTiles)->xMax, (*iterTiles)->yMin, (*iterTiles)->yMax))
                {
                    //std::cout << "Copying Pulse...\n";
                    SPDPulse *tmpPl = plsUtils.createSPDPulseDeepCopy(pulse);
                    //std::cout << "\tPulse (tmp) = " << tmpPl->pulseID << std::endl;
                    this->pls->push_back(tmpPl);
                    if(!(*iterTiles)->writerOpen)
                    {
                        //std::cout << "Open tile writer\n";
                        (*iterTiles)->writer->reopen((*iterTiles)->spdFile, (*iterTiles)->outFileName);
                        (*iterTiles)->writerOpen = true;
                        ++this->numOpenFiles;
                    }
                    (*iterTiles)->writer->writeDataColumn(this->pls, 0, 0);
                    if(this->numOpenFiles > 200)
                    {
                        std::cout << ".Flushing writers." << std::flush;
                        for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
                        {
                            if((*iterTiles)->writerOpen)
                            {
                                (*iterTiles)->writer->finaliseClose();
                                (*iterTiles)->writerOpen = false;
                            }
                        }
                        this->numOpenFiles = 0;
                    }
                }
            }
            SPDPulseUtils::deleteSPDPulse(pulse);
        }
        catch(SPDIOException &e)
        {
            throw e;
        }
        catch(std::exception &e)
        {
            throw SPDIOException(e.what());
        }
    }

    void SPDWrite2OverlapTiles::completeFileAndClose()throw(SPDIOException)
    {
        for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
        {
            if((*iterTiles)->writerOpen)
            {
                (*iterTiles)->writer->finaliseClose();
                (*iterTiles)->writerOpen = false;
            }
            delete (*iterTiles)->writer;
        }
    }

    void SPDWrite2OverlapTiles::setTiles(std::vector<SPDTile*> *tiles)
    {
        this->tiles = tiles;
    }

    bool SPDWrite2OverlapTiles::ptWithinTile(double x, double y, double xMin, double xMax, double yMin, double yMax)
    {
        bool withinTile = true;
        if(x < xMin)
        {
            withinTile = false;
        }
        else if(x > xMax)
        {
            withinTile = false;
        }

        if(y < yMin)
        {
            withinTile = false;
        }
        else if(y > yMax)
        {
            withinTile = false;
        }

        return withinTile;
    }

    SPDWrite2OverlapTiles::~SPDWrite2OverlapTiles()
    {
        delete this->pls;
    }





    SPDWrite2TilesCore::SPDWrite2TilesCore(std::vector<SPDTile*> *tiles) throw(SPDException):SPDImporterProcessor()
    {
        this->tiles = tiles;
        this->pls = new std::vector<SPDPulse*>();
        this->pls->reserve(1);
    }

    void SPDWrite2TilesCore::processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException)
    {
        SPDPulseUtils plsUtils;

        try
        {
            //std::cout << "Pulse = " << pulse->pulseID << std::endl;
            for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
            {
                if(this->ptWithinTile(pulse->xIdx, pulse->yIdx, (*iterTiles)->xMinCore, (*iterTiles)->xMaxCore, (*iterTiles)->yMinCore, (*iterTiles)->yMaxCore))
                {
                    //std::cout << "Copying Pulse...\n";
                    SPDPulse *tmpPl = plsUtils.createSPDPulseDeepCopy(pulse);
                    //std::cout << "\tPulse (tmp) = " << tmpPl->pulseID << std::endl;
                    this->pls->push_back(tmpPl);
                    (*iterTiles)->writer->writeDataColumn(this->pls, 0, 0);
                }
            }
            SPDPulseUtils::deleteSPDPulse(pulse);
        }
        catch(SPDIOException &e)
        {
            throw e;
        }
        catch(std::exception &e)
        {
            throw SPDIOException(e.what());
        }
    }

    void SPDWrite2TilesCore::completeFileAndClose()throw(SPDIOException)
    {
        for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
        {
            (*iterTiles)->writer->finaliseClose();
            delete (*iterTiles)->writer;
            (*iterTiles)->writerOpen = false;
        }
    }

    void SPDWrite2TilesCore::setTiles(std::vector<SPDTile*> *tiles)
    {
        this->tiles = tiles;
    }

    bool SPDWrite2TilesCore::ptWithinTile(double x, double y, double xMin, double xMax, double yMin, double yMax)
    {
        bool withinTile = true;
        if(x < xMin)
        {
            withinTile = false;
        }
        else if(x > xMax)
        {
            withinTile = false;
        }

        if(y < yMin)
        {
            withinTile = false;
        }
        else if(y > yMax)
        {
            withinTile = false;
        }

        return withinTile;
    }

    SPDWrite2TilesCore::~SPDWrite2TilesCore()
    {
        delete this->pls;
    }



}









