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
            
            boost::uint_fast32_t numCols = floor(sceneXSize/xSize);
            boost::uint_fast32_t numRows = floor(sceneYSize/ySize);
            
            double leftOverX = sceneXSize - (numCols * xSize);
            double leftOverY = sceneYSize - (numRows * ySize);
            
            if( leftOverX > 0)
            {
                ++leftOverX;
            }
            
            if( leftOverY > 0)
            {
                ++leftOverY;
            }
            
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
    
    void SPDTilesUtils::createTileSPDFiles(std::vector<SPDTile*> *tiles, SPDFile *templateSPDFile, std::string outputBase, double xSize, double ySize, double overlap, double xMin, double xMax, double yMin, double yMax, boost::uint_fast32_t rows, boost::uint_fast32_t cols) throw(SPDProcessingException)
    {
        SPDTextFileUtilities txtUtils;
        try
        {
            for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
            {
                //std::cout << "[" << (*iterTiles)->col << " " << (*iterTiles)->row << "] ALL: [" << (*iterTiles)->xMin << "," << (*iterTiles)->xMax << "][" << (*iterTiles)->yMin << "," << (*iterTiles)->yMax << "] CORE: [" << (*iterTiles)->xMinCore << "," << (*iterTiles)->xMaxCore << "][" << (*iterTiles)->yMinCore << "," << (*iterTiles)->yMaxCore << "]" << std::endl;
                if((*iterTiles)->outFileName == "")
                {
                    (*iterTiles)->outFileName = outputBase + std::string("_r") + txtUtils.uInt32bittostring((*iterTiles)->row) + std::string("c") + txtUtils.uInt32bittostring((*iterTiles)->col) + std::string(".spd");
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
    
    void SPDTilesUtils::populateTileWithData(std::vector<SPDTile*> *tiles, std::vector<std::string> inputFiles) throw(SPDProcessingException)
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
                    std::cout << "Input file: " << *iterFiles << std::endl;
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
		
    SPDTilesUtils::~SPDTilesUtils()
    {
	
    }
    
    
    
    SPDWrite2OverlapTiles::SPDWrite2OverlapTiles(std::vector<SPDTile*> *tiles) throw(SPDException):SPDImporterProcessor()
    {
        this->tiles = tiles;
        this->pls = new std::vector<SPDPulse*>();
        this->pls->reserve(1);
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
    
    void SPDWrite2OverlapTiles::completeFileAndClose()throw(SPDIOException)
    {
        for(std::vector<SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
        {
            (*iterTiles)->writer->finaliseClose();
            delete (*iterTiles)->writer;
            (*iterTiles)->writerOpen = false;
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

    
    
    
}









