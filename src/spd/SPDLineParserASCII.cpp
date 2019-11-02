/*
 *  SPDLineParserASCII.h
 *
 *  Created by Pete Bunting on 21/03/2012.
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

#include "spd/SPDLineParserASCII.h"


namespace spdlib
{
    
	
	SPDLineParserASCII::SPDLineParserASCII() : SPDTextLineProcessor(), sourceID(0), ptCount(0), lineCount(0), rgbValuesFound(false)
	{
		headerRead = true;
	}
	
	bool SPDLineParserASCII::haveReadheader()
	{
		return headerRead;
	}
	
	void SPDLineParserASCII::parseHeader(std::string) 
	{
		
	}
	
	bool SPDLineParserASCII::parseLine(std::string line, SPDPulse *pl, boost::uint_fast16_t) 
	{
		SPDTextFileUtilities textUtils;
		SPDPointUtils ptUtils;
		bool returnValue = false;
		
		if((!textUtils.blankline(line)) & (!textUtils.lineStart(line, commentChar)))
		{
			if(lineCount > numLinesIgnore)
            {
                std::vector<std::string> *tokens = new std::vector<std::string>();
                textUtils.tokenizeString(line, delimiter, tokens, true);
                
                if(tokens->size() == 1)
                {
                    if(ptCount > 0)
                    {
                        ++sourceID;
                    }
                    returnValue = false;
                }
                else
                {
                    bool definedX = false;
                    bool definedY = false;
                    bool definedZ = false;
                    
                    SPDPoint *pt = new SPDPoint();
                    ptUtils.initSPDPoint(pt);
                    
                    try 
                    {
                        for(std::vector<ASCIIField>::iterator iterFields = fields.begin(); iterFields != fields.end(); ++iterFields)
                        {
                            //cout << "Field: " << (*iterFields).name << endl;
                            if((*iterFields).name == POINTMEMBERNAME_X)
                            {
                                pt->x = textUtils.strtodouble(tokens->at((*iterFields).idx));
                                definedX = true;
                            }
                            else if((*iterFields).name == POINTMEMBERNAME_Y)
                            {
                                pt->y = textUtils.strtodouble(tokens->at((*iterFields).idx));
                                definedY = true;
                            }
                            else if((*iterFields).name == POINTMEMBERNAME_Z)
                            {
                                pt->z = textUtils.strtofloat(tokens->at((*iterFields).idx));
                                definedZ = true;
                            }
                            else if((*iterFields).name == POINTMEMBERNAME_HEIGHT)
                            {
                                pt->height = textUtils.strtofloat(tokens->at((*iterFields).idx));
                            }
                            else if((*iterFields).name == POINTMEMBERNAME_RANGE)
                            {
                                pt->range = textUtils.strtofloat(tokens->at((*iterFields).idx));
                            }
                            else if((*iterFields).name == POINTMEMBERNAME_AMPLITUDE_RETURN)
                            {
                                pt->amplitudeReturn = textUtils.strtofloat(tokens->at((*iterFields).idx));
                            }
                            else if((*iterFields).name == POINTMEMBERNAME_WIDTH_RETURN)
                            {
                                pt->widthReturn = textUtils.strtofloat(tokens->at((*iterFields).idx));
                            }
                            else if((*iterFields).name == POINTMEMBERNAME_RED)
                            {
                                if((*iterFields).dataType == spd_int)
                                {
                                    pt->red = textUtils.strto16bitInt(tokens->at((*iterFields).idx));
                                }
                                else if((*iterFields).dataType == spd_uint)
                                {
                                    pt->red = textUtils.strto16bitUInt(tokens->at((*iterFields).idx));
                                }
                                else if(((*iterFields).dataType == spd_float) |((*iterFields).dataType == spd_double))
                                {
                                    pt->red = floor(textUtils.strtodouble(tokens->at((*iterFields).idx)));
                                }
                            }
                            else if((*iterFields).name == POINTMEMBERNAME_GREEN)
                            {
                                if((*iterFields).dataType == spd_int)
                                {
                                    pt->green = textUtils.strto16bitInt(tokens->at((*iterFields).idx));
                                }
                                else if((*iterFields).dataType == spd_uint)
                                {
                                    pt->green = textUtils.strto16bitUInt(tokens->at((*iterFields).idx));
                                }
                                else if(((*iterFields).dataType == spd_float) |((*iterFields).dataType == spd_double))
                                {
                                    pt->green = floor(textUtils.strtodouble(tokens->at((*iterFields).idx)));
                                }
                            }
                            else if((*iterFields).name == POINTMEMBERNAME_BLUE)
                            {
                                if((*iterFields).dataType == spd_int)
                                {
                                    pt->blue = textUtils.strto16bitInt(tokens->at((*iterFields).idx));
                                }
                                else if((*iterFields).dataType == spd_uint)
                                {
                                    pt->blue = textUtils.strto16bitUInt(tokens->at((*iterFields).idx));
                                }
                                else if(((*iterFields).dataType == spd_float) |((*iterFields).dataType == spd_double))
                                {
                                    pt->blue = floor(textUtils.strtodouble(tokens->at((*iterFields).idx)));
                                }
                            }
                            else if((*iterFields).name == POINTMEMBERNAME_CLASSIFICATION)
                            {
                                pt->classification = textUtils.strto16bitUInt(tokens->at((*iterFields).idx));
                            }
                            else if((*iterFields).name == PULSEMEMBERNAME_AZIMUTH)
                            {
                                pl->azimuth = textUtils.strtodouble(tokens->at((*iterFields).idx));
                            }
                            else if((*iterFields).name == PULSEMEMBERNAME_ZENITH)
                            {
                                pl->zenith = textUtils.strtodouble(tokens->at((*iterFields).idx));
                            }
                            else if((*iterFields).name == PULSEMEMBERNAME_AMPLITUDE_PULSE)
                            {
                                pl->amplitudePulse = textUtils.strtodouble(tokens->at((*iterFields).idx));
                            }
                            else if((*iterFields).name == PULSEMEMBERNAME_WIDTH_PULSE)
                            {
                                pl->widthPulse = textUtils.strtodouble(tokens->at((*iterFields).idx));
                            }
                            else if((*iterFields).name == PULSEMEMBERNAME_SOURCE_ID)
                            {
                                pl->sourceID = textUtils.strto16bitUInt(tokens->at((*iterFields).idx));
                            }
                            else if((*iterFields).name == PULSEMEMBERNAME_SCANLINE)
                            {
                                pl->scanline = textUtils.strto32bitUInt(tokens->at((*iterFields).idx));
                            }
                            else if((*iterFields).name == PULSEMEMBERNAME_SCANLINE_IDX)
                            {
                                pl->scanlineIdx = textUtils.strto16bitUInt(tokens->at((*iterFields).idx));
                            }
                            else
                            {
                                std::cerr << "Could not find field: \'" << (*iterFields).name << "\'\n";
                                throw SPDIOException("Field was not recognised.");
                            }
                        }
                        
                        pl->xIdx = pt->x;
                        pl->yIdx = pt->y;
                        pl->pts->push_back(pt);
                        pl->numberOfReturns = 1;
                        ++ptCount;
                        
                        if(!definedX | !definedY | !definedZ)
                        {
                            throw SPDIOException("At the very minimum the X, Y, Z fields must be populated.");
                        }
                        
                        returnValue = true;
                    } 
                    catch (std::out_of_range &e) 
                    {
                        std::cerr << "WARNING: " << e.what() << std::endl;
                        std::cerr << "Could not parse line: " << line << std::endl;
                        std::cerr << "Processing has continued and this line has been ignored.\n";
                    }
                    catch (std::exception &e) 
                    {
                        std::cerr << "WARNING: " << e.what() << std::endl;
                        std::cerr << "Could not parse line: " << line << std::endl;
                        std::cerr << "Processing has continued and this line has been ignored.\n";
                    }
                }
            
                delete tokens;
            }
            
            ++lineCount;
		}
		return returnValue;
	}
    
	bool SPDLineParserASCII::isFileType(std::string fileType)
	{
		if(fileType == "ASCII")
		{
			return true;
		}
		return false;
	}
	
	void SPDLineParserASCII::saveHeaderValues(SPDFile *spdFile)
	{
		spdFile->setDiscretePtDefined(SPD_TRUE);
		spdFile->setDecomposedPtDefined(SPD_FALSE);
		spdFile->setTransWaveformDefined(SPD_FALSE);
        spdFile->setReceiveWaveformDefined(SPD_FALSE);
		if(rgbValuesFound)
		{
			spdFile->setRGBDefined(SPD_TRUE);
		}
		else
		{
			spdFile->setRGBDefined(SPD_FALSE);
		}
	}
	
	void SPDLineParserASCII::reset()
	{
		rgbValuesFound = false;
		sourceID = 0;
		ptCount = 0;
	}
    
    void SPDLineParserASCII::parseSchema(std::string schema)
    {
        SPDTextFileUtilities textUtils;
        XMLCh tempStr[100];
        xercesc::DOMImplementation *impl = NULL;
		xercesc::DOMLSParser* parser = NULL;
		xercesc::ErrorHandler* errHandler = NULL;
		xercesc::DOMDocument *doc = NULL;
		xercesc::DOMElement *rootElement = NULL;
		xercesc::DOMNodeList *fieldsList = NULL;
		xercesc::DOMElement *fieldElement = NULL;
        try 
		{
            xercesc::XMLPlatformUtils::Initialize();
            
            XMLCh *lineTag = xercesc::XMLString::transcode("line");
            XMLCh *fieldTag = xercesc::XMLString::transcode("field");
            XMLCh *delimterAttStr = xercesc::XMLString::transcode("delimiter");
            XMLCh *commentCharAttStr = xercesc::XMLString::transcode("comment");
            XMLCh *ignoreLinesAttStr = xercesc::XMLString::transcode("ignorelines");
            XMLCh *nameAttStr = xercesc::XMLString::transcode("name");
            XMLCh *typeAttStr = xercesc::XMLString::transcode("type");
            XMLCh *indexAttStr = xercesc::XMLString::transcode("index");
            
            XMLCh *optionSPDInt = xercesc::XMLString::transcode("spd_int");
            XMLCh *optionSPDUInt = xercesc::XMLString::transcode("spd_uint");
            XMLCh *optionSPDFloat = xercesc::XMLString::transcode("spd_float");
            XMLCh *optionSPDDouble = xercesc::XMLString::transcode("spd_double");
            
            xercesc::XMLString::transcode("LS", tempStr, 99);
			impl = xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
			if(impl == NULL)
			{
				throw SPDIOException("DOMImplementation is NULL");
			}
			
			// Create Parser
			parser = ((xercesc::DOMImplementationLS*)impl)->createLSParser(xercesc::DOMImplementationLS::MODE_SYNCHRONOUS, 0);
			errHandler = (xercesc::ErrorHandler*) new xercesc::HandlerBase();
			parser->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMErrorHandler, errHandler);
			            
			// Open Document
			doc = parser->parseURI(schema.c_str());	
			
			// Get the Root element
			rootElement = doc->getDocumentElement();
			//cout << "Root Element: " << xercesc::XMLString::transcode(rootElement->getTagName()) << endl;
			if(!xercesc::XMLString::equals(rootElement->getTagName(), lineTag))
			{
				throw SPDIOException("Incorrect root element; Root element should be \"list\"");
			}
            
            if(rootElement->hasAttribute(delimterAttStr))
            {
                char *charValue = xercesc::XMLString::transcode(rootElement->getAttribute(delimterAttStr));
                this->delimiter = charValue[0];
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                throw SPDIOException("No \'delimiter\' attribute was provided.");
            }
            
            if(rootElement->hasAttribute(commentCharAttStr))
            {
                char *charValue = xercesc::XMLString::transcode(rootElement->getAttribute(commentCharAttStr));
                this->commentChar = charValue[0];
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                throw SPDIOException("No \'comment\' attribute was provided.");
            }
            
            if(rootElement->hasAttribute(ignoreLinesAttStr))
            {
                char *charValue = xercesc::XMLString::transcode(rootElement->getAttribute(ignoreLinesAttStr));
                this->numLinesIgnore = textUtils.strto32bitUInt(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                throw SPDIOException("No \'ignorelines\' attribute was provided.");
            }
            
            

            fieldsList = rootElement->getElementsByTagName(fieldTag);
			boost::uint_fast32_t numFields = fieldsList->getLength();
			std::cout << "There are " << numFields << " fields in the schema." << std::endl;
			
            fields.reserve(numFields);
			
            std::string nameVal;
            SPDDataType dataTypeVal;
            boost::uint_fast16_t idxVal;
            
			for(int i = 0; i < numFields; i++)
			{
				fieldElement = static_cast<xercesc::DOMElement*>(fieldsList->item(i));
				nameVal = "";
                dataTypeVal = spd_float;
                idxVal = 0;
                
                if(fieldElement->hasAttribute(nameAttStr))
                {
                    char *charValue = xercesc::XMLString::transcode(fieldElement->getAttribute(nameAttStr));
                    nameVal = std::string(charValue);
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDIOException("No \'name\' attribute was provided.");
                }
                
                if(fieldElement->hasAttribute(typeAttStr))
                {
                    const XMLCh *dataTypeValXMLStr = fieldElement->getAttribute(typeAttStr);
                    if(xercesc::XMLString::equals(dataTypeValXMLStr, optionSPDInt))
                    {
                        dataTypeVal = spd_int;
                    }
                    else if(xercesc::XMLString::equals(dataTypeValXMLStr, optionSPDUInt))
                    {
                        dataTypeVal = spd_uint;
                    }
                    else if(xercesc::XMLString::equals(dataTypeValXMLStr, optionSPDFloat))
                    {
                        dataTypeVal = spd_float;
                    }
                    else if(xercesc::XMLString::equals(dataTypeValXMLStr, optionSPDDouble))
                    {
                        dataTypeVal = spd_double;
                    }
                    else
                    {
                        throw SPDIOException("datatype was not recognised.");
                    }
                }
                else
                {
                    throw SPDIOException("No \'type\' attribute was provided.");
                }
                
                
                if(fieldElement->hasAttribute(indexAttStr))
                {
                    char *charValue = xercesc::XMLString::transcode(fieldElement->getAttribute(indexAttStr));
                    idxVal = textUtils.strto16bitUInt(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDIOException("No \'index\' attribute was provided.");
                }
				
                fields.push_back(ASCIIField(nameVal, dataTypeVal, idxVal));
			}
                        
            parser->release();
			delete errHandler;
			xercesc::XMLString::release(&lineTag);
			xercesc::XMLString::release(&fieldTag);
			xercesc::XMLString::release(&delimterAttStr);
            xercesc::XMLString::release(&commentCharAttStr);
			xercesc::XMLString::release(&ignoreLinesAttStr);
			xercesc::XMLString::release(&nameAttStr);
            xercesc::XMLString::release(&typeAttStr);
			xercesc::XMLString::release(&indexAttStr);
            xercesc::XMLString::release(&optionSPDInt);
			xercesc::XMLString::release(&optionSPDUInt);
            xercesc::XMLString::release(&optionSPDFloat);
			xercesc::XMLString::release(&optionSPDDouble);
            			
			xercesc::XMLPlatformUtils::Terminate();
        }
        catch (const xercesc::XMLException& e) 
		{
			parser->release();
			char *message = xercesc::XMLString::transcode(e.getMessage());
			std::string outMessage =  std::string("XMLException : ") + std::string(message);
			throw SPDIOException(outMessage.c_str());
		}
		catch (const xercesc::DOMException& e) 
		{
			parser->release();
			char *message = xercesc::XMLString::transcode(e.getMessage());
			std::string outMessage =  std::string("DOMException : ") + std::string(message);
			throw SPDIOException(outMessage.c_str());
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
	
	SPDLineParserASCII::~SPDLineParserASCII()
	{
		
	}
	
}

