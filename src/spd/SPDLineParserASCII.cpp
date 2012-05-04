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
	
	void SPDLineParserASCII::parseHeader(string) throw(SPDIOException)
	{
		
	}
	
	bool SPDLineParserASCII::parseLine(string line, SPDPulse *pl, boost::uint_fast16_t) throw(SPDIOException)
	{
		SPDTextFileUtilities textUtils;
		SPDPointUtils ptUtils;
		bool returnValue = false;
		
		if((!textUtils.blankline(line)) & (!textUtils.lineStart(line, commentChar)))
		{
			if(lineCount > numLinesIgnore)
            {
                vector<string> *tokens = new vector<string>();
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
                        for(vector<ASCIIField>::iterator iterFields = fields.begin(); iterFields != fields.end(); ++iterFields)
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
                                cerr << "Could not find field: \'" << (*iterFields).name << "\'\n";
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
                    catch (out_of_range &e) 
                    {
                        cerr << "WARNING: " << e.what() << endl;
                        cerr << "Could not parse line: " << line << endl;
                        cerr << "Processing has continued and this line has been ignored.\n";
                    }
                    catch (exception &e) 
                    {
                        cerr << "WARNING: " << e.what() << endl;
                        cerr << "Could not parse line: " << line << endl;
                        cerr << "Processing has continued and this line has been ignored.\n";
                    }
                }
            
                delete tokens;
            }
            
            ++lineCount;
		}
		return returnValue;
	}
    
	bool SPDLineParserASCII::isFileType(string fileType)
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
    
    void SPDLineParserASCII::parseSchema(string schema)throw(SPDIOException)
    {
        SPDTextFileUtilities textUtils;
        XMLCh tempStr[100];
        DOMImplementation *impl = NULL;
		DOMLSParser* parser = NULL;
		ErrorHandler* errHandler = NULL;
		DOMDocument *doc = NULL;
		DOMElement *rootElement = NULL;
		DOMNodeList *fieldsList = NULL;
		DOMElement *fieldElement = NULL;
        try 
		{
            XMLPlatformUtils::Initialize();
            
            XMLCh *lineTag = XMLString::transcode("line");
            XMLCh *fieldTag = XMLString::transcode("field");
            XMLCh *delimterAttStr = XMLString::transcode("delimiter");
            XMLCh *commentCharAttStr = XMLString::transcode("comment");
            XMLCh *ignoreLinesAttStr = XMLString::transcode("ignorelines");
            XMLCh *nameAttStr = XMLString::transcode("name");
            XMLCh *typeAttStr = XMLString::transcode("type");
            XMLCh *indexAttStr = XMLString::transcode("index");
            
            XMLCh *optionSPDInt = XMLString::transcode("spd_int");
            XMLCh *optionSPDUInt = XMLString::transcode("spd_uint");
            XMLCh *optionSPDFloat = XMLString::transcode("spd_float");
            XMLCh *optionSPDDouble = XMLString::transcode("spd_double");
            
            XMLString::transcode("LS", tempStr, 99);
			impl = DOMImplementationRegistry::getDOMImplementation(tempStr);
			if(impl == NULL)
			{
				throw SPDIOException("DOMImplementation is NULL");
			}
			
			// Create Parser
			parser = ((DOMImplementationLS*)impl)->createLSParser(DOMImplementationLS::MODE_SYNCHRONOUS, 0);
			errHandler = (ErrorHandler*) new HandlerBase();
			parser->getDomConfig()->setParameter(XMLUni::fgDOMErrorHandler, errHandler);
			            
			// Open Document
			doc = parser->parseURI(schema.c_str());	
			
			// Get the Root element
			rootElement = doc->getDocumentElement();
			//cout << "Root Element: " << XMLString::transcode(rootElement->getTagName()) << endl;
			if(!XMLString::equals(rootElement->getTagName(), lineTag))
			{
				throw SPDIOException("Incorrect root element; Root element should be \"list\"");
			}
            
            if(rootElement->hasAttribute(delimterAttStr))
            {
                char *charValue = XMLString::transcode(rootElement->getAttribute(delimterAttStr));
                this->delimiter = charValue[0];
                XMLString::release(&charValue);
            }
            else
            {
                throw SPDIOException("No \'delimiter\' attribute was provided.");
            }
            
            if(rootElement->hasAttribute(commentCharAttStr))
            {
                char *charValue = XMLString::transcode(rootElement->getAttribute(commentCharAttStr));
                this->commentChar = charValue[0];
                XMLString::release(&charValue);
            }
            else
            {
                throw SPDIOException("No \'comment\' attribute was provided.");
            }
            
            if(rootElement->hasAttribute(ignoreLinesAttStr))
            {
                char *charValue = XMLString::transcode(rootElement->getAttribute(ignoreLinesAttStr));
                this->numLinesIgnore = textUtils.strto32bitUInt(string(charValue));
                XMLString::release(&charValue);
            }
            else
            {
                throw SPDIOException("No \'ignorelines\' attribute was provided.");
            }
            
            

            fieldsList = rootElement->getElementsByTagName(fieldTag);
			boost::uint_fast32_t numFields = fieldsList->getLength();
			cout << "There are " << numFields << " fields in the schema." << endl;
			
            fields.reserve(numFields);
			
            string nameVal;
            SPDDataType dataTypeVal;
            boost::uint_fast16_t idxVal;
            
			for(int i = 0; i < numFields; i++)
			{
				fieldElement = static_cast<DOMElement*>(fieldsList->item(i));
				nameVal = "";
                dataTypeVal = spd_float;
                idxVal = 0;
                
                if(fieldElement->hasAttribute(nameAttStr))
                {
                    char *charValue = XMLString::transcode(fieldElement->getAttribute(nameAttStr));
                    nameVal = string(charValue);
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDIOException("No \'name\' attribute was provided.");
                }
                
                if(fieldElement->hasAttribute(typeAttStr))
                {
                    const XMLCh *dataTypeValXMLStr = fieldElement->getAttribute(typeAttStr);
                    if(XMLString::equals(dataTypeValXMLStr, optionSPDInt))
                    {
                        dataTypeVal = spd_int;
                    }
                    else if(XMLString::equals(dataTypeValXMLStr, optionSPDUInt))
                    {
                        dataTypeVal = spd_uint;
                    }
                    else if(XMLString::equals(dataTypeValXMLStr, optionSPDFloat))
                    {
                        dataTypeVal = spd_float;
                    }
                    else if(XMLString::equals(dataTypeValXMLStr, optionSPDDouble))
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
                    char *charValue = XMLString::transcode(fieldElement->getAttribute(indexAttStr));
                    idxVal = textUtils.strto16bitUInt(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDIOException("No \'index\' attribute was provided.");
                }
				
                fields.push_back(ASCIIField(nameVal, dataTypeVal, idxVal));
			}
                        
            parser->release();
			delete errHandler;
			XMLString::release(&lineTag);
			XMLString::release(&fieldTag);
			XMLString::release(&delimterAttStr);
            XMLString::release(&commentCharAttStr);
			XMLString::release(&ignoreLinesAttStr);
			XMLString::release(&nameAttStr);
            XMLString::release(&typeAttStr);
			XMLString::release(&indexAttStr);
            XMLString::release(&optionSPDInt);
			XMLString::release(&optionSPDUInt);
            XMLString::release(&optionSPDFloat);
			XMLString::release(&optionSPDDouble);
            			
			XMLPlatformUtils::Terminate();
        }
        catch (const XMLException& e) 
		{
			parser->release();
			char *message = XMLString::transcode(e.getMessage());
			string outMessage =  string("XMLException : ") + string(message);
			throw SPDIOException(outMessage.c_str());
		}
		catch (const DOMException& e) 
		{
			parser->release();
			char *message = XMLString::transcode(e.getMessage());
			string outMessage =  string("DOMException : ") + string(message);
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

