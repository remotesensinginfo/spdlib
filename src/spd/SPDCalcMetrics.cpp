/*
 *  SPDCalcMetrics.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 17/03/2011.
 *  Copyright 2011 SPDLib. All rights reserved.
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
 */

#include "spd/SPDCalcMetrics.h"


namespace spdlib
{

    SPDCalcMetrics::SPDCalcMetrics()
    {
        
    }
    
    void SPDCalcMetrics::calcMetricToImage(string inXMLFilePath, string inputSPDFile, string outputImage, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize, float processingResolution, string gdalFormat) throw (SPDProcessingException)
    {
        try 
        {
            vector<SPDMetric*> *metrics = new vector<SPDMetric*>();
            vector<string> *fieldNames = new vector<string>();
            this->parseMetricsXML(inXMLFilePath, metrics, fieldNames);
            
            if(metrics->size() != fieldNames->size())
            {
                throw SPDProcessingException("The number of metrics and fieldnames needs to be the same.");
            }
            
            cout << metrics->size() << " metrics where found." << endl;
            
            
            SPDFile *spdInFile = new SPDFile(inputSPDFile);
            SPDPulseProcessor *pulseStatsProcessor = new SPDCalcImageMetrics(metrics, fieldNames);           
            SPDSetupProcessPulses processPulses = SPDSetupProcessPulses(blockXSize, blockYSize, true);
            processPulses.processPulsesWithOutputImage(pulseStatsProcessor, spdInFile, outputImage, metrics->size(), processingResolution, gdalFormat, false, 0);
            
            delete spdInFile;
            delete pulseStatsProcessor;
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
    }
    
    void SPDCalcMetrics::calcMetricToVectorShp(string inXMLFilePath, string inputSPDFile, string inputVectorShp, string outputVectorShp, bool deleteOutShp, bool copyAttributes) throw (SPDProcessingException)
    {
        try 
        {
            vector<SPDMetric*> *metrics = new vector<SPDMetric*>();
            vector<string> *fieldNames = new vector<string>();
            this->parseMetricsXML(inXMLFilePath, metrics, fieldNames);
            
            if(metrics->size() != fieldNames->size())
            {
                throw SPDProcessingException("The number of metrics and fieldnames needs to be the same.");
            }
            
            cout << metrics->size() << " metrics where found." << endl;
            
            SPDPolygonProcessor *polyProcessor = new SPDCalcPolyMetrics(metrics, fieldNames);
            SPDSetupProcessShapefilePolygons processPolygons;
            processPolygons.processPolygons(inputSPDFile, inputVectorShp, outputVectorShp, deleteOutShp, copyAttributes, polyProcessor);
            delete polyProcessor;
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
    }
    
    void SPDCalcMetrics::calcMetricForVector2ASCII(string inXMLFilePath, string inputSPDFile, string inputVectorShp, string outputASCII) throw (SPDProcessingException)
    {
        try 
        {
            vector<SPDMetric*> *metrics = new vector<SPDMetric*>();
            vector<string> *fieldNames = new vector<string>();
            this->parseMetricsXML(inXMLFilePath, metrics, fieldNames);
            
            if(metrics->size() != fieldNames->size())
            {
                throw SPDProcessingException("The number of metrics and fieldnames needs to be the same.");
            }
            
            cout << metrics->size() << " metrics where found." << endl;
            
            SPDPolygonProcessor *polyProcessor = new SPDCalcPolyMetrics(metrics, fieldNames);
            SPDSetupProcessShapefilePolygons processPolygons;
            processPolygons.processPolygons(inputSPDFile, inputVectorShp, outputASCII, polyProcessor);
            delete polyProcessor;
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
    }
    
    void SPDCalcMetrics::parseMetricsXML(string inXMLFilePath, vector<SPDMetric*> *metrics, vector<string> *fieldNames) throw(SPDProcessingException)
    {
        cout << "Reading XML file: " << inXMLFilePath << endl;
        DOMLSParser* parser = NULL;
		try 
		{
			XMLPlatformUtils::Initialize();
            
            XMLCh tempStr[100];
            DOMImplementation *impl = NULL;
            ErrorHandler* errHandler = NULL;
            DOMDocument *xmlDoc = NULL;
            DOMElement *rootElement = NULL;
            XMLCh *metricsTagStr = NULL;
            XMLCh *metricTagStr = NULL;
            DOMElement *metricElement = NULL;
            XMLCh *fieldXMLStr = XMLString::transcode("field");
            
            
			metricsTagStr = XMLString::transcode("spdlib:metrics");
			metricTagStr = XMLString::transcode("spdlib:metric");
            
			XMLString::transcode("LS", tempStr, 99);
			impl = DOMImplementationRegistry::getDOMImplementation(tempStr);
			if(impl == NULL)
			{
				throw SPDProcessingException("DOMImplementation is NULL");
			}
			
			// Create Parser
			parser = ((DOMImplementationLS*)impl)->createLSParser(DOMImplementationLS::MODE_SYNCHRONOUS, 0);
			errHandler = (ErrorHandler*) new HandlerBase();
			parser->getDomConfig()->setParameter(XMLUni::fgDOMErrorHandler, errHandler);
			
			// Open Document
			xmlDoc = parser->parseURI(inXMLFilePath.c_str());	
			
			// Get the Root element
			rootElement = xmlDoc->getDocumentElement();
			if(!XMLString::equals(rootElement->getTagName(), metricsTagStr))
			{
				throw SPDProcessingException("Incorrect root element; Root element should be \"spdlib:metrics\"");
			}
            
			boost::uint_fast32_t numMetrics = rootElement->getChildElementCount();
            metricElement = rootElement->getFirstElementChild();
            for(boost::uint_fast32_t i = 0; i < numMetrics; ++i)
            {
                // Retreive name (used for naming image band or vector attribute)
                if(metricElement->hasAttribute(fieldXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(fieldXMLStr));
                    fieldNames->push_back(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("No \'field\' attribute was provided for root metric.");
                }
                
                // Retrieve Metric and add to list.
                metrics->push_back(this->createMetric(metricElement));
                
                // Move on to next metric
				metricElement = metricElement->getNextElementSibling();
            }
            
            parser->release();
			delete errHandler;
			XMLString::release(&metricsTagStr);
			XMLString::release(&metricTagStr);
            XMLString::release(&fieldXMLStr);
			
			XMLPlatformUtils::Terminate();
        }
		catch (const XMLException& e) 
		{
			parser->release();
			char *message = XMLString::transcode(e.getMessage());
			string outMessage =  string("XMLException : ") + string(message);
			throw SPDProcessingException(outMessage.c_str());
		}
		catch (const DOMException& e) 
		{
			parser->release();
			char *message = XMLString::transcode(e.getMessage());
			string outMessage =  string("DOMException : ") + string(message);
			throw SPDProcessingException(outMessage.c_str());
		}
		catch(SPDProcessingException &e)
		{
			throw e;
		}
    }
    
    SPDMetric* SPDCalcMetrics::createMetric(DOMElement *metricElement) throw(SPDProcessingException)
    {
        XMLCh *metricadd = XMLString::transcode("add");
        XMLCh *metricminus = XMLString::transcode("minus");
        XMLCh *metricmultiply = XMLString::transcode("multiply");
        XMLCh *metricdivide = XMLString::transcode("divide");
        XMLCh *metricpow = XMLString::transcode("pow");
        XMLCh *metricabs = XMLString::transcode("abs");
        XMLCh *metricsqrt = XMLString::transcode("sqrt");
        XMLCh *metricsine = XMLString::transcode("sine");
        XMLCh *metriccosine = XMLString::transcode("cosine");
        XMLCh *metrictangent = XMLString::transcode("tangent");
        XMLCh *metricinvsine = XMLString::transcode("invsine");
        XMLCh *metricinvcos = XMLString::transcode("invcos");
        XMLCh *metricinvtan = XMLString::transcode("invtan");
        XMLCh *metriclog10 = XMLString::transcode("log10");
        XMLCh *metricln = XMLString::transcode("ln");
        XMLCh *metricexp = XMLString::transcode("exp");
        XMLCh *metricpercentage = XMLString::transcode("percentage");
        XMLCh *metricaddconst = XMLString::transcode("addconst");
        XMLCh *metricminusconstfrom = XMLString::transcode("minusconstfrom");
        XMLCh *metricminusfromconst = XMLString::transcode("minusfromconst");
        XMLCh *metricmultiplyconst = XMLString::transcode("multiplyconst");
        XMLCh *metricdividebyconst = XMLString::transcode("dividebyconst");
        XMLCh *metricdivideconstby = XMLString::transcode("divideconstby");
        XMLCh *metricpowmetricconst = XMLString::transcode("powmetricconst");
        XMLCh *metricpowconstmetric = XMLString::transcode("powconstmetric");
        XMLCh *metricnumpulses = XMLString::transcode("numpulses");
        XMLCh *metriccanopycover = XMLString::transcode("canopycover");
        XMLCh *metriccanopycoverpercent = XMLString::transcode("canopycoverpercent");
        XMLCh *metricleeopenness = XMLString::transcode("hscoi");
        XMLCh *metricnumreturnsheight = XMLString::transcode("numreturnsheight");
        XMLCh *metricsumheight = XMLString::transcode("sumheight");
        XMLCh *metricmeanheight = XMLString::transcode("meanheight");
        XMLCh *metricmedianheight = XMLString::transcode("medianheight");
        XMLCh *metricmodeheight = XMLString::transcode("modeheight");
        XMLCh *metricminheight = XMLString::transcode("minheight");
        XMLCh *metricmaxheight = XMLString::transcode("maxheight");
        XMLCh *metricmaxdominant = XMLString::transcode("dominantheight");
        XMLCh *metricstddevheight = XMLString::transcode("stddevheight");
        XMLCh *metricvarianceheight = XMLString::transcode("varianceheight");
        XMLCh *metricabsdeviationheight = XMLString::transcode("absdeviationheight");
        XMLCh *metriccoefficientofvariationheight = XMLString::transcode("coefficientofvariationheight");
        XMLCh *metricpercentileheight = XMLString::transcode("percentileheight");
        XMLCh *metricskewnessheight = XMLString::transcode("skewnessheight");
        XMLCh *metricpersonmodeheight = XMLString::transcode("personmodeheight");
        XMLCh *metricpersonmedianheight = XMLString::transcode("personmedianheight");
        XMLCh *metrickurtosisheight = XMLString::transcode("kurtosisheight");
        XMLCh *metricreturnsaboveheightmetric = XMLString::transcode("returnsaboveheightmetric");
        XMLCh *metricreturnsbelowheightmetric = XMLString::transcode("returnsbelowheightmetric");
        XMLCh *metricnumreturnsz = XMLString::transcode("numreturnsz");
        XMLCh *metricsumz = XMLString::transcode("sumz");
        XMLCh *metricmeanz = XMLString::transcode("meanz");
        XMLCh *metricmedianz = XMLString::transcode("medianz");
        XMLCh *metricmodez = XMLString::transcode("modez");
        XMLCh *metricminz = XMLString::transcode("minz");
        XMLCh *metricmaxz = XMLString::transcode("maxz");
        XMLCh *metricstddevz = XMLString::transcode("stddevz");
        XMLCh *metricvariancez = XMLString::transcode("variancez");
        XMLCh *metricabsdeviationz = XMLString::transcode("absdeviationz");
        XMLCh *metriccoefficientofvariationz = XMLString::transcode("coefficientofvariationz");
        XMLCh *metricpercentilez = XMLString::transcode("percentilez");
        XMLCh *metricskewnessz = XMLString::transcode("skewnessz");
        XMLCh *metricpersonmodez = XMLString::transcode("personmodez");
        XMLCh *metricpersonmedianz = XMLString::transcode("personmedianz");
        XMLCh *metrickurtosisz = XMLString::transcode("kurtosisz");
        XMLCh *metricreturnsabovezmetric = XMLString::transcode("returnsabovezmetric");
        XMLCh *metricreturnsbelowzmetric = XMLString::transcode("returnsbelowzmetric");
        XMLCh *metricnumreturnsamplitude = XMLString::transcode("numreturnsamplitude");
        XMLCh *metricsumamplitude = XMLString::transcode("sumamplitude");
        XMLCh *metricmeanamplitude = XMLString::transcode("meanamplitude");
        XMLCh *metricmedianamplitude = XMLString::transcode("medianamplitude");
        XMLCh *metricmodeamplitude = XMLString::transcode("modeamplitude");
        XMLCh *metricminamplitude = XMLString::transcode("minamplitude");
        XMLCh *metricmaxamplitude = XMLString::transcode("maxamplitude");
        XMLCh *metricstddevamplitude = XMLString::transcode("stddevamplitude");
        XMLCh *metricvarianceamplitude = XMLString::transcode("varianceamplitude");
        XMLCh *metricabsdeviationamplitude = XMLString::transcode("absdeviationamplitude");
        XMLCh *metriccoefficientofvariationamplitude = XMLString::transcode("coefficientofvariationamplitude");
        XMLCh *metricpercentileamplitude = XMLString::transcode("percentileamplitude");
        XMLCh *metricskewnessamplitude = XMLString::transcode("skewnessamplitude");
        XMLCh *metricpersonmodeamplitude = XMLString::transcode("personmodeamplitude");
        XMLCh *metricpersonmedianamplitude = XMLString::transcode("personmedianamplitude");
        XMLCh *metrickurtosisamplitude = XMLString::transcode("kurtosisamplitude");
        XMLCh *metricreturnsaboveamplitudemetric = XMLString::transcode("returnsaboveamplitudemetric");
        XMLCh *metricreturnsbelowamplitudemetric = XMLString::transcode("returnsbelowamplitudemetric");
        XMLCh *metricnumreturnsrange = XMLString::transcode("numreturnsrange");
        XMLCh *metricsumrange = XMLString::transcode("sumrange");
        XMLCh *metricmeanrange = XMLString::transcode("meanrange");
        XMLCh *metricmedianrange = XMLString::transcode("medianrange");
        XMLCh *metricmoderange = XMLString::transcode("moderange");
        XMLCh *metricminrange = XMLString::transcode("minrange");
        XMLCh *metricmaxrange = XMLString::transcode("maxrange");
        XMLCh *metricstddevrange = XMLString::transcode("stddevrange");
        XMLCh *metricvariancerange = XMLString::transcode("variancerange");
        XMLCh *metricabsdeviationrange = XMLString::transcode("absdeviationrange");
        XMLCh *metriccoefficientofvariationrange = XMLString::transcode("coefficientofvariationrange");
        XMLCh *metricpercentilerange = XMLString::transcode("percentilerange");
        XMLCh *metricskewnessrange = XMLString::transcode("skewnessrange");
        XMLCh *metricpersonmoderange = XMLString::transcode("personmoderange");
        XMLCh *metricpersonmedianrange = XMLString::transcode("personmedianrange");
        XMLCh *metrickurtosisrange = XMLString::transcode("kurtosisrange");
        XMLCh *metricreturnsaboverangemetric = XMLString::transcode("returnsaboverangemetric");
        XMLCh *metricreturnsbelowrangemetric = XMLString::transcode("returnsbelowrangemetric");
        XMLCh *metricnumreturnswidth = XMLString::transcode("numreturnswidth");
        XMLCh *metricsumwidth = XMLString::transcode("sumwidth");
        XMLCh *metricmeanwidth = XMLString::transcode("meanwidth");
        XMLCh *metricmedianwidth = XMLString::transcode("medianwidth");
        XMLCh *metricmodewidth = XMLString::transcode("modewidth");
        XMLCh *metricminwidth = XMLString::transcode("minwidth");
        XMLCh *metricmaxwidth = XMLString::transcode("maxwidth");
        XMLCh *metricstddevwidth = XMLString::transcode("stddevwidth");
        XMLCh *metricvariancewidth = XMLString::transcode("variancewidth");
        XMLCh *metricabsdeviationwidth = XMLString::transcode("absdeviationwidth");
        XMLCh *metriccoefficientofvariationwidth = XMLString::transcode("coefficientofvariationwidth");
        XMLCh *metricpercentilewidth = XMLString::transcode("percentilewidth");
        XMLCh *metricskewnesswidth = XMLString::transcode("skewnesswidth");
        XMLCh *metricpersonmodewidth = XMLString::transcode("personmodewidth");
        XMLCh *metricpersonmedianwidth = XMLString::transcode("personmedianwidth");
        XMLCh *metrickurtosiswidth = XMLString::transcode("kurtosiswidth");
        XMLCh *metricreturnsabovewidthmetric = XMLString::transcode("returnsabovewidthmetric");
        XMLCh *metricreturnsbelowwidthmetric = XMLString::transcode("returnsbelowwidthmetric");
        XMLCh *metricNameXMLStr = XMLString::transcode("metric");
        XMLCh *metricReturnXMLStr = XMLString::transcode("return");
        XMLCh *metricClassXMLStr = XMLString::transcode("class");
        XMLCh *metricMinNumReturnsXMLStr = XMLString::transcode("minNumReturns");
        XMLCh *metricUpThresholdXMLStr = XMLString::transcode("upthreshold");
        XMLCh *metricLowThresholdXMLStr = XMLString::transcode("lowthreshold");
        XMLCh *heightUpThresholdXMLStr = XMLString::transcode("heightup");
        XMLCh *heightLowThresholdXMLStr = XMLString::transcode("heightlow");
        XMLCh *allXMLStr = XMLString::transcode("All");
        XMLCh *notFirstXMLStr = XMLString::transcode("NotFirst");
        XMLCh *firstXMLStr = XMLString::transcode("First");
        XMLCh *lastXMLStr = XMLString::transcode("Last");
        XMLCh *firstLastXMLStr = XMLString::transcode("FirstLast");
        XMLCh *notGrdXMLStr = XMLString::transcode("NotGrd");
        XMLCh *vegXMLStr = XMLString::transcode("Veg");
        XMLCh *grdXMLStr = XMLString::transcode("Grd");
        
        SPDTextFileUtilities textUtils;
        const char *nanVal = "NaN";
        SPDMetric *metric = NULL;
        XMLCh *metricNameStr = NULL;
        
        boost::uint_fast32_t returnID = 0;
        boost::uint_fast32_t classID = 0;
        boost::uint_fast32_t minNumReturns = 0;
        double upThreshold = 0;
        double lowThreshold = 0;
        double heightUpThreshold = 0;
        double heightLowThreshold = 0;
        
        try
        {
            if(metricElement->hasAttribute(metricNameXMLStr))
            {
                char *charValue = XMLString::transcode(metricElement->getAttribute(metricNameXMLStr));
                metricNameStr = XMLString::transcode(charValue);
                XMLString::release(&charValue);
            }
            else
            {
                throw SPDProcessingException("The \'metric\' attribute was not provided for the metric element.");
            }
            
            if(metricElement->hasAttribute(metricReturnXMLStr))
            {
                if(XMLString::equals(allXMLStr, metricElement->getAttribute(metricReturnXMLStr)))
                {
                    returnID = SPD_ALL_RETURNS;
                }
                else if(XMLString::equals(notFirstXMLStr, metricElement->getAttribute(metricReturnXMLStr)))
                {
                    returnID = SPD_NOTFIRST_RETURNS;
                }
                else if(XMLString::equals(firstXMLStr, metricElement->getAttribute(metricReturnXMLStr)))
                {
                    returnID = SPD_FIRST_RETURNS;
                }
                else if(XMLString::equals(lastXMLStr, metricElement->getAttribute(metricReturnXMLStr)))
                {
                    returnID = SPD_LAST_RETURNS;
                }
                else if(XMLString::equals(firstLastXMLStr, metricElement->getAttribute(metricReturnXMLStr)))
                {
                    returnID = SPD_FIRSTLAST_RETURNS;
                }
                else
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(metricReturnXMLStr));
                    returnID = textUtils.strto32bitUInt(string(charValue));
                    XMLString::release(&charValue);
                }
            }
            else
            {
                returnID = SPD_ALL_RETURNS;
            }
            
            if(metricElement->hasAttribute(metricClassXMLStr))
            {
                if(XMLString::equals(allXMLStr, metricElement->getAttribute(metricClassXMLStr)))
                {
                    classID = SPD_ALL_CLASSES;
                }
                else if(XMLString::equals(notGrdXMLStr, metricElement->getAttribute(metricClassXMLStr)))
                {
                    classID = SPD_NOT_GROUND;
                }
                else if(XMLString::equals(grdXMLStr, metricElement->getAttribute(metricClassXMLStr)))
                {
                    classID = SPD_GROUND;
                }
                else if(XMLString::equals(vegXMLStr, metricElement->getAttribute(metricClassXMLStr)))
                {
                    classID = SPD_VEGETATION;
                }
                else
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(metricClassXMLStr));
                    classID = textUtils.strto32bitUInt(string(charValue));
                    XMLString::release(&charValue);
                }
            }
            else
            {
                classID = SPD_ALL_CLASSES;
            }
            
            if(metricElement->hasAttribute(metricMinNumReturnsXMLStr))
            {
                char *charValue = XMLString::transcode(metricElement->getAttribute(metricMinNumReturnsXMLStr));
                minNumReturns = textUtils.strto32bitUInt(string(charValue));
                XMLString::release(&charValue);
            }
            else
            {
                minNumReturns = 0;
            }
            
            if(metricElement->hasAttribute(metricUpThresholdXMLStr))
            {
                char *charValue = XMLString::transcode(metricElement->getAttribute(metricUpThresholdXMLStr));
                upThreshold = textUtils.strtodouble(string(charValue));
                XMLString::release(&charValue);
            }
            else
            {
                upThreshold = nan(nanVal);
            }
            
            if(metricElement->hasAttribute(metricLowThresholdXMLStr))
            {
                char *charValue = XMLString::transcode(metricElement->getAttribute(metricLowThresholdXMLStr));
                lowThreshold = textUtils.strtodouble(string(charValue));
                XMLString::release(&charValue);
            }
            else
            {
                lowThreshold = nan(nanVal);
            }
            
            if(metricElement->hasAttribute(heightUpThresholdXMLStr))
            {
                char *charValue = XMLString::transcode(metricElement->getAttribute(heightUpThresholdXMLStr));
                heightUpThreshold = textUtils.strtodouble(string(charValue));
                XMLString::release(&charValue);
            }
            else
            {
                heightUpThreshold = nan(nanVal);
            }
            
            if(metricElement->hasAttribute(heightLowThresholdXMLStr))
            {
                char *charValue = XMLString::transcode(metricElement->getAttribute(heightLowThresholdXMLStr));
                heightLowThreshold = textUtils.strtodouble(string(charValue));
                XMLString::release(&charValue);
            }
            else
            {
                heightLowThreshold = nan(nanVal);
            }
            
            
            if(XMLString::equals(metricNameStr, metricadd))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics < 2)
                {
                    cout << "Number of metrics = " << numMetrics << endl;
                    throw SPDProcessingException("The \'add\' metric needs at least two child metrics.");
                }
                
                vector<SPDMetric*> *metrics = new vector<SPDMetric*>();
                
                DOMElement *tmpMetricElement = metricElement->getFirstElementChild();
                for(boost::uint_fast32_t i = 0; i < numMetrics; ++i)
                {
                    // Retrieve Metric and add to list.
                    metrics->push_back(this->createMetric(tmpMetricElement));
                    
                    // Move on to next metric
                    tmpMetricElement = tmpMetricElement->getNextElementSibling();
                }
                
                metric = new SPDMetricAdd(metrics);
            }
            else if(XMLString::equals(metricNameStr, metricminus))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 2)
                {
                    throw SPDProcessingException("The \'minus\' metric needs two child metrics.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                metricElementIn = metricElementIn->getNextElementSibling();
                SPDMetric *metric2 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricMinus(metric1, metric2);
            }
            else if(XMLString::equals(metricNameStr, metricmultiply))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics < 2)
                {
                    throw SPDProcessingException("The \'multiply\' metric needs at least two child metrics.");
                }
                
                vector<SPDMetric*> *metrics = new vector<SPDMetric*>();
                
                DOMElement *tmpMetricElement = metricElement->getFirstElementChild();
                for(boost::uint_fast32_t i = 0; i < numMetrics; ++i)
                {
                    // Retrieve Metric and add to list.
                    metrics->push_back(this->createMetric(tmpMetricElement));
                    
                    // Move on to next metric
                    tmpMetricElement = tmpMetricElement->getNextElementSibling();
                }
                
                metric = new SPDMetricMultiply(metrics);
            }
            else if(XMLString::equals(metricNameStr, metricdivide))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 2)
                {
                    throw SPDProcessingException("The \'divide\' metric needs two child metrics.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                metricElementIn = metricElementIn->getNextElementSibling();
                SPDMetric *metric2 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricDivide(metric1, metric2);
            }
            else if(XMLString::equals(metricNameStr, metricpow))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 2)
                {
                    throw SPDProcessingException("The \'pow\' metric needs two child metrics.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                metricElementIn = metricElementIn->getNextElementSibling();
                SPDMetric *metric2 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricPow(metric1, metric2);
            }
            else if(XMLString::equals(metricNameStr, metricabs))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'abs\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricAbs(metric1);
            }
            else if(XMLString::equals(metricNameStr, metricsqrt))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'sqrt\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricSqrt(metric1);
            }
            else if(XMLString::equals(metricNameStr, metricsine))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'sine\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricSine(metric1);
            }
            else if(XMLString::equals(metricNameStr, metricabs))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'abs\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricAbs(metric1);
            }
            else if(XMLString::equals(metricNameStr, metriccosine))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'cosine\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricCosine(metric1);
            }
            else if(XMLString::equals(metricNameStr, metrictangent))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'tangent\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricTangent(metric1);
            }
            else if(XMLString::equals(metricNameStr, metricinvsine))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'invsine\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricInvSine(metric1);
            }
            else if(XMLString::equals(metricNameStr, metricinvcos))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'invcos\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricInvCos(metric1);
            }
            else if(XMLString::equals(metricNameStr, metricinvtan))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'invtan\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricInvTan(metric1);
            }
            else if(XMLString::equals(metricNameStr, metriclog10))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'log10\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricLog10(metric1);
            }
            else if(XMLString::equals(metricNameStr, metricln))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'ln\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricLn(metric1);
            }
            else if(XMLString::equals(metricNameStr, metricexp))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'exp\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricExp(metric1);
            }
            else if(XMLString::equals(metricNameStr, metricpercentage))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 2)
                {
                    throw SPDProcessingException("The \'percentage\' metric needs two child metrics.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                metricElementIn = metricElementIn->getNextElementSibling();
                SPDMetric *metric2 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricPercentage(metric1, metric2);
            }
            else if(XMLString::equals(metricNameStr, metricaddconst))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'add const\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                double constVal = 0;
                XMLCh *constXMLStr = XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                XMLString::release(&constXMLStr);
                
                metric = new SPDMetricAddConst(metric1, constVal);
            }
            else if(XMLString::equals(metricNameStr, metricminusconstfrom))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'minus const from\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                double constVal = 0;
                XMLCh *constXMLStr = XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                XMLString::release(&constXMLStr);
                
                metric = new SPDMetricMinusConstFrom(metric1, constVal);
            }
            else if(XMLString::equals(metricNameStr, metricminusfromconst))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'minus from const\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                double constVal = 0;
                XMLCh *constXMLStr = XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                XMLString::release(&constXMLStr);
                
                metric = new SPDMetricMinusFromConst(metric1, constVal);
            }
            else if(XMLString::equals(metricNameStr, metricmultiplyconst))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'multiply const\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                double constVal = 0;
                XMLCh *constXMLStr = XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                XMLString::release(&constXMLStr);
                
                metric = new SPDMetricMultiplyConst(metric1, constVal);
            }
            else if(XMLString::equals(metricNameStr, metricdividebyconst))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'divide by const\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                double constVal = 0;
                XMLCh *constXMLStr = XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                XMLString::release(&constXMLStr);
                
                metric = new SPDMetricDivideByConst(metric1, constVal);
            }
            else if(XMLString::equals(metricNameStr, metricdivideconstby))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'divide const by\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                double constVal = 0;
                XMLCh *constXMLStr = XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                XMLString::release(&constXMLStr);
                
                metric = new SPDMetricDivideConstBy(metric1, constVal);
            }
            else if(XMLString::equals(metricNameStr, metricpowmetricconst))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'pow metric const\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                double constVal = 0;
                XMLCh *constXMLStr = XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                XMLString::release(&constXMLStr);
                
                metric = new SPDMetricPowMetricConst(metric1, constVal);
            }
            else if(XMLString::equals(metricNameStr, metricpowconstmetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'pow const metric\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                double constVal = 0;
                XMLCh *constXMLStr = XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                XMLString::release(&constXMLStr);
                
                metric = new SPDMetricPowConstMetric(metric1, constVal);
            }
            else if(XMLString::equals(metricNameStr, metricnumpulses))
            {
                metric = new SPDMetricCalcNumPulses(minNumReturns);
            }
            else if(XMLString::equals(metricNameStr, metriccanopycover))
            {
                float resolution = 0;
                XMLCh *resolutionXMLStr = XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtofloat(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                XMLString::release(&resolutionXMLStr);

                float radius = 0;
                XMLCh *radiusXMLStr = XMLString::transcode("radius");
                if(metricElement->hasAttribute(radiusXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(radiusXMLStr));
                    radius = textUtils.strtofloat(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'radius\' value has not been provided.");
                }
                XMLString::release(&radiusXMLStr);
                
                metric = new SPDMetricCalcCanopyCover(resolution, radius, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metriccanopycoverpercent))
            {
                float resolution = 0;
                XMLCh *resolutionXMLStr = XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtofloat(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                XMLString::release(&resolutionXMLStr);
                
                float radius = 0;
                XMLCh *radiusXMLStr = XMLString::transcode("radius");
                if(metricElement->hasAttribute(radiusXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(radiusXMLStr));
                    radius = textUtils.strtofloat(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'radius\' value has not been provided.");
                }
                XMLString::release(&radiusXMLStr);
                
                metric = new SPDMetricCalcCanopyCoverPercent(resolution, radius, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricleeopenness))
            {
                float vres = 0;
                XMLCh *vresolutionXMLStr = XMLString::transcode("vres");
                if(metricElement->hasAttribute(vresolutionXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(vresolutionXMLStr));
                    vres = textUtils.strtofloat(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'vres\' value has not been provided.");
                }
                XMLString::release(&vresolutionXMLStr);
                
                metric = new SPDMetricCalcLeeOpennessHeight(vres, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }            
            else if(XMLString::equals(metricNameStr, metricnumreturnsheight))
            {
                metric = new SPDMetricCalcNumReturnsHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricsumheight))
            {
                metric = new SPDMetricCalcSumHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmeanheight))
            {
                metric = new SPDMetricCalcMeanHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmedianheight))
            {
                metric = new SPDMetricCalcMedianHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmodeheight))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                XMLString::release(&resolutionXMLStr);
                
                metric = new SPDMetricCalcModeHeight(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricminheight))
            {
                metric = new SPDMetricCalcMinHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmaxheight))
            {
                metric = new SPDMetricCalcMaxHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmaxdominant))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                XMLString::release(&resolutionXMLStr);
                
                metric = new SPDMetricCalcDominantHeight(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricstddevheight))
            {
                metric = new SPDMetricCalcStdDevHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricvarianceheight))
            {
                metric = new SPDMetricCalcVarianceHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricabsdeviationheight))
            {
                metric = new SPDMetricCalcAbsDeviationHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metriccoefficientofvariationheight))
            {
                metric = new SPDMetricCalcCoefficientOfVariationHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricpercentileheight))
            {
                boost::uint_fast32_t percentileVal = 0;
                XMLCh *percentileXMLStr = XMLString::transcode("percentile");
                if(metricElement->hasAttribute(percentileXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(percentileXMLStr));
                    percentileVal = textUtils.strto32bitUInt(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'percentile\' value has not been provided.");
                }
                XMLString::release(&percentileXMLStr);
                metric = new SPDMetricCalcPercentileHeight(percentileVal,returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricskewnessheight))
            {
                metric = new SPDMetricCalcSkewnessHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricpersonmodeheight))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcPersonModeSkewnessHeight(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricpersonmedianheight))
            {
                metric = new SPDMetricCalcPersonMedianSkewnessHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metrickurtosisheight))
            {
                metric = new SPDMetricCalcKurtosisHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricreturnsaboveheightmetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns above height metric\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricCalcNumReturnsAboveMetricHeight(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricreturnsbelowheightmetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns below height metric\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricCalcNumReturnsBelowMetricHeight(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricnumreturnsz))
            {
                metric = new SPDMetricCalcNumReturnsZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricsumz))
            {
                metric = new SPDMetricCalcSumZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmeanz))
            {
                metric = new SPDMetricCalcMeanZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmedianz))
            {
                metric = new SPDMetricCalcMedianZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmodez))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                XMLString::release(&resolutionXMLStr);
                
                metric = new SPDMetricCalcModeZ(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricminz))
            {
                metric = new SPDMetricCalcMinZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmaxz))
            {
                metric = new SPDMetricCalcMaxZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricstddevz))
            {
                metric = new SPDMetricCalcStdDevZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricvariancez))
            {
                metric = new SPDMetricCalcVarianceZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricabsdeviationz))
            {
                metric = new SPDMetricCalcAbsDeviationZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metriccoefficientofvariationz))
            {
                metric = new SPDMetricCalcCoefficientOfVariationZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricpercentilez))
            {
                boost::uint_fast32_t percentileVal = 0;
                XMLCh *percentileXMLStr = XMLString::transcode("percentile");
                if(metricElement->hasAttribute(percentileXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(percentileXMLStr));
                    percentileVal = textUtils.strto32bitUInt(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'percentile\' value has not been provided.");
                }
                XMLString::release(&percentileXMLStr);
                metric = new SPDMetricCalcPercentileZ(percentileVal,returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricskewnessz))
            {
                metric = new SPDMetricCalcSkewnessZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricpersonmodez))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcPersonModeSkewnessZ(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricpersonmedianz))
            {
                metric = new SPDMetricCalcPersonMedianSkewnessZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metrickurtosisz))
            {
                metric = new SPDMetricCalcKurtosisZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricreturnsabovezmetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns above z metric\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricCalcNumReturnsAboveMetricZ(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricreturnsbelowzmetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns below z metric\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricCalcNumReturnsBelowMetricZ(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricnumreturnsamplitude))
            {
                metric = new SPDMetricCalcNumReturnsAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricsumamplitude))
            {
                metric = new SPDMetricCalcSumAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmeanamplitude))
            {
                metric = new SPDMetricCalcMeanAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmedianamplitude))
            {
                metric = new SPDMetricCalcMedianAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmodeamplitude))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcModeAmplitude(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricminamplitude))
            {
                metric = new SPDMetricCalcMinAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmaxamplitude))
            {
                metric = new SPDMetricCalcMaxAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricstddevamplitude))
            {
                metric = new SPDMetricCalcStdDevAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricvarianceamplitude))
            {
                metric = new SPDMetricCalcVarianceAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricabsdeviationamplitude))
            {
                metric = new SPDMetricCalcAbsDeviationAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metriccoefficientofvariationamplitude))
            {
                metric = new SPDMetricCalcCoefficientOfVariationAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricpercentileamplitude))
            {
                boost::uint_fast32_t percentileVal = 0;
                XMLCh *percentileXMLStr = XMLString::transcode("percentile");
                if(metricElement->hasAttribute(percentileXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(percentileXMLStr));
                    percentileVal = textUtils.strto32bitUInt(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'percentile\' value has not been provided.");
                }
                XMLString::release(&percentileXMLStr);
                metric = new SPDMetricCalcPercentileAmplitude(percentileVal,returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricskewnessamplitude))
            {
                metric = new SPDMetricCalcSkewnessAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricpersonmodeamplitude))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcPersonModeSkewnessAmplitude(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricpersonmedianamplitude))
            {
                metric = new SPDMetricCalcPersonMedianSkewnessAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metrickurtosisamplitude))
            {
                metric = new SPDMetricCalcKurtosisAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricreturnsaboveamplitudemetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns above amplitude metric\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricCalcNumReturnsAboveMetricAmplitude(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricreturnsbelowamplitudemetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns below amplitude metric\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricCalcNumReturnsBelowMetricAmplitude(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricnumreturnsrange))
            {
                metric = new SPDMetricCalcNumReturnsRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricsumrange))
            {
                metric = new SPDMetricCalcSumRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmeanrange))
            {
                metric = new SPDMetricCalcMeanRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmedianrange))
            {
                metric = new SPDMetricCalcMedianRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmoderange))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcModeRange(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricminrange))
            {
                metric = new SPDMetricCalcMinRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmaxrange))
            {
                metric = new SPDMetricCalcMaxRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricstddevrange))
            {
                metric = new SPDMetricCalcStdDevRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricvariancerange))
            {
                metric = new SPDMetricCalcVarianceRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricabsdeviationrange))
            {
                metric = new SPDMetricCalcAbsDeviationRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metriccoefficientofvariationrange))
            {
                metric = new SPDMetricCalcCoefficientOfVariationRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricpercentilerange))
            {
                boost::uint_fast32_t percentileVal = 0;
                XMLCh *percentileXMLStr = XMLString::transcode("percentile");
                if(metricElement->hasAttribute(percentileXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(percentileXMLStr));
                    percentileVal = textUtils.strto32bitUInt(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'percentile\' value has not been provided.");
                }
                XMLString::release(&percentileXMLStr);
                metric = new SPDMetricCalcPercentileRange(percentileVal,returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricskewnessrange))
            {
                metric = new SPDMetricCalcSkewnessRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricpersonmoderange))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcPersonModeSkewnessRange(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricpersonmedianrange))
            {
                metric = new SPDMetricCalcPersonMedianSkewnessRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metrickurtosisrange))
            {
                metric = new SPDMetricCalcKurtosisRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricreturnsaboverangemetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns above range metric\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricCalcNumReturnsAboveMetricRange(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricreturnsbelowrangemetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns below range metric\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricCalcNumReturnsBelowMetricRange(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricnumreturnswidth))
            {
                metric = new SPDMetricCalcNumReturnsWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricsumwidth))
            {
                metric = new SPDMetricCalcSumWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmeanwidth))
            {
                metric = new SPDMetricCalcMeanWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmedianwidth))
            {
                metric = new SPDMetricCalcMedianWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmodewidth))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcModeWidth(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricminwidth))
            {
                metric = new SPDMetricCalcMinWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricmaxwidth))
            {
                metric = new SPDMetricCalcMaxWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricstddevwidth))
            {
                metric = new SPDMetricCalcStdDevWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricvariancewidth))
            {
                metric = new SPDMetricCalcVarianceWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricabsdeviationwidth))
            {
                metric = new SPDMetricCalcAbsDeviationWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metriccoefficientofvariationwidth))
            {
                metric = new SPDMetricCalcCoefficientOfVariationWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricpercentilewidth))
            {
                boost::uint_fast32_t percentileVal = 0;
                XMLCh *percentileXMLStr = XMLString::transcode("percentile");
                if(metricElement->hasAttribute(percentileXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(percentileXMLStr));
                    percentileVal = textUtils.strto32bitUInt(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'percentile\' value has not been provided.");
                }
                XMLString::release(&percentileXMLStr);
                metric = new SPDMetricCalcPercentileWidth(percentileVal,returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricskewnesswidth))
            {
                metric = new SPDMetricCalcSkewnessWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricpersonmodewidth))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(string(charValue));
                    XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcPersonModeSkewnessWidth(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricpersonmedianwidth))
            {
                metric = new SPDMetricCalcPersonMedianSkewnessWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metrickurtosiswidth))
            {
                metric = new SPDMetricCalcKurtosisWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricreturnsabovewidthmetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns above width metric\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricCalcNumReturnsAboveMetricWidth(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else if(XMLString::equals(metricNameStr, metricreturnsbelowwidthmetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns below width metric\' metric needs one child metric.");
                }
                
                DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                
                metric = new SPDMetricCalcNumReturnsBelowMetricWidth(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
            }
            else
            {
                string message = "Metric \'" + string(XMLString::transcode(metricNameStr)) + "\' has not been recognised.";
                throw SPDProcessingException(message);
            }
        }
        catch (const XMLException& e) 
		{
			char *message = XMLString::transcode(e.getMessage());
			string outMessage =  string("XMLException : ") + string(message);
			throw SPDProcessingException(outMessage.c_str());
		}
		catch (const DOMException& e) 
		{
			char *message = XMLString::transcode(e.getMessage());
			string outMessage =  string("DOMException : ") + string(message);
			throw SPDProcessingException(outMessage.c_str());
		}
		catch(SPDProcessingException &e)
		{
			throw e;
		}
        
        XMLString::release(&metricadd);
        XMLString::release(&metricminus);
        XMLString::release(&metricmultiply);
        XMLString::release(&metricdivide);
        XMLString::release(&metricpow);
        XMLString::release(&metricabs);
        XMLString::release(&metricsqrt);
        XMLString::release(&metricsine);
        XMLString::release(&metriccosine);
        XMLString::release(&metrictangent);
        XMLString::release(&metricinvsine);
        XMLString::release(&metricinvcos);
        XMLString::release(&metricinvtan);
        XMLString::release(&metriclog10);
        XMLString::release(&metricln);
        XMLString::release(&metricexp);
        XMLString::release(&metricpercentage);
        XMLString::release(&metricaddconst);
        XMLString::release(&metricminusconstfrom);
        XMLString::release(&metricminusfromconst);
        XMLString::release(&metricmultiplyconst);
        XMLString::release(&metricdividebyconst);
        XMLString::release(&metricdivideconstby);
        XMLString::release(&metricpowmetricconst);
        XMLString::release(&metricpowconstmetric);
        XMLString::release(&metricnumpulses);
        XMLString::release(&metriccanopycover);
        XMLString::release(&metriccanopycoverpercent);
        XMLString::release(&metricleeopenness);
        XMLString::release(&metricnumreturnsheight);
        XMLString::release(&metricmeanheight);
        XMLString::release(&metricsumheight);
        XMLString::release(&metricmedianheight);
        XMLString::release(&metricmodeheight);
        XMLString::release(&metricminheight);
        XMLString::release(&metricmaxheight);
        XMLString::release(&metricmaxdominant);
        XMLString::release(&metricstddevheight);
        XMLString::release(&metricvarianceheight);
        XMLString::release(&metricabsdeviationheight);
        XMLString::release(&metriccoefficientofvariationheight);
        XMLString::release(&metricpercentileheight);
        XMLString::release(&metricskewnessheight);
        XMLString::release(&metricpersonmodeheight);
        XMLString::release(&metricpersonmedianheight);
        XMLString::release(&metrickurtosisheight);
        XMLString::release(&metricreturnsaboveheightmetric);
        XMLString::release(&metricreturnsbelowheightmetric);
        XMLString::release(&metricnumreturnsz);
        XMLString::release(&metricmeanz);
        XMLString::release(&metricsumz);
        XMLString::release(&metricmedianz);
        XMLString::release(&metricmodez);
        XMLString::release(&metricminz);
        XMLString::release(&metricmaxz);
        XMLString::release(&metricstddevz);
        XMLString::release(&metricvariancez);
        XMLString::release(&metricabsdeviationz);
        XMLString::release(&metriccoefficientofvariationz);
        XMLString::release(&metricpercentilez);
        XMLString::release(&metricskewnessz);
        XMLString::release(&metricpersonmodez);
        XMLString::release(&metricpersonmedianz);
        XMLString::release(&metrickurtosisz);
        XMLString::release(&metricreturnsabovezmetric);
        XMLString::release(&metricreturnsbelowzmetric);
        XMLString::release(&metricnumreturnsamplitude);
        XMLString::release(&metricmeanamplitude);
        XMLString::release(&metricsumamplitude);
        XMLString::release(&metricmedianamplitude);
        XMLString::release(&metricmodeamplitude);
        XMLString::release(&metricminamplitude);
        XMLString::release(&metricmaxamplitude);
        XMLString::release(&metricstddevamplitude);
        XMLString::release(&metricvarianceamplitude);
        XMLString::release(&metricabsdeviationamplitude);
        XMLString::release(&metriccoefficientofvariationamplitude);
        XMLString::release(&metricpercentileamplitude);
        XMLString::release(&metricskewnessamplitude);
        XMLString::release(&metricpersonmodeamplitude);
        XMLString::release(&metricpersonmedianamplitude);
        XMLString::release(&metrickurtosisamplitude);
        XMLString::release(&metricreturnsaboveamplitudemetric);
        XMLString::release(&metricreturnsbelowamplitudemetric);
        XMLString::release(&metricnumreturnsrange);
        XMLString::release(&metricmeanrange);
        XMLString::release(&metricsumrange);
        XMLString::release(&metricmedianrange);
        XMLString::release(&metricmoderange);
        XMLString::release(&metricminrange);
        XMLString::release(&metricmaxrange);
        XMLString::release(&metricstddevrange);
        XMLString::release(&metricvariancerange);
        XMLString::release(&metricabsdeviationrange);
        XMLString::release(&metriccoefficientofvariationrange);
        XMLString::release(&metricpercentilerange);
        XMLString::release(&metricskewnessrange);
        XMLString::release(&metricpersonmoderange);
        XMLString::release(&metricpersonmedianrange);
        XMLString::release(&metrickurtosisrange);
        XMLString::release(&metricreturnsaboverangemetric);
        XMLString::release(&metricreturnsbelowrangemetric);
		XMLString::release(&metricnumreturnswidth);
        XMLString::release(&metricmeanwidth);
        XMLString::release(&metricsumwidth);
        XMLString::release(&metricmedianwidth);
        XMLString::release(&metricmodewidth);
        XMLString::release(&metricminwidth);
        XMLString::release(&metricmaxwidth);
        XMLString::release(&metricstddevwidth);
        XMLString::release(&metricvariancewidth);
        XMLString::release(&metricabsdeviationwidth);
        XMLString::release(&metriccoefficientofvariationwidth);
        XMLString::release(&metricpercentilewidth);
        XMLString::release(&metricskewnesswidth);
        XMLString::release(&metricpersonmodewidth);
        XMLString::release(&metricpersonmedianwidth);
        XMLString::release(&metrickurtosiswidth);
        XMLString::release(&metricreturnsabovewidthmetric);
        XMLString::release(&metricreturnsbelowwidthmetric);
        XMLString::release(&metricNameXMLStr);
        XMLString::release(&metricReturnXMLStr);
        XMLString::release(&metricClassXMLStr);
        XMLString::release(&metricMinNumReturnsXMLStr);
        XMLString::release(&metricUpThresholdXMLStr);
        XMLString::release(&metricLowThresholdXMLStr);
        XMLString::release(&heightUpThresholdXMLStr);
        XMLString::release(&heightLowThresholdXMLStr);
        XMLString::release(&allXMLStr);
        XMLString::release(&notFirstXMLStr);
        XMLString::release(&firstXMLStr);
        XMLString::release(&lastXMLStr);
        XMLString::release(&firstLastXMLStr);
        XMLString::release(&notGrdXMLStr);
        XMLString::release(&vegXMLStr);
        XMLString::release(&grdXMLStr);
        
        return metric;
    }

    SPDCalcMetrics::~SPDCalcMetrics()
    {
        
    }
    
    
    
    
    SPDCalcImageMetrics::SPDCalcImageMetrics(vector<SPDMetric*> *metrics, vector<string> *fieldNames)
    {
        this->metrics = metrics;
        this->fieldNames = fieldNames;
    }
        
    void SPDCalcImageMetrics::processDataColumnImage(SPDFile *inSPDFile, vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
    {
        if(numImgBands != metrics->size())
        {
            throw SPDProcessingException("The number of image bands needs to be the same as the number of metrics.");
        }
        
        try
        {
            double tlX = cenPts->x - (binSize/2);
            double tlY = cenPts->y + (binSize/2);
            double brX = cenPts->x + (binSize/2);
            double brY = cenPts->y - (binSize/2);
            
            double xCoords[5];
            double yCoords[5];
            xCoords[0] = tlX;
            yCoords[0] = tlY;
            xCoords[1] = brX;
            yCoords[1] = tlY;
            xCoords[2] = brX;
            yCoords[2] = brY;
            xCoords[3] = tlX;
            yCoords[3] = brY;
            xCoords[4] = tlX;
            yCoords[4] = tlY;
            
            OGRLinearRing *polyRing = new OGRLinearRing();
            polyRing->setPoints(5, xCoords, yCoords);
            OGRPolygon *geom = new OGRPolygon();
            geom->addRingDirectly(polyRing);
            
            boost::uint_fast32_t idx = 0;
            for(vector<SPDMetric*>::iterator iterMetrics = metrics->begin(); iterMetrics != metrics->end(); ++iterMetrics)
            {
                imageData[idx++] = (*iterMetrics)->calcValue(pulses, inSPDFile, geom);
            }
            
            delete geom;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
    }

    vector<string> SPDCalcImageMetrics::getImageBandDescriptions() throw(SPDProcessingException)
    {
        if(metrics->size() != fieldNames->size())
        {
            throw SPDProcessingException("The number of metrics and fieldnames needs to be the same.");
        }
        cout << "Executing for metrics: \n";
        vector<string> bandNames;
        for(vector<string>::iterator iterNames = fieldNames->begin(); iterNames != fieldNames->end(); ++iterNames)
        {
            bandNames.push_back(*iterNames);
            cout << *iterNames << endl;
        }
        return bandNames;
    }
    
    void SPDCalcImageMetrics::setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
    {
        // Nothing to do here.
    }
        
    SPDCalcImageMetrics::~SPDCalcImageMetrics()
    {
        
    }
    
    
    

      

    SPDCalcPolyMetrics::SPDCalcPolyMetrics(vector<SPDMetric*> *metrics, vector<string> *fieldNames)
    {
        this->metrics = metrics;
        this->fieldNames = fieldNames;
    }
		
    void SPDCalcPolyMetrics::processFeature(OGRFeature *inFeature, OGRFeature *outFeature, boost::uint_fast64_t fid, vector<SPDPulse*> *pulses, SPDFile *spdFile) throw(SPDProcessingException)
    {
        vector<SPDMetric*>::iterator iterMetrics = metrics->begin();
        vector<string>::iterator iterNames = fieldNames->begin();
        
        OGRFeatureDefn *outFeatureDefn = outFeature->GetDefnRef();
        
        OGRGeometry *geometry = inFeature->GetGeometryRef();
        
        boost::uint_fast32_t numMetrics = metrics->size();
        double outVal = 0;
        for(boost::uint_fast32_t i = 0; i < numMetrics; ++i)
        {
            outVal = (*iterMetrics)->calcValue(pulses, spdFile, geometry);            
            outFeature->SetField(outFeatureDefn->GetFieldIndex((*iterNames).c_str()), outVal);
            
            ++iterMetrics;
            ++iterNames;
        }
    }
    
    void SPDCalcPolyMetrics::processFeature(OGRFeature *inFeature, ofstream *outASCIIFile, boost::uint_fast64_t fid, vector<SPDPulse*> *pulses, SPDFile *spdFile) throw(SPDProcessingException)
    {
        (*outASCIIFile) << fid;
        
        OGRGeometry *geometry = inFeature->GetGeometryRef();
        
        double outVal = 0;
        for(vector<SPDMetric*>::iterator iterMetrics = metrics->begin(); iterMetrics != metrics->end(); ++iterMetrics)
        {
            outVal = (*iterMetrics)->calcValue(pulses, spdFile, geometry);            
            (*outASCIIFile) << "," << outVal;
        }
        
        (*outASCIIFile) << endl;
    }
    
    void SPDCalcPolyMetrics::createOutputLayerDefinition(OGRLayer *outputLayer, OGRFeatureDefn *inFeatureDefn) throw(SPDProcessingException)
    {
        if(metrics->size() != fieldNames->size())
        {
            throw SPDProcessingException("The number of metrics and fieldnames needs to be the same.");
        }
        
        for(vector<string>::iterator iterNames = fieldNames->begin(); iterNames != fieldNames->end(); ++iterNames)
        {
            OGRFieldDefn shpField((*iterNames).c_str(), OFTReal);
            shpField.SetPrecision(10);
            if( outputLayer->CreateField( &shpField ) != OGRERR_NONE )
            {
                string message = string("Creating shapefile field ") + *iterNames + string(" has failed");
                throw SPDProcessingException(message);
            }
        }
    }
    
    void SPDCalcPolyMetrics::writeASCIIHeader(ofstream *outASCIIFile) throw(SPDProcessingException)
    {
        (*outASCIIFile) << "FID";
        for(vector<string>::iterator iterNames = fieldNames->begin(); iterNames != fieldNames->end(); ++iterNames)
        {
            (*outASCIIFile) << "," << (*iterNames);
        }
        (*outASCIIFile) << endl;
    }
    
    SPDCalcPolyMetrics::~SPDCalcPolyMetrics()
    {
        
    }
    

}


