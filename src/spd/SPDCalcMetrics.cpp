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

    void SPDCalcMetrics::calcMetricToImage(std::string inXMLFilePath, std::string inputSPDFile, std::string outputImage, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize, float processingResolution, std::string gdalFormat) throw (SPDProcessingException)
    {
        try
        {
            std::vector<SPDMetric*> *metrics = new std::vector<SPDMetric*>();
            std::vector<std::string> *fieldNames = new std::vector<std::string>();
            this->parseMetricsXML(inXMLFilePath, metrics, fieldNames);

            if(metrics->size() != fieldNames->size())
            {
                throw SPDProcessingException("The number of metrics and fieldnames needs to be the same.");
            }

            std::cout << metrics->size() << " metrics where found." << std::endl;


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

    void SPDCalcMetrics::calcMetricToVectorShp(std::string inXMLFilePath, std::string inputSPDFile, std::string inputVectorShp, std::string outputVectorShp, bool deleteOutShp, bool copyAttributes) throw (SPDProcessingException)
    {
        try
        {
            std::vector<SPDMetric*> *metrics = new std::vector<SPDMetric*>();
            std::vector<std::string> *fieldNames = new std::vector<std::string>();
            this->parseMetricsXML(inXMLFilePath, metrics, fieldNames);

            if(metrics->size() != fieldNames->size())
            {
                throw SPDProcessingException("The number of metrics and fieldnames needs to be the same.");
            }

            std::cout << metrics->size() << " metrics where found." << std::endl;

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

    void SPDCalcMetrics::calcMetricForVector2ASCII(std::string inXMLFilePath, std::string inputSPDFile, std::string inputVectorShp, std::string outputASCII) throw (SPDProcessingException)
    {
        try
        {
            std::vector<SPDMetric*> *metrics = new std::vector<SPDMetric*>();
            std::vector<std::string> *fieldNames = new std::vector<std::string>();
            this->parseMetricsXML(inXMLFilePath, metrics, fieldNames);

            if(metrics->size() != fieldNames->size())
            {
                throw SPDProcessingException("The number of metrics and fieldnames needs to be the same.");
            }

            std::cout << metrics->size() << " metrics where found." << std::endl;

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

    void SPDCalcMetrics::parseMetricsXML(std::string inXMLFilePath, std::vector<SPDMetric*> *metrics, std::vector<std::string> *fieldNames) throw(SPDProcessingException)
    {
        
        // Check file exists
        struct stat stFileInfo;
        if (stat(inXMLFilePath.c_str(), &stFileInfo) != 0)
        {
            throw SPDProcessingException("XML file provided does not exist.");
        }
        
        std::cout << "Reading XML file: " << inXMLFilePath << std::endl;
        
        xercesc::DOMLSParser* parser = NULL;
		try
		{
			xercesc::XMLPlatformUtils::Initialize();

            XMLCh tempStr[100];
            xercesc::DOMImplementation *impl = NULL;
            xercesc::ErrorHandler* errHandler = NULL;
            xercesc::DOMDocument *xmlDoc = NULL;
            xercesc::DOMElement *rootElement = NULL;
            XMLCh *metricsTagStr = NULL;
            XMLCh *metricTagStr = NULL;
            xercesc::DOMElement *metricElement = NULL;
            XMLCh *fieldXMLStr = xercesc::XMLString::transcode("field");


			metricsTagStr = xercesc::XMLString::transcode("spdlib:metrics");
			metricTagStr = xercesc::XMLString::transcode("spdlib:metric");

			xercesc::XMLString::transcode("LS", tempStr, 99);
			impl = xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
			if(impl == NULL)
			{
				throw SPDProcessingException("DOMImplementation is NULL");
			}

			// Create Parser
			parser = ((xercesc::DOMImplementationLS*)impl)->createLSParser(xercesc::DOMImplementationLS::MODE_SYNCHRONOUS, 0);
			errHandler = (xercesc::ErrorHandler*) new xercesc::HandlerBase();
			parser->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMErrorHandler, errHandler);

			// Open Document
			xmlDoc = parser->parseURI(inXMLFilePath.c_str());

			// Get the Root element
			rootElement = xmlDoc->getDocumentElement();
			if(!xercesc::XMLString::equals(rootElement->getTagName(), metricsTagStr))
			{
				throw SPDProcessingException("Incorrect root element; Root element should be \"spdlib:metrics\"");
			}

			boost::uint_fast32_t numMetrics = rootElement->getChildElementCount();
            metricElement = rootElement->getFirstElementChild();
            for(boost::uint_fast32_t i = 0; i < numMetrics; ++i)
            {
                // Retreive name (used for naming image band or std::vector< attribute)
                if(metricElement->hasAttribute(fieldXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(fieldXMLStr));
                    fieldNames->push_back(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
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
			xercesc::XMLString::release(&metricsTagStr);
			xercesc::XMLString::release(&metricTagStr);
            xercesc::XMLString::release(&fieldXMLStr);

			xercesc::XMLPlatformUtils::Terminate();
        }
		catch (const xercesc::XMLException& e)
		{
			parser->release();
			char *message = xercesc::XMLString::transcode(e.getMessage());
			std::string outMessage =  std::string("XMLException : ") + std::string(message);
			throw SPDProcessingException(outMessage.c_str());
		}
		catch (const xercesc::DOMException& e)
		{
			parser->release();
			char *message = xercesc::XMLString::transcode(e.getMessage());
			std::string outMessage =  std::string("DOMException : ") + std::string(message);
			throw SPDProcessingException(outMessage.c_str());
		}
		catch(SPDProcessingException &e)
		{
			throw e;
		}
    }

    SPDMetric* SPDCalcMetrics::createMetric(xercesc::DOMElement *metricElement) throw(SPDProcessingException)
    {
        XMLCh *metricadd = xercesc::XMLString::transcode("add");
        XMLCh *metricminus = xercesc::XMLString::transcode("minus");
        XMLCh *metricmultiply = xercesc::XMLString::transcode("multiply");
        XMLCh *metricdivide = xercesc::XMLString::transcode("divide");
        XMLCh *metricpow = xercesc::XMLString::transcode("pow");
        XMLCh *metricabs = xercesc::XMLString::transcode("abs");
        XMLCh *metricsqrt = xercesc::XMLString::transcode("sqrt");
        XMLCh *metricsine = xercesc::XMLString::transcode("sine");
        XMLCh *metriccosine = xercesc::XMLString::transcode("cosine");
        XMLCh *metrictangent = xercesc::XMLString::transcode("tangent");
        XMLCh *metricinvsine = xercesc::XMLString::transcode("invsine");
        XMLCh *metricinvcos = xercesc::XMLString::transcode("invcos");
        XMLCh *metricinvtan = xercesc::XMLString::transcode("invtan");
        XMLCh *metriclog10 = xercesc::XMLString::transcode("log10");
        XMLCh *metricln = xercesc::XMLString::transcode("ln");
        XMLCh *metricexp = xercesc::XMLString::transcode("exp");
        XMLCh *metricpercentage = xercesc::XMLString::transcode("percentage");
        XMLCh *metricaddconst = xercesc::XMLString::transcode("addconst");
        XMLCh *metricminusconstfrom = xercesc::XMLString::transcode("minusconstfrom");
        XMLCh *metricminusfromconst = xercesc::XMLString::transcode("minusfromconst");
        XMLCh *metricmultiplyconst = xercesc::XMLString::transcode("multiplyconst");
        XMLCh *metricdividebyconst = xercesc::XMLString::transcode("dividebyconst");
        XMLCh *metricdivideconstby = xercesc::XMLString::transcode("divideconstby");
        XMLCh *metricpowmetricconst = xercesc::XMLString::transcode("powmetricconst");
        XMLCh *metricpowconstmetric = xercesc::XMLString::transcode("powconstmetric");
        XMLCh *metricnumpulses = xercesc::XMLString::transcode("numpulses");
        XMLCh *metriccanopycover = xercesc::XMLString::transcode("canopycover");
        XMLCh *metriccanopycoverpercent = xercesc::XMLString::transcode("canopycoverpercent");
        XMLCh *metricleeopenness = xercesc::XMLString::transcode("hscoi");
        XMLCh *metricnumreturnsheight = xercesc::XMLString::transcode("numreturnsheight");
        XMLCh *metricsumheight = xercesc::XMLString::transcode("sumheight");
        XMLCh *metricmeanheight = xercesc::XMLString::transcode("meanheight");
        XMLCh *metricmedianheight = xercesc::XMLString::transcode("medianheight");
        XMLCh *metricmodeheight = xercesc::XMLString::transcode("modeheight");
        XMLCh *metricminheight = xercesc::XMLString::transcode("minheight");
        XMLCh *metricmaxheight = xercesc::XMLString::transcode("maxheight");
        XMLCh *metricmaxdominant = xercesc::XMLString::transcode("dominantheight");
        XMLCh *metricstddevheight = xercesc::XMLString::transcode("stddevheight");
        XMLCh *metricvarianceheight = xercesc::XMLString::transcode("varianceheight");
        XMLCh *metricabsdeviationheight = xercesc::XMLString::transcode("absdeviationheight");
        XMLCh *metriccoefficientofvariationheight = xercesc::XMLString::transcode("coefficientofvariationheight");
        XMLCh *metricpercentileheight = xercesc::XMLString::transcode("percentileheight");
        XMLCh *metricskewnessheight = xercesc::XMLString::transcode("skewnessheight");
        XMLCh *metricpersonmodeheight = xercesc::XMLString::transcode("personmodeheight");
        XMLCh *metricpersonmedianheight = xercesc::XMLString::transcode("personmedianheight");
        XMLCh *metrickurtosisheight = xercesc::XMLString::transcode("kurtosisheight");
        XMLCh *metricreturnsaboveheightmetric = xercesc::XMLString::transcode("returnsaboveheightmetric");
        XMLCh *metricreturnsbelowheightmetric = xercesc::XMLString::transcode("returnsbelowheightmetric");
        XMLCh *metricnumreturnsz = xercesc::XMLString::transcode("numreturnsz");
        XMLCh *metricsumz = xercesc::XMLString::transcode("sumz");
        XMLCh *metricmeanz = xercesc::XMLString::transcode("meanz");
        XMLCh *metricmedianz = xercesc::XMLString::transcode("medianz");
        XMLCh *metricmodez = xercesc::XMLString::transcode("modez");
        XMLCh *metricminz = xercesc::XMLString::transcode("minz");
        XMLCh *metricmaxz = xercesc::XMLString::transcode("maxz");
        XMLCh *metricstddevz = xercesc::XMLString::transcode("stddevz");
        XMLCh *metricvariancez = xercesc::XMLString::transcode("variancez");
        XMLCh *metricabsdeviationz = xercesc::XMLString::transcode("absdeviationz");
        XMLCh *metriccoefficientofvariationz = xercesc::XMLString::transcode("coefficientofvariationz");
        XMLCh *metricpercentilez = xercesc::XMLString::transcode("percentilez");
        XMLCh *metricskewnessz = xercesc::XMLString::transcode("skewnessz");
        XMLCh *metricpersonmodez = xercesc::XMLString::transcode("personmodez");
        XMLCh *metricpersonmedianz = xercesc::XMLString::transcode("personmedianz");
        XMLCh *metrickurtosisz = xercesc::XMLString::transcode("kurtosisz");
        XMLCh *metricreturnsabovezmetric = xercesc::XMLString::transcode("returnsabovezmetric");
        XMLCh *metricreturnsbelowzmetric = xercesc::XMLString::transcode("returnsbelowzmetric");
        XMLCh *metricnumreturnsamplitude = xercesc::XMLString::transcode("numreturnsamplitude");
        XMLCh *metricsumamplitude = xercesc::XMLString::transcode("sumamplitude");
        XMLCh *metricmeanamplitude = xercesc::XMLString::transcode("meanamplitude");
        XMLCh *metricmedianamplitude = xercesc::XMLString::transcode("medianamplitude");
        XMLCh *metricmodeamplitude = xercesc::XMLString::transcode("modeamplitude");
        XMLCh *metricminamplitude = xercesc::XMLString::transcode("minamplitude");
        XMLCh *metricmaxamplitude = xercesc::XMLString::transcode("maxamplitude");
        XMLCh *metricstddevamplitude = xercesc::XMLString::transcode("stddevamplitude");
        XMLCh *metricvarianceamplitude = xercesc::XMLString::transcode("varianceamplitude");
        XMLCh *metricabsdeviationamplitude = xercesc::XMLString::transcode("absdeviationamplitude");
        XMLCh *metriccoefficientofvariationamplitude = xercesc::XMLString::transcode("coefficientofvariationamplitude");
        XMLCh *metricpercentileamplitude = xercesc::XMLString::transcode("percentileamplitude");
        XMLCh *metricskewnessamplitude = xercesc::XMLString::transcode("skewnessamplitude");
        XMLCh *metricpersonmodeamplitude = xercesc::XMLString::transcode("personmodeamplitude");
        XMLCh *metricpersonmedianamplitude = xercesc::XMLString::transcode("personmedianamplitude");
        XMLCh *metrickurtosisamplitude = xercesc::XMLString::transcode("kurtosisamplitude");
        XMLCh *metricreturnsaboveamplitudemetric = xercesc::XMLString::transcode("returnsaboveamplitudemetric");
        XMLCh *metricreturnsbelowamplitudemetric = xercesc::XMLString::transcode("returnsbelowamplitudemetric");
        XMLCh *metricnumreturnsrange = xercesc::XMLString::transcode("numreturnsrange");
        XMLCh *metricsumrange = xercesc::XMLString::transcode("sumrange");
        XMLCh *metricmeanrange = xercesc::XMLString::transcode("meanrange");
        XMLCh *metricmedianrange = xercesc::XMLString::transcode("medianrange");
        XMLCh *metricmoderange = xercesc::XMLString::transcode("moderange");
        XMLCh *metricminrange = xercesc::XMLString::transcode("minrange");
        XMLCh *metricmaxrange = xercesc::XMLString::transcode("maxrange");
        XMLCh *metricstddevrange = xercesc::XMLString::transcode("stddevrange");
        XMLCh *metricvariancerange = xercesc::XMLString::transcode("variancerange");
        XMLCh *metricabsdeviationrange = xercesc::XMLString::transcode("absdeviationrange");
        XMLCh *metriccoefficientofvariationrange = xercesc::XMLString::transcode("coefficientofvariationrange");
        XMLCh *metricpercentilerange = xercesc::XMLString::transcode("percentilerange");
        XMLCh *metricskewnessrange = xercesc::XMLString::transcode("skewnessrange");
        XMLCh *metricpersonmoderange = xercesc::XMLString::transcode("personmoderange");
        XMLCh *metricpersonmedianrange = xercesc::XMLString::transcode("personmedianrange");
        XMLCh *metrickurtosisrange = xercesc::XMLString::transcode("kurtosisrange");
        XMLCh *metricreturnsaboverangemetric = xercesc::XMLString::transcode("returnsaboverangemetric");
        XMLCh *metricreturnsbelowrangemetric = xercesc::XMLString::transcode("returnsbelowrangemetric");
        XMLCh *metricnumreturnswidth = xercesc::XMLString::transcode("numreturnswidth");
        XMLCh *metricsumwidth = xercesc::XMLString::transcode("sumwidth");
        XMLCh *metricmeanwidth = xercesc::XMLString::transcode("meanwidth");
        XMLCh *metricmedianwidth = xercesc::XMLString::transcode("medianwidth");
        XMLCh *metricmodewidth = xercesc::XMLString::transcode("modewidth");
        XMLCh *metricminwidth = xercesc::XMLString::transcode("minwidth");
        XMLCh *metricmaxwidth = xercesc::XMLString::transcode("maxwidth");
        XMLCh *metricstddevwidth = xercesc::XMLString::transcode("stddevwidth");
        XMLCh *metricvariancewidth = xercesc::XMLString::transcode("variancewidth");
        XMLCh *metricabsdeviationwidth = xercesc::XMLString::transcode("absdeviationwidth");
        XMLCh *metriccoefficientofvariationwidth = xercesc::XMLString::transcode("coefficientofvariationwidth");
        XMLCh *metricpercentilewidth = xercesc::XMLString::transcode("percentilewidth");
        XMLCh *metricskewnesswidth = xercesc::XMLString::transcode("skewnesswidth");
        XMLCh *metricpersonmodewidth = xercesc::XMLString::transcode("personmodewidth");
        XMLCh *metricpersonmedianwidth = xercesc::XMLString::transcode("personmedianwidth");
        XMLCh *metrickurtosiswidth = xercesc::XMLString::transcode("kurtosiswidth");
        XMLCh *metrichome = xercesc::XMLString::transcode("home");
        XMLCh *metricreturnsabovewidthmetric = xercesc::XMLString::transcode("returnsabovewidthmetric");
        XMLCh *metricreturnsbelowwidthmetric = xercesc::XMLString::transcode("returnsbelowwidthmetric");
        XMLCh *metricNameXMLStr = xercesc::XMLString::transcode("metric");
        XMLCh *metricReturnXMLStr = xercesc::XMLString::transcode("return");
        XMLCh *metricClassXMLStr = xercesc::XMLString::transcode("class");
        XMLCh *metricMinNumReturnsXMLStr = xercesc::XMLString::transcode("minNumReturns");
        XMLCh *metricUpThresholdXMLStr = xercesc::XMLString::transcode("upthreshold");
        XMLCh *metricLowThresholdXMLStr = xercesc::XMLString::transcode("lowthreshold");
        XMLCh *heightUpThresholdXMLStr = xercesc::XMLString::transcode("heightup");
        XMLCh *heightLowThresholdXMLStr = xercesc::XMLString::transcode("heightlow");
        XMLCh *allXMLStr = xercesc::XMLString::transcode("All");
        XMLCh *notFirstXMLStr = xercesc::XMLString::transcode("NotFirst");
        XMLCh *firstXMLStr = xercesc::XMLString::transcode("First");
        XMLCh *lastXMLStr = xercesc::XMLString::transcode("Last");
        XMLCh *firstLastXMLStr = xercesc::XMLString::transcode("FirstLast");
        XMLCh *notGrdXMLStr = xercesc::XMLString::transcode("NotGrd");
        XMLCh *vegXMLStr = xercesc::XMLString::transcode("Veg");
        XMLCh *grdXMLStr = xercesc::XMLString::transcode("Grd");

        SPDTextFileUtilities textUtils;
        const char *nanVal = "NaN";
        SPDMetric *metric = NULL;
        XMLCh *metricNameStr = NULL;

        boost::uint_fast32_t returnID = 0;
        boost::uint_fast32_t classID = 0;
        boost::uint_fast32_t minNumReturns = 1;
        double upThreshold = 0;
        double lowThreshold = 0;
        double heightUpThreshold = 0;
        double heightLowThreshold = 0;

        try
        {
            if(metricElement->hasAttribute(metricNameXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(metricNameXMLStr));
                metricNameStr = xercesc::XMLString::transcode(charValue);
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                throw SPDProcessingException("The \'metric\' attribute was not provided for the metric element.");
            }

            if(metricElement->hasAttribute(metricReturnXMLStr))
            {
                if(xercesc::XMLString::equals(allXMLStr, metricElement->getAttribute(metricReturnXMLStr)))
                {
                    returnID = SPD_ALL_RETURNS;
                }
                else if(xercesc::XMLString::equals(notFirstXMLStr, metricElement->getAttribute(metricReturnXMLStr)))
                {
                    returnID = SPD_NOTFIRST_RETURNS;
                }
                else if(xercesc::XMLString::equals(firstXMLStr, metricElement->getAttribute(metricReturnXMLStr)))
                {
                    returnID = SPD_FIRST_RETURNS;
                }
                else if(xercesc::XMLString::equals(lastXMLStr, metricElement->getAttribute(metricReturnXMLStr)))
                {
                    returnID = SPD_LAST_RETURNS;
                }
                else if(xercesc::XMLString::equals(firstLastXMLStr, metricElement->getAttribute(metricReturnXMLStr)))
                {
                    returnID = SPD_FIRSTLAST_RETURNS;
                }
                else
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(metricReturnXMLStr));
                    returnID = textUtils.strto32bitUInt(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
            }
            else
            {
                returnID = SPD_ALL_RETURNS;
            }

            if(metricElement->hasAttribute(metricClassXMLStr))
            {
                if(xercesc::XMLString::equals(allXMLStr, metricElement->getAttribute(metricClassXMLStr)))
                {
                    classID = SPD_ALL_CLASSES;
                }
                else if(xercesc::XMLString::equals(notGrdXMLStr, metricElement->getAttribute(metricClassXMLStr)))
                {
                    classID = SPD_NOT_GROUND;
                }
                else if(xercesc::XMLString::equals(grdXMLStr, metricElement->getAttribute(metricClassXMLStr)))
                {
                    classID = SPD_GROUND;
                }
                else if(xercesc::XMLString::equals(vegXMLStr, metricElement->getAttribute(metricClassXMLStr)))
                {
                    classID = SPD_VEGETATION;
                }
                else
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(metricClassXMLStr));
                    classID = textUtils.strto32bitUInt(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
            }
            else
            {
                classID = SPD_ALL_CLASSES;
            }

            if(metricElement->hasAttribute(metricMinNumReturnsXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(metricMinNumReturnsXMLStr));
                minNumReturns = textUtils.strto32bitUInt(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                minNumReturns = 1;
            }

            if(metricElement->hasAttribute(metricUpThresholdXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(metricUpThresholdXMLStr));
                upThreshold = textUtils.strtodouble(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                upThreshold = nan(nanVal);
            }

            if(metricElement->hasAttribute(metricLowThresholdXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(metricLowThresholdXMLStr));
                lowThreshold = textUtils.strtodouble(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                lowThreshold = nan(nanVal);
            }

            if(metricElement->hasAttribute(heightUpThresholdXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(heightUpThresholdXMLStr));
                heightUpThreshold = textUtils.strtodouble(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                heightUpThreshold = nan(nanVal);
            }

            if(metricElement->hasAttribute(heightLowThresholdXMLStr))
            {
                char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(heightLowThresholdXMLStr));
                heightLowThreshold = textUtils.strtodouble(std::string(charValue));
                xercesc::XMLString::release(&charValue);
            }
            else
            {
                heightLowThreshold = nan(nanVal);
            }

            /* Create boolean to store if metric has been found.
             * Avoids implementing as large if/if else which was causing
             * problems when compiling under Windows.
             */
            bool foundMetric = false;

            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricadd))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics < 2)
                {
                    std::cout << "Number of metrics = " << numMetrics << std::endl;
                    throw SPDProcessingException("The \'add\' metric needs at least two child metrics.");
                }

                std::vector<SPDMetric*> *metrics = new std::vector<SPDMetric*>();

                xercesc::DOMElement *tmpMetricElement = metricElement->getFirstElementChild();
                for(boost::uint_fast32_t i = 0; i < numMetrics; ++i)
                {
                    // Retrieve Metric and add to list.
                    metrics->push_back(this->createMetric(tmpMetricElement));

                    // Move on to next metric
                    tmpMetricElement = tmpMetricElement->getNextElementSibling();
                }

                metric = new SPDMetricAdd(metrics);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricminus))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 2)
                {
                    throw SPDProcessingException("The \'minus\' metric needs two child metrics.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                metricElementIn = metricElementIn->getNextElementSibling();
                SPDMetric *metric2 = this->createMetric(metricElementIn);

                metric = new SPDMetricMinus(metric1, metric2);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmultiply))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics < 2)
                {
                    throw SPDProcessingException("The \'multiply\' metric needs at least two child metrics.");
                }

                std::vector<SPDMetric*> *metrics = new std::vector<SPDMetric*>();

                xercesc::DOMElement *tmpMetricElement = metricElement->getFirstElementChild();
                for(boost::uint_fast32_t i = 0; i < numMetrics; ++i)
                {
                    // Retrieve Metric and add to list.
                    metrics->push_back(this->createMetric(tmpMetricElement));

                    // Move on to next metric
                    tmpMetricElement = tmpMetricElement->getNextElementSibling();
                }

                metric = new SPDMetricMultiply(metrics);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricdivide))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 2)
                {
                    throw SPDProcessingException("The \'divide\' metric needs two child metrics.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                metricElementIn = metricElementIn->getNextElementSibling();
                SPDMetric *metric2 = this->createMetric(metricElementIn);

                metric = new SPDMetricDivide(metric1, metric2);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpow))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 2)
                {
                    throw SPDProcessingException("The \'pow\' metric needs two child metrics.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                metricElementIn = metricElementIn->getNextElementSibling();
                SPDMetric *metric2 = this->createMetric(metricElementIn);

                metric = new SPDMetricPow(metric1, metric2);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricabs))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'abs\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricAbs(metric1);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricsqrt))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'sqrt\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricSqrt(metric1);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricsine))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'sine\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricSine(metric1);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricabs))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'abs\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricAbs(metric1);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metriccosine))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'cosine\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricCosine(metric1);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metrictangent))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'tangent\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricTangent(metric1);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricinvsine))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'invsine\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricInvSine(metric1);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricinvcos))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'invcos\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricInvCos(metric1);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricinvtan))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'invtan\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricInvTan(metric1);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metriclog10))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'log10\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricLog10(metric1);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricln))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'ln\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricLn(metric1);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricexp))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'exp\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricExp(metric1);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpercentage))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 2)
                {
                    throw SPDProcessingException("The \'percentage\' metric needs two child metrics.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);
                metricElementIn = metricElementIn->getNextElementSibling();
                SPDMetric *metric2 = this->createMetric(metricElementIn);

                metric = new SPDMetricPercentage(metric1, metric2);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricaddconst))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'add const\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                double constVal = 0;
                XMLCh *constXMLStr = xercesc::XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                xercesc::XMLString::release(&constXMLStr);

                metric = new SPDMetricAddConst(metric1, constVal);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricminusconstfrom))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'minus const from\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                double constVal = 0;
                XMLCh *constXMLStr = xercesc::XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                xercesc::XMLString::release(&constXMLStr);

                metric = new SPDMetricMinusConstFrom(metric1, constVal);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricminusfromconst))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'minus from const\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                double constVal = 0;
                XMLCh *constXMLStr = xercesc::XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                xercesc::XMLString::release(&constXMLStr);

                metric = new SPDMetricMinusFromConst(metric1, constVal);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmultiplyconst))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'multiply const\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                double constVal = 0;
                XMLCh *constXMLStr = xercesc::XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                xercesc::XMLString::release(&constXMLStr);

                metric = new SPDMetricMultiplyConst(metric1, constVal);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricdividebyconst))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'divide by const\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                double constVal = 0;
                XMLCh *constXMLStr = xercesc::XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                xercesc::XMLString::release(&constXMLStr);

                metric = new SPDMetricDivideByConst(metric1, constVal);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricdivideconstby))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'divide const by\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                double constVal = 0;
                XMLCh *constXMLStr = xercesc::XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                xercesc::XMLString::release(&constXMLStr);

                metric = new SPDMetricDivideConstBy(metric1, constVal);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpowmetricconst))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'pow metric const\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                double constVal = 0;
                XMLCh *constXMLStr = xercesc::XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                xercesc::XMLString::release(&constXMLStr);

                metric = new SPDMetricPowMetricConst(metric1, constVal);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpowconstmetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'pow const metric\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                double constVal = 0;
                XMLCh *constXMLStr = xercesc::XMLString::transcode("const");
                if(metricElement->hasAttribute(constXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(constXMLStr));
                    constVal = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'const\' value has not been provided.");
                }
                xercesc::XMLString::release(&constXMLStr);

                metric = new SPDMetricPowConstMetric(metric1, constVal);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricnumpulses))
            {
                metric = new SPDMetricCalcNumPulses(minNumReturns);
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metriccanopycover))
            {
                float resolution = 0;
                XMLCh *resolutionXMLStr = xercesc::XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtofloat(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                xercesc::XMLString::release(&resolutionXMLStr);

                float radius = 0;
                XMLCh *radiusXMLStr = xercesc::XMLString::transcode("radius");
                if(metricElement->hasAttribute(radiusXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(radiusXMLStr));
                    radius = textUtils.strtofloat(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'radius\' value has not been provided.");
                }
                xercesc::XMLString::release(&radiusXMLStr);

                metric = new SPDMetricCalcCanopyCover(resolution, radius, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metriccanopycoverpercent))
            {
                float resolution = 0;
                XMLCh *resolutionXMLStr = xercesc::XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtofloat(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                xercesc::XMLString::release(&resolutionXMLStr);

                float radius = 0;
                XMLCh *radiusXMLStr = xercesc::XMLString::transcode("radius");
                if(metricElement->hasAttribute(radiusXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(radiusXMLStr));
                    radius = textUtils.strtofloat(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'radius\' value has not been provided.");
                }
                xercesc::XMLString::release(&radiusXMLStr);

                metric = new SPDMetricCalcCanopyCoverPercent(resolution, radius, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricleeopenness))
            {
                float vres = 0;
                XMLCh *vresolutionXMLStr = xercesc::XMLString::transcode("vres");
                if(metricElement->hasAttribute(vresolutionXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(vresolutionXMLStr));
                    vres = textUtils.strtofloat(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'vres\' value has not been provided.");
                }
                xercesc::XMLString::release(&vresolutionXMLStr);

                metric = new SPDMetricCalcLeeOpennessHeight(vres, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricnumreturnsheight))
            {
                metric = new SPDMetricCalcNumReturnsHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricsumheight))
            {
                metric = new SPDMetricCalcSumHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmeanheight))
            {
                metric = new SPDMetricCalcMeanHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmedianheight))
            {
                metric = new SPDMetricCalcMedianHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmodeheight))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = xercesc::XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                xercesc::XMLString::release(&resolutionXMLStr);

                metric = new SPDMetricCalcModeHeight(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricminheight))
            {
                metric = new SPDMetricCalcMinHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmaxheight))
            {
                metric = new SPDMetricCalcMaxHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmaxdominant))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = xercesc::XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                xercesc::XMLString::release(&resolutionXMLStr);

                metric = new SPDMetricCalcDominantHeight(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricstddevheight))
            {
                metric = new SPDMetricCalcStdDevHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricvarianceheight))
            {
                metric = new SPDMetricCalcVarianceHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricabsdeviationheight))
            {
                metric = new SPDMetricCalcAbsDeviationHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metriccoefficientofvariationheight))
            {
                metric = new SPDMetricCalcCoefficientOfVariationHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpercentileheight))
            {
                boost::uint_fast32_t percentileVal = 0;
                XMLCh *percentileXMLStr = xercesc::XMLString::transcode("percentile");
                if(metricElement->hasAttribute(percentileXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(percentileXMLStr));
                    percentileVal = textUtils.strto32bitUInt(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'percentile\' value has not been provided.");
                }
                xercesc::XMLString::release(&percentileXMLStr);
                metric = new SPDMetricCalcPercentileHeight(percentileVal,returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricskewnessheight))
            {
                metric = new SPDMetricCalcSkewnessHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpersonmodeheight))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = xercesc::XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                xercesc::XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcPersonModeSkewnessHeight(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpersonmedianheight))
            {
                metric = new SPDMetricCalcPersonMedianSkewnessHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metrickurtosisheight))
            {
                metric = new SPDMetricCalcKurtosisHeight(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricreturnsaboveheightmetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns above height metric\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricCalcNumReturnsAboveMetricHeight(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricreturnsbelowheightmetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns below height metric\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricCalcNumReturnsBelowMetricHeight(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricnumreturnsz))
            {
                metric = new SPDMetricCalcNumReturnsZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricsumz))
            {
                metric = new SPDMetricCalcSumZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmeanz))
            {
                metric = new SPDMetricCalcMeanZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmedianz))
            {
                metric = new SPDMetricCalcMedianZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmodez))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = xercesc::XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                xercesc::XMLString::release(&resolutionXMLStr);

                metric = new SPDMetricCalcModeZ(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricminz))
            {
                metric = new SPDMetricCalcMinZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmaxz))
            {
                metric = new SPDMetricCalcMaxZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricstddevz))
            {
                metric = new SPDMetricCalcStdDevZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricvariancez))
            {
                metric = new SPDMetricCalcVarianceZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricabsdeviationz))
            {
                metric = new SPDMetricCalcAbsDeviationZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metriccoefficientofvariationz))
            {
                metric = new SPDMetricCalcCoefficientOfVariationZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpercentilez))
            {
                boost::uint_fast32_t percentileVal = 0;
                XMLCh *percentileXMLStr = xercesc::XMLString::transcode("percentile");
                if(metricElement->hasAttribute(percentileXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(percentileXMLStr));
                    percentileVal = textUtils.strto32bitUInt(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'percentile\' value has not been provided.");
                }
                xercesc::XMLString::release(&percentileXMLStr);
                metric = new SPDMetricCalcPercentileZ(percentileVal,returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricskewnessz))
            {
                metric = new SPDMetricCalcSkewnessZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpersonmodez))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = xercesc::XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                xercesc::XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcPersonModeSkewnessZ(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpersonmedianz))
            {
                metric = new SPDMetricCalcPersonMedianSkewnessZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metrickurtosisz))
            {
                metric = new SPDMetricCalcKurtosisZ(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricreturnsabovezmetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns above z metric\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricCalcNumReturnsAboveMetricZ(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricreturnsbelowzmetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns below z metric\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricCalcNumReturnsBelowMetricZ(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricnumreturnsamplitude))
            {
                metric = new SPDMetricCalcNumReturnsAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricsumamplitude))
            {
                metric = new SPDMetricCalcSumAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmeanamplitude))
            {
                metric = new SPDMetricCalcMeanAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmedianamplitude))
            {
                metric = new SPDMetricCalcMedianAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmodeamplitude))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = xercesc::XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                xercesc::XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcModeAmplitude(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricminamplitude))
            {
                metric = new SPDMetricCalcMinAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmaxamplitude))
            {
                metric = new SPDMetricCalcMaxAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricstddevamplitude))
            {
                metric = new SPDMetricCalcStdDevAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricvarianceamplitude))
            {
                metric = new SPDMetricCalcVarianceAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricabsdeviationamplitude))
            {
                metric = new SPDMetricCalcAbsDeviationAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metriccoefficientofvariationamplitude))
            {
                metric = new SPDMetricCalcCoefficientOfVariationAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpercentileamplitude))
            {
                boost::uint_fast32_t percentileVal = 0;
                XMLCh *percentileXMLStr = xercesc::XMLString::transcode("percentile");
                if(metricElement->hasAttribute(percentileXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(percentileXMLStr));
                    percentileVal = textUtils.strto32bitUInt(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'percentile\' value has not been provided.");
                }
                xercesc::XMLString::release(&percentileXMLStr);
                metric = new SPDMetricCalcPercentileAmplitude(percentileVal,returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricskewnessamplitude))
            {
                metric = new SPDMetricCalcSkewnessAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpersonmodeamplitude))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = xercesc::XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                xercesc::XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcPersonModeSkewnessAmplitude(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpersonmedianamplitude))
            {
                metric = new SPDMetricCalcPersonMedianSkewnessAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metrickurtosisamplitude))
            {
                metric = new SPDMetricCalcKurtosisAmplitude(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricreturnsaboveamplitudemetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns above amplitude metric\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricCalcNumReturnsAboveMetricAmplitude(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricreturnsbelowamplitudemetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns below amplitude metric\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricCalcNumReturnsBelowMetricAmplitude(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricnumreturnsrange))
            {
                metric = new SPDMetricCalcNumReturnsRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricsumrange))
            {
                metric = new SPDMetricCalcSumRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmeanrange))
            {
                metric = new SPDMetricCalcMeanRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmedianrange))
            {
                metric = new SPDMetricCalcMedianRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmoderange))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = xercesc::XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                xercesc::XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcModeRange(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricminrange))
            {
                metric = new SPDMetricCalcMinRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmaxrange))
            {
                metric = new SPDMetricCalcMaxRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricstddevrange))
            {
                metric = new SPDMetricCalcStdDevRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricvariancerange))
            {
                metric = new SPDMetricCalcVarianceRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricabsdeviationrange))
            {
                metric = new SPDMetricCalcAbsDeviationRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metriccoefficientofvariationrange))
            {
                metric = new SPDMetricCalcCoefficientOfVariationRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpercentilerange))
            {
                boost::uint_fast32_t percentileVal = 0;
                XMLCh *percentileXMLStr = xercesc::XMLString::transcode("percentile");
                if(metricElement->hasAttribute(percentileXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(percentileXMLStr));
                    percentileVal = textUtils.strto32bitUInt(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'percentile\' value has not been provided.");
                }
                xercesc::XMLString::release(&percentileXMLStr);
                metric = new SPDMetricCalcPercentileRange(percentileVal,returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricskewnessrange))
            {
                metric = new SPDMetricCalcSkewnessRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpersonmoderange))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = xercesc::XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                xercesc::XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcPersonModeSkewnessRange(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpersonmedianrange))
            {
                metric = new SPDMetricCalcPersonMedianSkewnessRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metrickurtosisrange))
            {
                metric = new SPDMetricCalcKurtosisRange(returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricreturnsaboverangemetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns above range metric\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricCalcNumReturnsAboveMetricRange(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricreturnsbelowrangemetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns below range metric\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricCalcNumReturnsBelowMetricRange(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricnumreturnswidth))
            {
                metric = new SPDMetricCalcNumReturnsWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricsumwidth))
            {
                metric = new SPDMetricCalcSumWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmeanwidth))
            {
                metric = new SPDMetricCalcMeanWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmedianwidth))
            {
                metric = new SPDMetricCalcMedianWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmodewidth))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = xercesc::XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                xercesc::XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcModeWidth(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricminwidth))
            {
                metric = new SPDMetricCalcMinWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricmaxwidth))
            {
                metric = new SPDMetricCalcMaxWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricstddevwidth))
            {
                metric = new SPDMetricCalcStdDevWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricvariancewidth))
            {
                metric = new SPDMetricCalcVarianceWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricabsdeviationwidth))
            {
                metric = new SPDMetricCalcAbsDeviationWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metriccoefficientofvariationwidth))
            {
                metric = new SPDMetricCalcCoefficientOfVariationWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpercentilewidth))
            {
                boost::uint_fast32_t percentileVal = 0;
                XMLCh *percentileXMLStr = xercesc::XMLString::transcode("percentile");
                if(metricElement->hasAttribute(percentileXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(percentileXMLStr));
                    percentileVal = textUtils.strto32bitUInt(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'percentile\' value has not been provided.");
                }
                xercesc::XMLString::release(&percentileXMLStr);
                metric = new SPDMetricCalcPercentileWidth(percentileVal,returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricskewnesswidth))
            {
                metric = new SPDMetricCalcSkewnessWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpersonmodewidth))
            {
                double resolution = 0;
                XMLCh *resolutionXMLStr = xercesc::XMLString::transcode("resolution");
                if(metricElement->hasAttribute(resolutionXMLStr))
                {
                    char *charValue = xercesc::XMLString::transcode(metricElement->getAttribute(resolutionXMLStr));
                    resolution = textUtils.strtodouble(std::string(charValue));
                    xercesc::XMLString::release(&charValue);
                }
                else
                {
                    throw SPDProcessingException("The \'resolution\' value has not been provided.");
                }
                xercesc::XMLString::release(&resolutionXMLStr);
                metric = new SPDMetricCalcPersonModeSkewnessWidth(resolution, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricpersonmedianwidth))
            {
                metric = new SPDMetricCalcPersonMedianSkewnessWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metrickurtosiswidth))
            {
                metric = new SPDMetricCalcKurtosisWidth(returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricreturnsabovewidthmetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns above width metric\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricCalcNumReturnsAboveMetricWidth(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metricreturnsbelowwidthmetric))
            {
                unsigned int numMetrics = metricElement->getChildElementCount();
                if(numMetrics != 1)
                {
                    throw SPDProcessingException("The \'returns below width metric\' metric needs one child metric.");
                }

                xercesc::DOMElement *metricElementIn = metricElement->getFirstElementChild();
                SPDMetric *metric1 = this->createMetric(metricElementIn);

                metric = new SPDMetricCalcNumReturnsBelowMetricWidth(metric1, returnID, classID, minNumReturns, upThreshold, lowThreshold, heightUpThreshold, heightLowThreshold);
                foundMetric = true;
            }
            if(!foundMetric && xercesc::XMLString::equals(metricNameStr, metrichome))
            {
                // Number of returns is set to 0 as only using waveform data
                // The lower threshold is fixed at 0 so only above ground returns are considered.
                metric = new SPDMetricCalcHeightOfMedianEnergy(returnID, classID, 0, upThreshold, 0);
                foundMetric = true;
            }
            if(!foundMetric)
            {
                std::string message = "Metric \'" + std::string(xercesc::XMLString::transcode(metricNameStr)) + "\' has not been recognised.";
                throw SPDProcessingException(message);
            }
        }
        catch (const xercesc::XMLException& e)
		{
			char *message = xercesc::XMLString::transcode(e.getMessage());
			std::string outMessage =  std::string("XMLException : ") + std::string(message);
			throw SPDProcessingException(outMessage.c_str());
		}
		catch (const xercesc::DOMException& e)
		{
			char *message = xercesc::XMLString::transcode(e.getMessage());
			std::string outMessage =  std::string("DOMException : ") + std::string(message);
			throw SPDProcessingException(outMessage.c_str());
		}
		catch(SPDProcessingException &e)
		{
			throw e;
		}

        xercesc::XMLString::release(&metricadd);
        xercesc::XMLString::release(&metricminus);
        xercesc::XMLString::release(&metricmultiply);
        xercesc::XMLString::release(&metricdivide);
        xercesc::XMLString::release(&metricpow);
        xercesc::XMLString::release(&metricabs);
        xercesc::XMLString::release(&metricsqrt);
        xercesc::XMLString::release(&metricsine);
        xercesc::XMLString::release(&metriccosine);
        xercesc::XMLString::release(&metrictangent);
        xercesc::XMLString::release(&metricinvsine);
        xercesc::XMLString::release(&metricinvcos);
        xercesc::XMLString::release(&metricinvtan);
        xercesc::XMLString::release(&metriclog10);
        xercesc::XMLString::release(&metricln);
        xercesc::XMLString::release(&metricexp);
        xercesc::XMLString::release(&metricpercentage);
        xercesc::XMLString::release(&metricaddconst);
        xercesc::XMLString::release(&metricminusconstfrom);
        xercesc::XMLString::release(&metricminusfromconst);
        xercesc::XMLString::release(&metricmultiplyconst);
        xercesc::XMLString::release(&metricdividebyconst);
        xercesc::XMLString::release(&metricdivideconstby);
        xercesc::XMLString::release(&metricpowmetricconst);
        xercesc::XMLString::release(&metricpowconstmetric);
        xercesc::XMLString::release(&metricnumpulses);
        xercesc::XMLString::release(&metriccanopycover);
        xercesc::XMLString::release(&metriccanopycoverpercent);
        xercesc::XMLString::release(&metricleeopenness);
        xercesc::XMLString::release(&metricnumreturnsheight);
        xercesc::XMLString::release(&metricmeanheight);
        xercesc::XMLString::release(&metricsumheight);
        xercesc::XMLString::release(&metricmedianheight);
        xercesc::XMLString::release(&metricmodeheight);
        xercesc::XMLString::release(&metricminheight);
        xercesc::XMLString::release(&metricmaxheight);
        xercesc::XMLString::release(&metricmaxdominant);
        xercesc::XMLString::release(&metricstddevheight);
        xercesc::XMLString::release(&metricvarianceheight);
        xercesc::XMLString::release(&metricabsdeviationheight);
        xercesc::XMLString::release(&metriccoefficientofvariationheight);
        xercesc::XMLString::release(&metricpercentileheight);
        xercesc::XMLString::release(&metricskewnessheight);
        xercesc::XMLString::release(&metricpersonmodeheight);
        xercesc::XMLString::release(&metricpersonmedianheight);
        xercesc::XMLString::release(&metrickurtosisheight);
        xercesc::XMLString::release(&metricreturnsaboveheightmetric);
        xercesc::XMLString::release(&metricreturnsbelowheightmetric);
        xercesc::XMLString::release(&metricnumreturnsz);
        xercesc::XMLString::release(&metricmeanz);
        xercesc::XMLString::release(&metricsumz);
        xercesc::XMLString::release(&metricmedianz);
        xercesc::XMLString::release(&metricmodez);
        xercesc::XMLString::release(&metricminz);
        xercesc::XMLString::release(&metricmaxz);
        xercesc::XMLString::release(&metricstddevz);
        xercesc::XMLString::release(&metricvariancez);
        xercesc::XMLString::release(&metricabsdeviationz);
        xercesc::XMLString::release(&metriccoefficientofvariationz);
        xercesc::XMLString::release(&metricpercentilez);
        xercesc::XMLString::release(&metricskewnessz);
        xercesc::XMLString::release(&metricpersonmodez);
        xercesc::XMLString::release(&metricpersonmedianz);
        xercesc::XMLString::release(&metrickurtosisz);
        xercesc::XMLString::release(&metricreturnsabovezmetric);
        xercesc::XMLString::release(&metricreturnsbelowzmetric);
        xercesc::XMLString::release(&metricnumreturnsamplitude);
        xercesc::XMLString::release(&metricmeanamplitude);
        xercesc::XMLString::release(&metricsumamplitude);
        xercesc::XMLString::release(&metricmedianamplitude);
        xercesc::XMLString::release(&metricmodeamplitude);
        xercesc::XMLString::release(&metricminamplitude);
        xercesc::XMLString::release(&metricmaxamplitude);
        xercesc::XMLString::release(&metricstddevamplitude);
        xercesc::XMLString::release(&metricvarianceamplitude);
        xercesc::XMLString::release(&metricabsdeviationamplitude);
        xercesc::XMLString::release(&metriccoefficientofvariationamplitude);
        xercesc::XMLString::release(&metricpercentileamplitude);
        xercesc::XMLString::release(&metricskewnessamplitude);
        xercesc::XMLString::release(&metricpersonmodeamplitude);
        xercesc::XMLString::release(&metricpersonmedianamplitude);
        xercesc::XMLString::release(&metrickurtosisamplitude);
        xercesc::XMLString::release(&metricreturnsaboveamplitudemetric);
        xercesc::XMLString::release(&metricreturnsbelowamplitudemetric);
        xercesc::XMLString::release(&metricnumreturnsrange);
        xercesc::XMLString::release(&metricmeanrange);
        xercesc::XMLString::release(&metricsumrange);
        xercesc::XMLString::release(&metricmedianrange);
        xercesc::XMLString::release(&metricmoderange);
        xercesc::XMLString::release(&metricminrange);
        xercesc::XMLString::release(&metricmaxrange);
        xercesc::XMLString::release(&metricstddevrange);
        xercesc::XMLString::release(&metricvariancerange);
        xercesc::XMLString::release(&metricabsdeviationrange);
        xercesc::XMLString::release(&metriccoefficientofvariationrange);
        xercesc::XMLString::release(&metricpercentilerange);
        xercesc::XMLString::release(&metricskewnessrange);
        xercesc::XMLString::release(&metricpersonmoderange);
        xercesc::XMLString::release(&metricpersonmedianrange);
        xercesc::XMLString::release(&metrickurtosisrange);
        xercesc::XMLString::release(&metricreturnsaboverangemetric);
        xercesc::XMLString::release(&metricreturnsbelowrangemetric);
		xercesc::XMLString::release(&metricnumreturnswidth);
        xercesc::XMLString::release(&metricmeanwidth);
        xercesc::XMLString::release(&metricsumwidth);
        xercesc::XMLString::release(&metricmedianwidth);
        xercesc::XMLString::release(&metricmodewidth);
        xercesc::XMLString::release(&metricminwidth);
        xercesc::XMLString::release(&metricmaxwidth);
        xercesc::XMLString::release(&metricstddevwidth);
        xercesc::XMLString::release(&metricvariancewidth);
        xercesc::XMLString::release(&metricabsdeviationwidth);
        xercesc::XMLString::release(&metriccoefficientofvariationwidth);
        xercesc::XMLString::release(&metricpercentilewidth);
        xercesc::XMLString::release(&metricskewnesswidth);
        xercesc::XMLString::release(&metricpersonmodewidth);
        xercesc::XMLString::release(&metricpersonmedianwidth);
        xercesc::XMLString::release(&metrickurtosiswidth);
        xercesc::XMLString::release(&metricreturnsabovewidthmetric);
        xercesc::XMLString::release(&metricreturnsbelowwidthmetric);
        xercesc::XMLString::release(&metricNameXMLStr);
        xercesc::XMLString::release(&metricReturnXMLStr);
        xercesc::XMLString::release(&metricClassXMLStr);
        xercesc::XMLString::release(&metricMinNumReturnsXMLStr);
        xercesc::XMLString::release(&metricUpThresholdXMLStr);
        xercesc::XMLString::release(&metricLowThresholdXMLStr);
        xercesc::XMLString::release(&heightUpThresholdXMLStr);
        xercesc::XMLString::release(&heightLowThresholdXMLStr);
        xercesc::XMLString::release(&allXMLStr);
        xercesc::XMLString::release(&notFirstXMLStr);
        xercesc::XMLString::release(&firstXMLStr);
        xercesc::XMLString::release(&lastXMLStr);
        xercesc::XMLString::release(&firstLastXMLStr);
        xercesc::XMLString::release(&notGrdXMLStr);
        xercesc::XMLString::release(&vegXMLStr);
        xercesc::XMLString::release(&grdXMLStr);

        return metric;
    }

    SPDCalcMetrics::~SPDCalcMetrics()
    {

    }




    SPDCalcImageMetrics::SPDCalcImageMetrics(std::vector<SPDMetric*> *metrics, std::vector<std::string> *fieldNames)
    {
        this->metrics = metrics;
        this->fieldNames = fieldNames;
    }

    void SPDCalcImageMetrics::processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
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
            //std::cout << "Number of Pulses = " << pulses->size() << std::endl;
            for(std::vector<SPDMetric*>::iterator iterMetrics = metrics->begin(); iterMetrics != metrics->end(); ++iterMetrics)
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

    std::vector<std::string> SPDCalcImageMetrics::getImageBandDescriptions() throw(SPDProcessingException)
    {
        if(metrics->size() != fieldNames->size())
        {
            throw SPDProcessingException("The number of metrics and fieldnames needs to be the same.");
        }
        std::cout << "Executing for metrics: \n";
        std::vector<std::string> bandNames;
        for(std::vector<std::string>::iterator iterNames = fieldNames->begin(); iterNames != fieldNames->end(); ++iterNames)
        {
            bandNames.push_back(*iterNames);
            std::cout << *iterNames << std::endl;
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






    SPDCalcPolyMetrics::SPDCalcPolyMetrics(std::vector<SPDMetric*> *metrics, std::vector<std::string> *fieldNames)
    {
        this->metrics = metrics;
        this->fieldNames = fieldNames;
    }

    void SPDCalcPolyMetrics::processFeature(OGRFeature *inFeature, OGRFeature *outFeature, boost::uint_fast64_t fid, std::vector<SPDPulse*> *pulses, SPDFile *spdFile) throw(SPDProcessingException)
    {
        std::vector<SPDMetric*>::iterator iterMetrics = metrics->begin();
        std::vector<std::string>::iterator iterNames = fieldNames->begin();

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

    void SPDCalcPolyMetrics::processFeature(OGRFeature *inFeature, std::ofstream *outASCIIFile, boost::uint_fast64_t fid, std::vector<SPDPulse*> *pulses, SPDFile *spdFile) throw(SPDProcessingException)
    {
        (*outASCIIFile) << fid;

        OGRGeometry *geometry = inFeature->GetGeometryRef();

        double outVal = 0;
        for(std::vector<SPDMetric*>::iterator iterMetrics = metrics->begin(); iterMetrics != metrics->end(); ++iterMetrics)
        {
            outVal = (*iterMetrics)->calcValue(pulses, spdFile, geometry);
            (*outASCIIFile) << "," << outVal;
        }

        (*outASCIIFile) << std::endl;
    }

    void SPDCalcPolyMetrics::createOutputLayerDefinition(OGRLayer *outputLayer, OGRFeatureDefn *inFeatureDefn) throw(SPDProcessingException)
    {
        if(metrics->size() != fieldNames->size())
        {
            throw SPDProcessingException("The number of metrics and fieldnames needs to be the same.");
        }

        for(std::vector<std::string>::iterator iterNames = fieldNames->begin(); iterNames != fieldNames->end(); ++iterNames)
        {
            OGRFieldDefn shpField((*iterNames).c_str(), OFTReal);
            shpField.SetPrecision(10);
            if( outputLayer->CreateField( &shpField ) != OGRERR_NONE )
            {
                std::string message = std::string("Creating shapefile field ") + *iterNames + std::string(" has failed");
                throw SPDProcessingException(message);
            }
        }
    }

    void SPDCalcPolyMetrics::writeASCIIHeader(std::ofstream *outASCIIFile) throw(SPDProcessingException)
    {
        (*outASCIIFile) << "FID";
        for(std::vector<std::string>::iterator iterNames = fieldNames->begin(); iterNames != fieldNames->end(); ++iterNames)
        {
            (*outASCIIFile) << "," << (*iterNames);
        }
        (*outASCIIFile) << std::endl;
    }

    SPDCalcPolyMetrics::~SPDCalcPolyMetrics()
    {

    }

    
    
    
    
    
    SPDCalcZMedianVal::SPDCalcZMedianVal()
    {
        this->colMedianVals = new std::vector<double>();
    }
    
    void SPDCalcZMedianVal::processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException)
    {
        if(pulses->size() > 0)
        {
            std::vector<double> ptVals;
            
            for(std::vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
            {
                if((*iterPulses)->numberOfReturns > 0)
                {
                    for(std::vector<SPDPoint*>::iterator iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
                    {
                        ptVals.push_back((*iterPoints)->z);
                    }
                }
            }
            
            if(ptVals.size() > 0)
            {
                gsl_sort(&ptVals[0], 1, ptVals.size());
                double median = gsl_stats_median_from_sorted_data(&ptVals[0], 1, ptVals.size());
                colMedianVals->push_back(median);

            }
        }
    }
    
    double SPDCalcZMedianVal::getMedianMedianVal()
    {
        gsl_sort(&(*colMedianVals)[0], 1, colMedianVals->size());
        double median = gsl_stats_median_from_sorted_data(&(*colMedianVals)[0], 1, colMedianVals->size());
        return median;
    }
    
    SPDCalcZMedianVal::~SPDCalcZMedianVal()
    {
        delete this->colMedianVals;
    }
    
    
    
    
    
    
    
    
    
    
    

}


