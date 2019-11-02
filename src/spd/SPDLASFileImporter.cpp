/*
 *  SPDLASFileImporter.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 02/12/2010.
 *  Copyright 2010 SPDLib. All rights reserved.
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

#include "spd/SPDLASFileImporter.h"

namespace spdlib
{
    std::string getWKTfromLAS(LASheader *header)
    {
        /**
         Get EPSG projection code from LAS file header and convert to WKT string.

         TODO: Needs testing with a range of coordinate systems. Within lasinfo a number of
         checks for differnent keys are used. Need to confirm only checking for key id 3072 is
         sufficient and if not how best to create a WKT file from multiple keys.

         */
        bool foundProjection = false;
        char *pszWKT = NULL;

        for (int j = 0; j < header->vlr_geo_keys->number_of_keys; j++)
        {
            if(header->vlr_geo_key_entries[j].key_id == 3072)
            {

                OGRSpatialReference lasSRS;
                if(lasSRS.importFromEPSG(header->vlr_geo_key_entries[j].value_offset) == 0)
                {
                    lasSRS.exportToWkt(&pszWKT);
                    foundProjection = true;
                }
                else
                {
                    std::cerr << "  WARNING: Could not get coordinate system from code: " << header->vlr_geo_key_entries[j].value_offset << std::endl;
                }
            }
        }

        if(foundProjection)
        {
            return (std::string) pszWKT;
        }
        else
        {
            std::cerr << "WARNING: Could not get projection from LAS file. Not setting." << std::endl;
            return "";
        }

    }

    SPDLASFileImporter::SPDLASFileImporter(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold):SPDDataImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold)
    {

    }

    SPDDataImporter* SPDLASFileImporter::getInstance(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshol)
    {
        this->strictPulses = false;
        return new SPDLASFileImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
    }

    std::list<SPDPulse*>* SPDLASFileImporter::readAllDataToList(std::string inputFile, SPDFile *spdFile)
    {
        /* This has been removed when updated to LASlib.
         * It is only required when exporting to a format which needs the number
         * of out points (export->needNumOutPts()).
         * Will add back and update to use LASlib if it causes problems.
         */
        throw SPDIOException("Reading LAS to a list is not currently supported");
    }

    std::vector<SPDPulse*>* SPDLASFileImporter::readAllDataToVector(std::string inputFile, SPDFile *spdFile)
    {
        /**
         * Class to read LAS format input data to a vector of SPDPulses.
         *
         * Uses LASlib to read data.
         *
         * For discrete return data (LAS 1.0 - 1.2) pulses are constructed using first and last return
         * and stored without a digitised waveform.
         *
         * For full waveform data (LAS 1.3) the digitised waveform and additional pulse information is
         * stored for points where it is available.
         *
         */
        SPDPulseUtils pulseUtils;
        std::vector<SPDPulse*> *pulses = new std::vector<SPDPulse*>();
        boost::uint_fast64_t numOfPulses = 0;
        boost::uint_fast64_t numOfPoints = 0;

        double xMin = 0;
        double xMax = 0;
        double yMin = 0;
        double yMax = 0;
        double zMin = 0;
        double zMax = 0;
        bool first = true;
        bool firstZ = true;
        bool lasHeaderInfoSet = false;

        classWarningGiven = false;
        bool haveWaveforms = false;
        bool pulseHasWaveform = false;
        unsigned int numReversedWaveformVectors = 0;

        try
        {
            // Open LAS file
            LASreadOpener lasreadopener;
            LASreader* lasreader = lasreadopener.open(inputFile.c_str());

            if(lasreader != 0)
            {
                // Get header
                LASheader *header = &lasreader->header;

                spdFile->setFileSignature(header->file_signature);
                spdFile->setSystemIdentifier(header->system_identifier);

                if(spdFile->getSpatialReference() == "")
                {
                    // Get WKT string from LAS header
                    spdFile->setSpatialReference(getWKTfromLAS(header));
                }

                if(convertCoords)
                {
                    this->initCoordinateSystemTransformation(spdFile);
                }

                // Check for waveforms
                LASwaveform13reader *laswaveformreader = lasreadopener.open_waveform13(&lasreader->header);
                if (laswaveformreader != 0)
                {
                    haveWaveforms = true;
                    std::cout << "Have waveform data" << std::endl;
                }
                else
                {
                    std::cout << "No waveform data" << std::endl;
                }

                pulses->reserve(header->number_of_point_records);

                boost::uint_fast64_t reportedNumOfPts = header->number_of_point_records;
                boost::uint_fast64_t feedback = reportedNumOfPts/10;
                unsigned int feedbackCounter = 0;

                SPDPoint *spdPt = NULL;
                SPDPulse *spdPulse = NULL;

                boost::uint_fast16_t nPtIdx = 0;
                double x0 = 0.0; // First point in pulse
                double y0 = 0.0;
                double z0 = 0.0;
                double x1 = 0.0; // Last point in pulse or last sample in pulse (if waveform data is availabe)
                double y1 = 0.0;
                double z1 = 0.0;

                std::cout << "Started (Read Data) ." << std::flush;
                while (lasreader->read_point())
                {
                    unsigned int lasindex = lasreader->point.wavepacket.getIndex();
                    if(lasindex)
                    {
                        // Convert bin spacing to nano seconds
                        if(!lasHeaderInfoSet)
                        {
                            spdFile->setTemporalBinSpacing(lasreader->header.vlr_wave_packet_descr[lasindex]->getTemporalSpacing()/1000);

                            lasHeaderInfoSet = true;
                        }
                    }
                    //std::cout << numOfPoints << std::endl;
                    if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
                    {
                        std::cout << "." << feedbackCounter << "." << std::flush;
                        feedbackCounter += 10;
                    }

                    if(lasreader->point.get_return_number() <= 1)
                    {
                        pulseHasWaveform = false;

                        spdPt = this->createSPDPoint(lasreader->point);
                        ++numOfPoints;

                        if(firstZ)
                        {
                            zMin = spdPt->z;
                            zMax = spdPt->z;
                            firstZ = false;
                        }
                        else
                        {
                            if(spdPt->z < zMin)
                            {
                                zMin = spdPt->z;
                            }
                            else if(spdPt->z > zMax)
                            {
                                zMax = spdPt->z;
                            }
                        }

                        /* Set pulse origin based on first point
                           If there is waveform data will use this
                           instead */
                        x0 = spdPt->x;
                        y0 = spdPt->y;
                        z0 = spdPt->z;

                        spdPulse = new SPDPulse();
                        pulseUtils.initSPDPulse(spdPulse);
                        spdPulse->pulseID = numOfPulses;
                        spdPulse->numberOfReturns = lasreader->point.get_number_of_returns();
                        if(spdPulse->numberOfReturns == 0)
                        {
                            spdPulse->numberOfReturns = 1;
                        }
                        if(spdPulse->numberOfReturns > 0)
                        {
                            spdPulse->pts->reserve(spdPulse->numberOfReturns);
                        }

                        /* If first return (only copy for one return) and file has waveform data
                         copy waveform to SPD Pulse
                         */
                        if(haveWaveforms)
                        {
                            if(laswaveformreader->read_waveform(&lasreader->point))
                            {
                                if(laswaveformreader->nsamples > 0)
                                {
                                    pulseHasWaveform = true;

                                    // Set number of samples
                                    spdPulse->numOfReceivedBins = laswaveformreader->nsamples;

                                    // Get pulse time
                                    double pulse_duration = spdPulse->numOfReceivedBins*(lasreader->header.vlr_wave_packet_descr[lasindex]->getTemporalSpacing());

                                    /* Get the offset (in ps) from the first digitized value
                                    to the location within the waveform packet that the associated
                                    return pulse was detected.*/
                                    double location = lasreader->point.wavepacket.getLocation();

                                    // Save to point (in ns)

                                    // Set pulse GPS time (ns)
                                    spdPulse->gpsTime = spdPt->gpsTime - spdPt->waveformOffset;

                                    /* Set the start location of the return pulse
                                    This is calculated as the location of the first return
                                    minus the time offset multiplied by XYZ(t) which is a vector
                                    away from the laser origin */

                                    spdPulse->x0 = x0 - location*laswaveformreader->XYZt[0];
                                    spdPulse->y0 = y0 - location*laswaveformreader->XYZt[1];
                                    spdPulse->z0 = z0 - location*laswaveformreader->XYZt[2];

                                    /* Get the end location of the return pulse
                                    This is calculated as start location of the pulse
                                    plus the pulse duration multipled by XYZ(t)
                                    It is only used to get the azimuth and zenith angle
                                    of the pulse */
                                    x1 = spdPulse->x0 + pulse_duration*laswaveformreader->XYZt[0];
                                    y1 = spdPulse->y0 + pulse_duration*laswaveformreader->XYZt[1];
                                    z1 = spdPulse->z0 + pulse_duration*laswaveformreader->XYZt[2];

                                    /* Check if the first sample of the pulse is higher than the last
                                    If it is this probably means the direction vector (XYZt) is going
                                    in the oposite direction (towards the laser origin) so need
                                    to recalculate the location of the first and last sample
                                    I think this problem is specific to LAS 1.3 files exported
                                    with earlier versions of Leica's ALSPP software.
                                    */
                                    if(spdPulse->z0 < z1)
                                    {
                                        // Only print warning once.
                                        if(numReversedWaveformVectors == 0)
                                        {
                                            std::cout << "\nWARNING: Reversing direction vector (XYZt)." << std::endl;
                                        }
                                        ++numReversedWaveformVectors;
                                        spdPulse->x0 = x0 + location*laswaveformreader->XYZt[0];
                                        spdPulse->y0 = y0 + location*laswaveformreader->XYZt[1];
                                        spdPulse->z0 = z0 + location*laswaveformreader->XYZt[2];

                                        x1 = spdPulse->x0 - pulse_duration*laswaveformreader->XYZt[0];
                                        y1 = spdPulse->y0 - pulse_duration*laswaveformreader->XYZt[1];
                                        z1 = spdPulse->z0 - pulse_duration*laswaveformreader->XYZt[2];
                                    }
                                    else if(numReversedWaveformVectors > 0)
                                    {
                                        throw SPDIOException("Direction vector (XYZt) is not the same direction for all pulses in file");
                                    }

                                    // Set intensity gain and offset
                                    spdPulse->receiveWaveGain = lasreader->header.vlr_wave_packet_descr[lasindex]->getDigitizerGain();
                                    spdPulse->receiveWaveOffset = lasreader->header.vlr_wave_packet_descr[lasindex]->getDigitizerOffset();

                                    // Copy waveform data
                                    spdPulse->received = new boost::uint_fast32_t[spdPulse->numOfReceivedBins];
                                    for(unsigned int s = 0; s < spdPulse->numOfReceivedBins; s++)
                                    {
                                        spdPulse->received[s] = (unsigned int)laswaveformreader->samples[s];
                                    }
                                }
                            }
                        }

                        if(lasreader->point.get_return_number() == 0)
                        {
                            nPtIdx = 1;
                        }
                        else
                        {
                            nPtIdx = 2;
                        }

                        spdPulse->pts->push_back(spdPt);

                        for(boost::uint_fast16_t i = 0; i < (spdPulse->numberOfReturns-1); ++i)
                        {
                            if(lasreader->read_point())
                            {
                                if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
                                {
                                    std::cout << "." << feedbackCounter << "." << std::flush;
                                    feedbackCounter += 10;
                                }

                                if(nPtIdx != lasreader->point.get_return_number())
                                {
                                    if(this->strictPulses)
                                    {
                                        std::cerr << "\nThe return number was: " << lasreader->point.get_return_number() << std::endl;
                                        std::cerr << "The next return number should have been: " << nPtIdx << std::endl;

                                        throw SPDIOException("Error in point numbering when building pulses.");
                                    }
                                    else
                                    {
                                        std::cerr << "WARNING: Pulse was written as incompleted pulse.\n";

                                    }
                                    spdPulse->numberOfReturns = i+1;
                                    break;
                                }

                                spdPt = this->createSPDPoint(lasreader->point);

                                if(spdPt->z < zMin)
                                {
                                    zMin = spdPt->z;
                                }
                                else if(spdPt->z > zMax)
                                {
                                    zMax = spdPt->z;
                                }

                                /* If waveform data is available x1, y1, z1
                                will be the location of the last sample in the
                                digitsed waveform.
                                Otherwise will use last point */
                                if(!pulseHasWaveform)
                                {
                                    x1 = spdPt->x;
                                    y1 = spdPt->y;
                                    z1 = spdPt->z;
                                }

                                /* If waveform data is available, set offset for
                                each point to start of wavepacket (in ns) */
                                if(pulseHasWaveform)
                                {
                                    spdPt->waveformOffset = lasreader->point.wavepacket.getLocation()*1E3;
                                }
                                spdPulse->pts->push_back(spdPt);

                                ++numOfPoints;
                                ++nPtIdx;
                            }
                            else
                            {
                                std::cerr << "\nWarning: The file ended unexpectedly.\n";
                                std::cerr << "Expected " << spdPulse->numberOfReturns << " but only found " << i + 1 << " returns" << std::endl;
                                spdPulse->numberOfReturns = i+1;
                                if(this->strictPulses)
                                {
                                    throw SPDIOException("Unexpected end to the file.");
                                }
                                break;
                            }
                        }
                        ++numOfPulses;

                        if(lasreader->point.get_edge_of_flight_line() == 1)
                        {
                            spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                        }
                        else
                        {
                            spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                        }

                        if(lasreader->point.get_scan_direction_flag() == 1)
                        {
                            spdPulse->scanDirectionFlag = SPD_POSITIVE;
                        }
                        else
                        {
                            spdPulse->scanDirectionFlag = SPD_NEGATIVE;
                        }

                        if((spdPulse->numberOfReturns > 1) | pulseHasWaveform)
                        {

                            double range = 0;
                            double zenith = 0;
                            double azimuth = 0;

                            SPDConvertToSpherical(x0, y0, z0, x1, y1, z1, &zenith, &azimuth, &range);

                            spdPulse->zenith = zenith;
                            spdPulse->azimuth = azimuth;

                            if(spdPulse->azimuth < 0)
                            {
                                std::cout << spdPulse->azimuth << std::endl;
                                spdPulse->azimuth = spdPulse->azimuth + M_PI * 2;
                            }
                        }
                        else
                        {
                            spdPulse->zenith = 0.0;
                            spdPulse->azimuth = 0.0;
                        }
                        spdPulse->user = lasreader->point.get_scan_angle_rank() + 90;

                        spdPulse->sourceID = lasreader->point.get_point_source_ID();

                        if(indexCoords == SPD_FIRST_RETURN)
                        {
                            spdPulse->xIdx = spdPulse->pts->front()->x;
                            spdPulse->yIdx = spdPulse->pts->front()->y;
                        }
                        else if(indexCoords == SPD_LAST_RETURN)
                        {
                            spdPulse->xIdx = spdPulse->pts->back()->x;
                            spdPulse->yIdx = spdPulse->pts->back()->y;
                        }
                        else if(indexCoords == SPD_START_OF_RECEIVED_WAVEFORM)
                        {
                            if(pulseHasWaveform)
                            {
                                spdPulse->xIdx = spdPulse->x0;
                                spdPulse->yIdx = spdPulse->y0;
                            }
                            else
                            {
                                spdPulse->xIdx = x0;
                                spdPulse->yIdx = y0;
                            }
                        }
                        else if(indexCoords == SPD_END_OF_RECEIVED_WAVEFORM)
                        {
                            spdPulse->xIdx = x1;
                            spdPulse->yIdx = y1;
                        }
                        else
                        {
                            throw SPDIOException("Indexing type unsupported");
                        }

                        if(first)
                        {
                            xMin = spdPulse->xIdx;
                            xMax = spdPulse->xIdx;
                            yMin = spdPulse->yIdx;
                            yMax = spdPulse->yIdx;
                            first = false;
                        }
                        else
                        {
                            if(spdPulse->xIdx < xMin)
                            {
                                xMin = spdPulse->xIdx;
                            }
                            else if(spdPulse->xIdx > xMax)
                            {
                                xMax = spdPulse->xIdx;
                            }

                            if(spdPulse->yIdx < yMin)
                            {
                                yMin = spdPulse->yIdx;
                            }
                            else if(spdPulse->yIdx > yMax)
                            {
                                yMax = spdPulse->yIdx;
                            }
                        }

                        pulses->push_back(spdPulse);
                    }
                    else
                    {
                        //std::cerr << "p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                        //std::cerr << "p.GetNumberOfReturns() = " << p.GetNumberOfReturns() << std::endl;
                        std::cerr << "Warning: Point ignored. It is the first in pulse but has a return number greater than 1.\n";
                    }
                }

                lasreader->close();
                std::cout << ". Complete\n";
                spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
                if(convertCoords)
                {
                    spdFile->setSpatialReference(outputProjWKT);
                }
                spdFile->setNumberOfPulses(numOfPulses);
                spdFile->setNumberOfPoints(numOfPoints);
                spdFile->setDiscretePtDefined(SPD_TRUE);
                spdFile->setDecomposedPtDefined(SPD_FALSE);
                spdFile->setTransWaveformDefined(SPD_FALSE);
                if(haveWaveforms)
                {
                    spdFile->setReceiveWaveformDefined(SPD_TRUE);
                    spdFile->setOriginDefined(SPD_TRUE);
                }
                else
                {
                    spdFile->setReceiveWaveformDefined(SPD_FALSE);
                    spdFile->setOriginDefined(SPD_FALSE);
                }
            }
            else
            {
                throw SPDIOException("LAS file could not be opened.");
            }
        }
        catch (SPDIOException &e)
        {
            throw e;
        }
        catch(std::invalid_argument &e)
        {
            throw SPDIOException(e.what());
        }
        catch(std::runtime_error &e)
        {
            throw SPDIOException(e.what());
        }

        return pulses;
    }

    void SPDLASFileImporter::readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor)
    {
        /**
         * Class to read LAS format input data and process. Used for UPD import.
         *
         * Uses LASlib to read data.
         *
         * For discrete return data (LAS 1.0 - 1.2) pulses are constructed using first and last return
         * and stored without a digitised waveform.
         *
         * For full waveform data (LAS 1.3) the digitised waveform and additional pulse information is
         * stored for points where it is available.
         */

        SPDPulseUtils pulseUtils;
        boost::uint_fast64_t numOfPulses = 0;
        boost::uint_fast64_t numOfPoints = 0;

        double xMin = 0;
        double xMax = 0;
        double yMin = 0;
        double yMax = 0;
        double zMin = 0;
        double zMax = 0;
        bool first = true;
        bool firstZ = true;
        bool lasHeaderInfoSet = false;

        classWarningGiven = false;
        bool haveWaveforms = false;
        bool pulseHasWaveform = false;
        unsigned int numReversedWaveformVectors = 0;

        try
        {
            // Open LAS file
            LASreadOpener lasreadopener;
            LASreader* lasreader = lasreadopener.open(inputFile.c_str());

            if(lasreader != 0)
            {
                // Get header
                LASheader *header = &lasreader->header;

                spdFile->setFileSignature(header->file_signature);
                spdFile->setSystemIdentifier(header->system_identifier);

                if(spdFile->getSpatialReference() == "")
                {
                    // Get WKT string from LAS header
                    spdFile->setSpatialReference(getWKTfromLAS(header));
                }

                if(convertCoords)
                {
                    this->initCoordinateSystemTransformation(spdFile);
                }

                // Check for waveforms
                LASwaveform13reader *laswaveformreader = lasreadopener.open_waveform13(&lasreader->header);
                if (laswaveformreader != 0)
                {
                    haveWaveforms = true;
                    std::cout << "Have waveform data" << std::endl;
                }
                else
                {
                    std::cout << "No waveform data" << std::endl;
                }

                boost::uint_fast64_t reportedNumOfPts = header->number_of_point_records;
                boost::uint_fast64_t feedback = reportedNumOfPts/10;
                unsigned int feedbackCounter = 0;

                SPDPoint *spdPt = NULL;
                SPDPulse *spdPulse = NULL;

                boost::uint_fast16_t nPtIdx = 0;
                double x0 = 0.0; // First point in pulse
                double y0 = 0.0;
                double z0 = 0.0;
                double x1 = 0.0; // Last point in pulse or last sample in pulse (if waveform data is availabe)
                double y1 = 0.0;
                double z1 = 0.0;

                std::cout << "Started (Read Data) ." << std::flush;
                while (lasreader->read_point())
                {
                    unsigned int lasindex = lasreader->point.wavepacket.getIndex();
                    if(lasindex)
                    {
                        // Convert bin spacing to nano seconds
                        if(!lasHeaderInfoSet)
                        {
                            spdFile->setTemporalBinSpacing(lasreader->header.vlr_wave_packet_descr[lasindex]->getTemporalSpacing()/1000);

                            lasHeaderInfoSet = true;
                        }
                    }
                    //std::cout << numOfPoints << std::endl;
                    if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
                    {
                        std::cout << "." << feedbackCounter << "." << std::flush;
                        feedbackCounter += 10;
                    }

                    if(lasreader->point.get_return_number() <= 1)
                    {
                        pulseHasWaveform = false;

                        spdPt = this->createSPDPoint(lasreader->point);
                        ++numOfPoints;

                        if(firstZ)
                        {
                            zMin = spdPt->z;
                            zMax = spdPt->z;
                            firstZ = false;
                        }
                        else
                        {
                            if(spdPt->z < zMin)
                            {
                                zMin = spdPt->z;
                            }
                            else if(spdPt->z > zMax)
                            {
                                zMax = spdPt->z;
                            }
                        }

                        /* Set pulse origin based on first point
                           If there is waveform data will use this
                           instead */
                        x0 = spdPt->x;
                        y0 = spdPt->y;
                        z0 = spdPt->z;

                        spdPulse = new SPDPulse();
                        pulseUtils.initSPDPulse(spdPulse);
                        spdPulse->pulseID = numOfPulses;
                        spdPulse->numberOfReturns = lasreader->point.get_number_of_returns();
                        if(spdPulse->numberOfReturns == 0)
                        {
                            spdPulse->numberOfReturns = 1;
                        }
                        if(spdPulse->numberOfReturns > 0)
                        {
                            spdPulse->pts->reserve(spdPulse->numberOfReturns);
                        }

                        /* If first return (only copy for one return) and file has waveform data
                         copy waveform to SPD Pulse
                         */
                        if(haveWaveforms)
                        {
                            if(laswaveformreader->read_waveform(&lasreader->point))
                            {
                                if(laswaveformreader->nsamples > 0)
                                {
                                    pulseHasWaveform = true;

                                    // Set number of samples
                                    spdPulse->numOfReceivedBins = laswaveformreader->nsamples;

                                    // Get pulse time
                                    double pulse_duration = spdPulse->numOfReceivedBins*(lasreader->header.vlr_wave_packet_descr[lasindex]->getTemporalSpacing());

                                    /* Get the offset (in ps) from the first digitized value
                                    to the location within the waveform packet that the associated
                                    return pulse was detected.*/
                                    double location = lasreader->point.wavepacket.getLocation();

                                    // Save to point (in ns)

                                    // Set pulse GPS time (ns)
                                    spdPulse->gpsTime = spdPt->gpsTime - spdPt->waveformOffset;

                                    /* Set the start location of the return pulse
                                    This is calculated as the location of the first return
                                    minus the time offset multiplied by XYZ(t) which is a vector
                                    away from the laser origin */

                                    spdPulse->x0 = x0 - location*laswaveformreader->XYZt[0];
                                    spdPulse->y0 = y0 - location*laswaveformreader->XYZt[1];
                                    spdPulse->z0 = z0 - location*laswaveformreader->XYZt[2];

                                    /* Get the end location of the return pulse
                                    This is calculated as start location of the pulse
                                    plus the pulse duration multipled by XYZ(t)
                                    It is only used to get the azimuth and zenith angle
                                    of the pulse */
                                    x1 = spdPulse->x0 + pulse_duration*laswaveformreader->XYZt[0];
                                    y1 = spdPulse->y0 + pulse_duration*laswaveformreader->XYZt[1];
                                    z1 = spdPulse->z0 + pulse_duration*laswaveformreader->XYZt[2];

                                    /* Check if the first sample of the pulse is higher than the last
                                    If it is this probably means the direction vector (XYZt) is going
                                    in the oposite direction (towards the laser origin) so need
                                    to recalculate the location of the first and last sample
                                    I think this problem is specific to LAS 1.3 files exported
                                    with earlier versions of Leica's ALSPP software.
                                    */
                                    if(spdPulse->z0 < z1)
                                    {
                                        // Only print warning once.
                                        if(numReversedWaveformVectors == 0)
                                        {
                                            std::cout << "\nWARNING: Reversing direction vector (XYZt)." << std::endl;
                                        }
                                        ++numReversedWaveformVectors;
                                        spdPulse->x0 = x0 + location*laswaveformreader->XYZt[0];
                                        spdPulse->y0 = y0 + location*laswaveformreader->XYZt[1];
                                        spdPulse->z0 = z0 + location*laswaveformreader->XYZt[2];

                                        x1 = spdPulse->x0 - pulse_duration*laswaveformreader->XYZt[0];
                                        y1 = spdPulse->y0 - pulse_duration*laswaveformreader->XYZt[1];
                                        z1 = spdPulse->z0 - pulse_duration*laswaveformreader->XYZt[2];
                                    }
                                    else if(numReversedWaveformVectors > 0)
                                    {
                                        throw SPDIOException("Direction vector (XYZt) is not the same direction for all pulses in file");
                                    }

                                    // Set intensity gain and offset
                                    spdPulse->receiveWaveGain = lasreader->header.vlr_wave_packet_descr[lasindex]->getDigitizerGain();
                                    spdPulse->receiveWaveOffset = lasreader->header.vlr_wave_packet_descr[lasindex]->getDigitizerOffset();

                                    // Copy waveform data
                                    spdPulse->received = new boost::uint_fast32_t[spdPulse->numOfReceivedBins];
                                    for(unsigned int s = 0; s < spdPulse->numOfReceivedBins; s++)
                                    {
                                        spdPulse->received[s] = (unsigned int)laswaveformreader->samples[s];
                                    }
                                }
                            }
                        }

                        if(lasreader->point.get_return_number() == 0)
                        {
                            nPtIdx = 1;
                        }
                        else
                        {
                            nPtIdx = 2;
                        }

                        spdPulse->pts->push_back(spdPt);

                        for(boost::uint_fast16_t i = 0; i < (spdPulse->numberOfReturns-1); ++i)
                        {
                            if(lasreader->read_point())
                            {
                                if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
                                {
                                    std::cout << "." << feedbackCounter << "." << std::flush;
                                    feedbackCounter += 10;
                                }

                                if(nPtIdx != lasreader->point.get_return_number())
                                {
                                    if(this->strictPulses)
                                    {
                                        std::cerr << "\nThe return number was: " << lasreader->point.get_return_number() << std::endl;
                                        std::cerr << "The next return number should have been: " << nPtIdx << std::endl;

                                        throw SPDIOException("Error in point numbering when building pulses.");
                                    }
                                    else
                                    {
                                        std::cerr << "WARNING: Pulse was written as incompleted pulse.\n";
                                    }
                                    spdPulse->numberOfReturns = i+1;
                                    //throw SPDIOException("Error in point numbering when building pulses.");
                                    break;
                                }

                                spdPt = this->createSPDPoint(lasreader->point);

                                if(spdPt->z < zMin)
                                {
                                    zMin = spdPt->z;
                                }
                                else if(spdPt->z > zMax)
                                {
                                    zMax = spdPt->z;
                                }

                                /* If waveform data is available x1, y1, z1
                                will be the location of the last sample in the
                                digitsed waveform.
                                Otherwise will use last point */
                                if(!pulseHasWaveform)
                                {
                                    x1 = spdPt->x;
                                    y1 = spdPt->y;
                                    z1 = spdPt->z;
                                }

                                /* If waveform data is available, set offset for
                                each point to start of wavepacket (in ns) */
                                if(pulseHasWaveform)
                                {
                                    spdPt->waveformOffset = lasreader->point.wavepacket.getLocation()*1E3;
                                }
                                spdPulse->pts->push_back(spdPt);

                                ++numOfPoints;
                                ++nPtIdx;
                            }
                            else
                            {
                                std::cerr << "\nWarning: The file ended unexpectedly.\n";
                                std::cerr << "Expected " << spdPulse->numberOfReturns << " but only found " << i + 1 << " returns" << std::endl;
                                spdPulse->numberOfReturns = i+1;
                                if(this->strictPulses)
                                {
                                    throw SPDIOException("Unexpected end to the file.");
                                }
                                break;
                            }
                        }
                        ++numOfPulses;

                        if(lasreader->point.get_edge_of_flight_line() == 1)
                        {
                            spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                        }
                        else
                        {
                            spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                        }

                        if(lasreader->point.get_scan_direction_flag() == 1)
                        {
                            spdPulse->scanDirectionFlag = SPD_POSITIVE;
                        }
                        else
                        {
                            spdPulse->scanDirectionFlag = SPD_NEGATIVE;
                        }

                        if((spdPulse->numberOfReturns > 1) | pulseHasWaveform)
                        {

                            double range = 0;
                            double zenith = 0;
                            double azimuth = 0;

                            SPDConvertToSpherical(x0, y0, z0, x1, y1, z1, &zenith, &azimuth, &range);

                            spdPulse->zenith = zenith;
                            spdPulse->azimuth = azimuth;

                            if(spdPulse->azimuth < 0)
                            {
                                std::cout << spdPulse->azimuth << std::endl;
                                spdPulse->azimuth = spdPulse->azimuth + M_PI * 2;
                            }
                        }
                        else
                        {
                            spdPulse->zenith = 0.0;
                            spdPulse->azimuth = 0.0;
                        }
                        spdPulse->user = lasreader->point.get_scan_angle_rank() + 90;

                        spdPulse->sourceID = lasreader->point.get_point_source_ID();

                        if(indexCoords == SPD_FIRST_RETURN)
                        {
                            spdPulse->xIdx = spdPulse->pts->front()->x;
                            spdPulse->yIdx = spdPulse->pts->front()->y;
                        }
                        else if(indexCoords == SPD_LAST_RETURN)
                        {
                            spdPulse->xIdx = spdPulse->pts->back()->x;
                            spdPulse->yIdx = spdPulse->pts->back()->y;
                        }
                        else if(indexCoords == SPD_START_OF_RECEIVED_WAVEFORM)
                        {
                            if(pulseHasWaveform)
                            {
                                spdPulse->xIdx = spdPulse->x0;
                                spdPulse->yIdx = spdPulse->y0;
                            }
                            else
                            {
                                spdPulse->xIdx = x0;
                                spdPulse->yIdx = y0;
                            }
                        }
                        else if(indexCoords == SPD_END_OF_RECEIVED_WAVEFORM)
                        {
                            spdPulse->xIdx = x1;
                            spdPulse->yIdx = y1;
                        }
                        else
                        {
                            throw SPDIOException("Indexing type unsupported");
                        }

                        if(first)
                        {
                            xMin = spdPulse->xIdx;
                            xMax = spdPulse->xIdx;
                            yMin = spdPulse->yIdx;
                            yMax = spdPulse->yIdx;
                            first = false;
                        }
                        else
                        {
                            if(spdPulse->xIdx < xMin)
                            {
                                xMin = spdPulse->xIdx;
                            }
                            else if(spdPulse->xIdx > xMax)
                            {
                                xMax = spdPulse->xIdx;
                            }

                            if(spdPulse->yIdx < yMin)
                            {
                                yMin = spdPulse->yIdx;
                            }
                            else if(spdPulse->yIdx > yMax)
                            {
                                yMax = spdPulse->yIdx;
                            }
                        }

                        processor->processImportedPulse(spdFile, spdPulse);
                    }
                    else
                    {
                        //std::cerr << "p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                        //std::cerr << "p.GetNumberOfReturns() = " << p.GetNumberOfReturns() << std::endl;
                        std::cerr << "Warning: Point ignored. It is the first in pulse but has a return number greater than 1.\n";
                    }
                }

                lasreader->close();
                std::cout << ". Complete\n";
                spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
                if(convertCoords)
                {
                    spdFile->setSpatialReference(outputProjWKT);
                }
                spdFile->setNumberOfPulses(numOfPulses);
                spdFile->setNumberOfPoints(numOfPoints);
                spdFile->setDiscretePtDefined(SPD_TRUE);
                spdFile->setDecomposedPtDefined(SPD_FALSE);
                spdFile->setTransWaveformDefined(SPD_FALSE);
                if(haveWaveforms)
                {
                    spdFile->setReceiveWaveformDefined(SPD_TRUE);
                    spdFile->setOriginDefined(SPD_TRUE);
                }
                else
                {
                    spdFile->setReceiveWaveformDefined(SPD_FALSE);
                    spdFile->setOriginDefined(SPD_FALSE);
                }
            }
            else
            {
                throw SPDIOException("LAS file could not be opened.");
            }
        }
        catch (SPDIOException &e)
        {
            throw e;
        }
        catch(std::invalid_argument &e)
        {
            throw SPDIOException(e.what());
        }
        catch(std::runtime_error &e)
        {
            throw SPDIOException(e.what());
        }
    }

    bool SPDLASFileImporter::isFileType(std::string fileType)
    {
        if(fileType == "LAS")
        {
            return true;
        }
        return false;
    }

    void SPDLASFileImporter::readHeaderInfo(std::string, SPDFile*) 
    {
        // No Header to Read..
    }

    SPDPoint* SPDLASFileImporter::createSPDPoint(LASpoint const& pt)
    {
        try
        {
            SPDPointUtils spdPtUtils;
            SPDPoint *spdPt = new SPDPoint();
            spdPtUtils.initSPDPoint(spdPt);
            // Get scaled values of x, y and z
            double x = pt.get_x();
            double y = pt.get_y();
            double z = pt.get_z();

            if(convertCoords)
            {
                this->transformCoordinateSystem(&x, &y, &z);
            }

            spdPt->x = x;
            spdPt->y = y;
            spdPt->z = z;
            spdPt->amplitudeReturn = pt.get_intensity();
            spdPt->user = pt.get_user_data();

            unsigned int lasClass = pt.get_classification();

            switch (lasClass)
            {
                case 0:
                    spdPt->classification = SPD_CREATED;
                    break;
                case 1:
                    spdPt->classification = SPD_UNCLASSIFIED;
                    break;
                case 2:
                    spdPt->classification = SPD_GROUND;
                    break;
                case 3:
                    spdPt->classification = SPD_LOW_VEGETATION;
                    break;
                case 4:
                    spdPt->classification = SPD_MEDIUM_VEGETATION;
                    break;
                case 5:
                    spdPt->classification = SPD_HIGH_VEGETATION;
                    break;
                case 6:
                    spdPt->classification = SPD_BUILDING;
                    break;
                case 7:
                    spdPt->classification = SPD_CREATED;
                    spdPt->lowPoint = SPD_TRUE;
                    break;
                case 8:
                    spdPt->classification = SPD_CREATED;
                    spdPt->modelKeyPoint = SPD_TRUE;
                    break;
                case 9:
                    spdPt->classification = SPD_WATER;
                    break;
                case 12:
                    spdPt->classification = SPD_CREATED;
                    spdPt->overlap = SPD_TRUE;
                    break;
                default:
                    spdPt->classification = SPD_CREATED;
                    if(!classWarningGiven)
                    {
                        std::cerr << "\nWARNING: The class ID " << lasClass<< " was not recognised - check the classes points were allocated too." << std::endl;
                        classWarningGiven = true;
                    }
                    break;
            }

            // Get array of RBG values (of type U16 in LASlib typedef)
            const unsigned short *rgb = pt.get_rgb();

            spdPt->red = rgb[0];
            spdPt->green = rgb[1];
            spdPt->blue = rgb[2];

            spdPt->returnID = pt.get_return_number();
            // Convert GPS time in s to ns for SPDLib (stored as 64 bit float)
            spdPt->gpsTime = pt.get_gps_time()*1E9;

            return spdPt;
        }
        catch (SPDIOException &e)
        {
            throw e;
        }
        catch(std::invalid_argument &e)
        {
            throw SPDIOException(e.what());
        }
        catch(std::runtime_error &e)
        {
            throw SPDIOException(e.what());
        }
    }

    SPDLASFileImporter::~SPDLASFileImporter()
    {

    }

    SPDLASFileImporterStrictPulses::SPDLASFileImporterStrictPulses(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold):SPDDataImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold)
    {

    }



    SPDDataImporter* SPDLASFileImporterStrictPulses::getInstance(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold)
    {
        this->lasDataImporter = new SPDLASFileImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
        this->lasDataImporter->setStrict(true);
        return (SPDDataImporter*) this->lasDataImporter;
    }

    std::list<SPDPulse*>* SPDLASFileImporterStrictPulses::readAllDataToList(std::string inputFile, SPDFile *spdFile)
    {
        return this->lasDataImporter->readAllDataToList(inputFile, spdFile);
    }
		std::vector<SPDPulse*>* readAllDataToVector(std::string inputFile, SPDFile *spdFile);
		void readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) ;

    std::vector<SPDPulse*>* SPDLASFileImporterStrictPulses::readAllDataToVector(std::string inputFile, SPDFile *spdFile)
    {
        return this->lasDataImporter->readAllDataToVector(inputFile, spdFile);
    }

    void SPDLASFileImporterStrictPulses::readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor)
    {
        this->lasDataImporter->readAndProcessAllData(inputFile, spdFile, processor);
    }

    bool SPDLASFileImporterStrictPulses::isFileType(std::string fileType)
    {
        if(fileType == "LASSTRICT")
        {
            return true;
        }
        return false;
    }

    void SPDLASFileImporterStrictPulses::readHeaderInfo(std::string, SPDFile*) 
    {
        // No Header to Read..
    }

    SPDLASFileImporterStrictPulses::~SPDLASFileImporterStrictPulses()
    {

    }

    SPDLASFileNoPulsesImporter::SPDLASFileNoPulsesImporter(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold):SPDDataImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold)
    {

    }

    SPDDataImporter* SPDLASFileNoPulsesImporter::getInstance(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold)
    {
        return new SPDLASFileNoPulsesImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
    }

    std::list<SPDPulse*>* SPDLASFileNoPulsesImporter::readAllDataToList(std::string inputFile, SPDFile *spdFile)
    {
        /* This has been removed when updated to LASlib.
         * It is only required when exporting to a format which needs the number
         * of out points (export->needNumOutPts()).
         * Will add back and update to use LASlib if it causes problems.
         */
        throw SPDIOException("Reading LAS to a list is not currently supported");
    }

    std::vector<SPDPulse*>* SPDLASFileNoPulsesImporter::readAllDataToVector(std::string inputFile, SPDFile *spdFile)
    {
        /**
         * Class to read LAS format input data to a vector of SPDPulses.
         * Doesn't try to recreate pulse or read waveform data - just generates
         * a 'pulse' for each point.
         */
        SPDPulseUtils pulseUtils;
        std::vector<SPDPulse*> *pulses = new std::vector<SPDPulse*>();
        boost::uint_fast64_t numOfPulses = 0;
        boost::uint_fast64_t numOfPoints = 0;

        double xMin = 0;
        double xMax = 0;
        double yMin = 0;
        double yMax = 0;
        double zMin = 0;
        double zMax = 0;
        bool first = true;
        bool firstZ = true;

        classWarningGiven = false;

        try
        {
            // Open LAS file
            LASreadOpener lasreadopener;
            LASreader* lasreader = lasreadopener.open(inputFile.c_str());

            if(lasreader != 0)
            {
                // Get header
                LASheader *header = &lasreader->header;

                spdFile->setFileSignature(header->file_signature);
                spdFile->setSystemIdentifier(header->system_identifier);

                if(spdFile->getSpatialReference() == "")
                {
                    // Get WKT string from LAS header
                    spdFile->setSpatialReference(getWKTfromLAS(header));
                }

                if(convertCoords)
                {
                    this->initCoordinateSystemTransformation(spdFile);
                }


                pulses->reserve(header->number_of_point_records);

                boost::uint_fast64_t reportedNumOfPts = header->number_of_point_records;
                boost::uint_fast64_t feedback = reportedNumOfPts/10;
                unsigned int feedbackCounter = 0;

                SPDPoint *spdPt = NULL;
                SPDPulse *spdPulse = NULL;

                std::cout << "Started (Read Data) ." << std::flush;
                while (lasreader->read_point())
                {
                    if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
                    {
                        std::cout << "." << feedbackCounter << "." << std::flush;
                        feedbackCounter += 10;
                    }

                    spdPt = this->createSPDPoint(lasreader->point);
                    ++numOfPoints;

                    if(firstZ)
                    {
                        zMin = spdPt->z;
                        zMax = spdPt->z;
                        firstZ = false;
                    }
                    else
                    {
                        if(spdPt->z < zMin)
                        {
                            zMin = spdPt->z;
                        }
                        else if(spdPt->z > zMax)
                        {
                            zMax = spdPt->z;
                        }
                    }

                    spdPulse = new SPDPulse();
                    pulseUtils.initSPDPulse(spdPulse);
                    spdPulse->pulseID = numOfPulses;
                    spdPulse->numberOfReturns = 1;

                    spdPulse->pts->push_back(spdPt);

                    ++numOfPulses;

                    if(lasreader->point.get_edge_of_flight_line() == 1)
                    {
                        spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                    }
                    else
                    {
                        spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                    }

                    if(lasreader->point.get_scan_direction_flag() == 1)
                    {
                        spdPulse->scanDirectionFlag = SPD_POSITIVE;
                    }
                    else
                    {
                        spdPulse->scanDirectionFlag = SPD_NEGATIVE;
                    }

                    spdPulse->zenith = 0.0;
                    spdPulse->azimuth = 0.0;

                    spdPulse->user = lasreader->point.get_scan_angle_rank() + 90;

                    spdPulse->sourceID = lasreader->point.get_point_source_ID();

                    if(indexCoords == SPD_FIRST_RETURN)
                    {
                        spdPulse->xIdx = spdPulse->pts->front()->x;
                        spdPulse->yIdx = spdPulse->pts->front()->y;
                    }
                    else if(indexCoords == SPD_LAST_RETURN)
                    {
                        spdPulse->xIdx = spdPulse->pts->back()->x;
                        spdPulse->yIdx = spdPulse->pts->back()->y;
                    }
                    else
                    {
                        throw SPDIOException("Indexing type unsupported");
                    }

                    if(first)
                    {
                        xMin = spdPulse->xIdx;
                        xMax = spdPulse->xIdx;
                        yMin = spdPulse->yIdx;
                        yMax = spdPulse->yIdx;
                        first = false;
                    }
                    else
                    {
                        if(spdPulse->xIdx < xMin)
                        {
                            xMin = spdPulse->xIdx;
                        }
                        else if(spdPulse->xIdx > xMax)
                        {
                            xMax = spdPulse->xIdx;
                        }

                        if(spdPulse->yIdx < yMin)
                        {
                            yMin = spdPulse->yIdx;
                        }
                        else if(spdPulse->yIdx > yMax)
                        {
                            yMax = spdPulse->yIdx;
                        }
                    }

                    pulses->push_back(spdPulse);
                }
                lasreader->close();
                std::cout << ". Complete\n";
                spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
                if(convertCoords)
                {
                    spdFile->setSpatialReference(outputProjWKT);
                }
                spdFile->setNumberOfPulses(numOfPulses);
                spdFile->setNumberOfPoints(numOfPoints);
                spdFile->setOriginDefined(SPD_FALSE);
                spdFile->setDiscretePtDefined(SPD_TRUE);
                spdFile->setDecomposedPtDefined(SPD_FALSE);
                spdFile->setTransWaveformDefined(SPD_FALSE);
            }
            else
            {
                throw SPDIOException("LAS file could not be opened.");
            }
        }
        catch (SPDIOException &e)
        {
            throw e;
        }
        catch(std::invalid_argument &e)
        {
            throw SPDIOException(e.what());
        }
        catch(std::runtime_error &e)
        {
            throw SPDIOException(e.what());
        }

        return pulses;

    }

    void SPDLASFileNoPulsesImporter::readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor)
    {
        /**
         * Class to read LAS format input data and process
         * Doesn't try to recreate pulse or read waveform data - just generates
         * a 'pulse' for each point.
         */
        SPDPulseUtils pulseUtils;
        boost::uint_fast64_t numOfPulses = 0;
        boost::uint_fast64_t numOfPoints = 0;

        double xMin = 0;
        double xMax = 0;
        double yMin = 0;
        double yMax = 0;
        double zMin = 0;
        double zMax = 0;
        bool first = true;
        bool firstZ = true;

        classWarningGiven = false;

        try
        {
            // Open LAS file
            LASreadOpener lasreadopener;
            LASreader* lasreader = lasreadopener.open(inputFile.c_str());

            if(lasreader != 0)
            {
                // Get header
                LASheader *header = &lasreader->header;

                spdFile->setFileSignature(header->file_signature);
                spdFile->setSystemIdentifier(header->system_identifier);

                if(spdFile->getSpatialReference() == "")
                {
                    // Get WKT string from LAS header
                    spdFile->setSpatialReference(getWKTfromLAS(header));
                }

                if(convertCoords)
                {
                    this->initCoordinateSystemTransformation(spdFile);
                }

                boost::uint_fast64_t reportedNumOfPts = header->number_of_point_records;
                boost::uint_fast64_t feedback = reportedNumOfPts/10;
                unsigned int feedbackCounter = 0;

                SPDPoint *spdPt = NULL;
                SPDPulse *spdPulse = NULL;

                std::cout << "Started (Read Data) ." << std::flush;
                while (lasreader->read_point())
                {
                    if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
                    {
                        std::cout << "." << feedbackCounter << "." << std::flush;
                        feedbackCounter += 10;
                    }

                    spdPt = this->createSPDPoint(lasreader->point);
                    ++numOfPoints;

                    if(firstZ)
                    {
                        zMin = spdPt->z;
                        zMax = spdPt->z;
                        firstZ = false;
                    }
                    else
                    {
                        if(spdPt->z < zMin)
                        {
                            zMin = spdPt->z;
                        }
                        else if(spdPt->z > zMax)
                        {
                            zMax = spdPt->z;
                        }
                    }

                    spdPulse = new SPDPulse();
                    pulseUtils.initSPDPulse(spdPulse);
                    spdPulse->pulseID = numOfPulses;
                    spdPulse->numberOfReturns = 1;

                    spdPulse->pts->push_back(spdPt);

                    ++numOfPulses;

                    if(lasreader->point.get_edge_of_flight_line() == 1)
                    {
                        spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                    }
                    else
                    {
                        spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                    }

                    if(lasreader->point.get_scan_direction_flag() == 1)
                    {
                        spdPulse->scanDirectionFlag = SPD_POSITIVE;
                    }
                    else
                    {
                        spdPulse->scanDirectionFlag = SPD_NEGATIVE;
                    }

                    spdPulse->zenith = 0.0;
                    spdPulse->azimuth = 0.0;

                    spdPulse->user = lasreader->point.get_scan_angle_rank() + 90;

                    spdPulse->sourceID = lasreader->point.get_point_source_ID();

                    if(indexCoords == SPD_FIRST_RETURN)
                    {
                        spdPulse->xIdx = spdPulse->pts->front()->x;
                        spdPulse->yIdx = spdPulse->pts->front()->y;
                    }
                    else if(indexCoords == SPD_LAST_RETURN)
                    {
                        spdPulse->xIdx = spdPulse->pts->back()->x;
                        spdPulse->yIdx = spdPulse->pts->back()->y;
                    }
                    else
                    {
                        throw SPDIOException("Indexing type unsupported");
                    }

                    if(first)
                    {
                        xMin = spdPulse->xIdx;
                        xMax = spdPulse->xIdx;
                        yMin = spdPulse->yIdx;
                        yMax = spdPulse->yIdx;
                        first = false;
                    }
                    else
                    {
                        if(spdPulse->xIdx < xMin)
                        {
                            xMin = spdPulse->xIdx;
                        }
                        else if(spdPulse->xIdx > xMax)
                        {
                            xMax = spdPulse->xIdx;
                        }

                        if(spdPulse->yIdx < yMin)
                        {
                            yMin = spdPulse->yIdx;
                        }
                        else if(spdPulse->yIdx > yMax)
                        {
                            yMax = spdPulse->yIdx;
                        }
                    }

                    processor->processImportedPulse(spdFile, spdPulse);

                }
                lasreader->close();
                std::cout << ". Complete\n";
                spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
                if(convertCoords)
                {
                    spdFile->setSpatialReference(outputProjWKT);
                }
                spdFile->setNumberOfPulses(numOfPulses);
                spdFile->setNumberOfPoints(numOfPoints);
                spdFile->setOriginDefined(SPD_FALSE);
                spdFile->setDiscretePtDefined(SPD_TRUE);
                spdFile->setDecomposedPtDefined(SPD_FALSE);
                spdFile->setTransWaveformDefined(SPD_FALSE);
            }
            else
            {
                throw SPDIOException("LAS file could not be opened.");
            }
        }
        catch (SPDIOException &e)
        {
            throw e;
        }
        catch(std::invalid_argument &e)
        {
            throw SPDIOException(e.what());
        }
        catch(std::runtime_error &e)
        {
            throw SPDIOException(e.what());
        }

    }

    bool SPDLASFileNoPulsesImporter::isFileType(std::string fileType)
    {
        if(fileType == "LASNP")
        {
            return true;
        }
        return false;
    }

    void SPDLASFileNoPulsesImporter::readHeaderInfo(std::string, SPDFile*) 
    {
        // No Header to Read..
    }

    SPDPoint* SPDLASFileNoPulsesImporter::createSPDPoint(LASpoint const& pt)
    {
        try
        {
            SPDPointUtils spdPtUtils;
            SPDPoint *spdPt = new SPDPoint();
            spdPtUtils.initSPDPoint(spdPt);
            // Get scaled values of x, y and z
            double x = pt.get_x();
            double y = pt.get_y();
            double z = pt.get_z();

            if(convertCoords)
            {
                this->transformCoordinateSystem(&x, &y, &z);
            }

            spdPt->x = x;
            spdPt->y = y;
            spdPt->z = z;
            spdPt->amplitudeReturn = pt.get_intensity();
            spdPt->user = pt.get_user_data();

            unsigned int lasClass = pt.get_classification();

            switch (lasClass)
            {
                case 0:
                    spdPt->classification = SPD_CREATED;
                    break;
                case 1:
                    spdPt->classification = SPD_UNCLASSIFIED;
                    break;
                case 2:
                    spdPt->classification = SPD_GROUND;
                    break;
                case 3:
                    spdPt->classification = SPD_LOW_VEGETATION;
                    break;
                case 4:
                    spdPt->classification = SPD_MEDIUM_VEGETATION;
                    break;
                case 5:
                    spdPt->classification = SPD_HIGH_VEGETATION;
                    break;
                case 6:
                    spdPt->classification = SPD_BUILDING;
                    break;
                case 7:
                    spdPt->classification = SPD_CREATED;
                    spdPt->lowPoint = SPD_TRUE;
                    break;
                case 8:
                    spdPt->classification = SPD_CREATED;
                    spdPt->modelKeyPoint = SPD_TRUE;
                    break;
                case 9:
                    spdPt->classification = SPD_WATER;
                    break;
                case 12:
                    spdPt->classification = SPD_CREATED;
                    spdPt->overlap = SPD_TRUE;
                    break;
                default:
                    spdPt->classification = SPD_CREATED;
                    if(!classWarningGiven)
                    {
                        std::cerr << "\nWARNING: The class ID " << lasClass<< " was not recognised - check the classes points were allocated too." << std::endl;
                        classWarningGiven = true;
                    }
                    break;
            }

            // Get array of RBG values (of type U16 in LASlib typedef)
            const unsigned short *rgb = pt.get_rgb();

            spdPt->red = rgb[0];
            spdPt->green = rgb[1];
            spdPt->blue = rgb[2];

            spdPt->returnID = pt.get_return_number();
            // Convert GPS time in s to ns for SPDLib (stored as 64 bit float)
            spdPt->gpsTime = pt.get_gps_time()*1E9;

            return spdPt;
        }
        catch (SPDIOException &e)
        {
            throw e;
        }
        catch(std::invalid_argument &e)
        {
            throw SPDIOException(e.what());
        }
        catch(std::runtime_error &e)
        {
            throw SPDIOException(e.what());
        }

    }

    SPDLASFileNoPulsesImporter::~SPDLASFileNoPulsesImporter()
    {

    }


}
