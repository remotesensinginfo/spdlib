/*
 *  SPDLASFileExporter.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 17/02/2011.
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
 *
 */

#include "spd/SPDLASFileExporter.h"
#include "spd/spd-config.h"


namespace spdlib
{
    SPDLASFileExporter::SPDLASFileExporter() : SPDDataExporter("LAS")
    {

    }

    SPDLASFileExporter::SPDLASFileExporter(const SPDDataExporter &dataExporter)  : SPDDataExporter(dataExporter)
    {
        if(fileOpened)
        {
            throw SPDException("Cannot make a copy of a file exporter when a file is open.");
        }
    }

    SPDLASFileExporter& SPDLASFileExporter::operator=(const SPDLASFileExporter& dataExporter) 
    {
        if(fileOpened)
        {
            throw SPDException("Cannot make a copy of a file exporter when a file is open.");
        }

        this->spdFile = dataExporter.spdFile;
        this->outputFile = dataExporter.outputFile;
        return *this;
    }

    SPDDataExporter* SPDLASFileExporter::getInstance()
    {
        return new SPDLASFileExporter();
    }

    bool SPDLASFileExporter::open(SPDFile *spdFile, std::string outputFile) 
    {
        fileOpened = false;
        try
        {
            if (spdFile->getDecomposedPtDefined() == SPD_TRUE)
            {
                std::cout << "Decomposed Point data found - Note. Widths are not stored.\n";
            }
            else if(spdFile->getDiscretePtDefined() == SPD_TRUE)
            {
                std::cout << "Point data found\n";
            }
            else
            {
                throw SPDIOException("This writer can only export point data.");
            }

            std::cout << "Outputting to " << outputFile << std::endl;

            this->lasFileHeader = new LASheader;
            lasFileHeader->point_data_format = 2;
            lasFileHeader->point_data_record_length = 26;

            if(spdFile->getSpatialReference() != "")
            {
                /*liblas::SpatialReference lasSpatRef;
                lasSpatRef.SetWKT(spdFile->getSpatialReference());
                lasFileHeader.SetSRS(lasSpatRef);*/
            }

            // Set scale factors
            lasFileHeader->x_scale_factor = LAS_SCALE_FACTOR;
            lasFileHeader->y_scale_factor = LAS_SCALE_FACTOR;
            lasFileHeader->z_scale_factor = LAS_SCALE_FACTOR;

            // Set bounding box
            if(this->exportZasH)
            {
                lasFileHeader->set_bounding_box(spdFile->getXMin(),spdFile->getYMin(),0,
                                                spdFile->getXMax(),spdFile->getYMax(),0);
            }
            else
            {
                lasFileHeader->set_bounding_box(spdFile->getXMin(),spdFile->getYMin(),spdFile->getZMin(),
                                                spdFile->getXMax(),spdFile->getYMax(),spdFile->getZMax());
            }
            strncpy(lasFileHeader->generating_software, SPDLIB_PACKAGE_STRING, 32);
            strncpy(lasFileHeader->system_identifier, "EXPORT", 32);

            // Open output file for writing
            LASwriteOpener laswriteopener;
            laswriteopener.set_file_name(outputFile.c_str());
            this->lasWriter = laswriteopener.open(lasFileHeader);

            fileOpened = true;
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
        catch (std::exception const& e)
        {
          throw SPDIOException(e.what());
        }

        finalisedClosed = false;

        return fileOpened;
    }

    bool SPDLASFileExporter::reopen(SPDFile *spdFile, std::string outputFile) 
    {
        throw SPDIOException("No reopen option available.");
    }

    void SPDLASFileExporter::writeDataColumn(std::list<SPDPulse*> *plsIn, boost::uint_fast32_t col, boost::uint_fast32_t row)
    {
        SPDPulseUtils pulseUtils;
        SPDPointUtils pointUtils;

        if(!fileOpened)
        {
            throw SPDIOException("Output file not open, cannot write to the file.");
        }

        try
        {
            std::vector<SPDPoint*>::iterator iterPts;
            std::list<SPDPulse*>::iterator iterInPls;

            // Setup LAS point
            LASpoint point;
            point.init(this->lasFileHeader, this->lasFileHeader->point_data_format, this->lasFileHeader->point_data_record_length, 0);

            if(plsIn->size() > 0)
            {
                for(iterInPls = plsIn->begin(); iterInPls != plsIn->end(); ++iterInPls)
                {
                    if((*iterInPls)->numberOfReturns)
                    {
                        for(iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
                        {
                            //cout << "PT (list): [" << (*iterPts)->x << ", " << (*iterPts)->y << ", " << (*iterPts)->z << "]\n";
                            point.set_X((*iterPts)->x/LAS_SCALE_FACTOR);
                            point.set_Y((*iterPts)->y/LAS_SCALE_FACTOR);
                            // If exportZasH write point height to Z field
                            if(this->exportZasH)
                            {
                                point.set_Z((*iterPts)->height/LAS_SCALE_FACTOR);
                            }
                            else
                            {
                                point.set_Z((*iterPts)->z/LAS_SCALE_FACTOR);
                            }
                            point.set_intensity((*iterPts)->amplitudeReturn);
                            // Convert SPD GPS time (ns) to LAS (s)
                            point.set_gps_time((*iterPts)->gpsTime/1E9);
                            point.set_intensity((*iterPts)->amplitudeReturn);
                            point.set_return_number((*iterPts)->returnID);
                            point.set_number_of_returns((*iterInPls)->numberOfReturns);
                            //point.scan_angle_rank((*iterInPls)->zenith*180.0/3.141592653589793+0.5-180.0);
                            point.set_user_data((*iterPts)->widthReturn);

                            // Add RGB
                            point.rgb[0] = (*iterPts)->red;
                            point.rgb[1] = (*iterPts)->green;
                            point.rgb[2] = (*iterPts)->blue;

                            switch ((*iterPts)->classification)
                            {
                                case SPD_CREATED:
                                    if((*iterPts)->lowPoint == SPD_TRUE)
                                    {
                                        point.set_classification(7);
                                    }
                                    else if((*iterPts)->modelKeyPoint == SPD_TRUE)
                                    {
                                        point.set_classification(8);
                                    }
                                    else if((*iterPts)->overlap == SPD_TRUE)
                                    {
                                        point.set_classification(12);
                                    }
                                    else
                                    {
                                        point.set_classification(0);
                                    }
                                    break;
                                case SPD_UNCLASSIFIED:
                                    point.set_classification(1);
                                    break;
                                case SPD_GROUND:
                                    point.set_classification(2);
                                    break;
                                case SPD_LOW_VEGETATION:
                                    point.set_classification(3);
                                    break;
                                case SPD_MEDIUM_VEGETATION:
                                    point.set_classification(4);
                                    break;
                                case SPD_HIGH_VEGETATION:
                                    point.set_classification(5);
                                    break;
                                case SPD_BUILDING:
                                    point.set_classification(6);
                                    break;
                                case SPD_WATER:
                                    point.set_classification(9);
                                    break;
                                default:
                                    point.set_classification(0);
                                    break;
                            }

                            // write the point
                            this->lasWriter->write_point(&point);
                            // add it to the inventory
                            this->lasWriter->update_inventory(&point);
                        }
                    }
                    SPDPulseUtils::deleteSPDPulse(*iterInPls);
                }
                plsIn->clear();
            }

        }
        catch(SPDIOException &e)
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

    void SPDLASFileExporter::writeDataColumn(std::vector<SPDPulse*> *plsIn, boost::uint_fast32_t col, boost::uint_fast32_t row)
    {
        SPDPulseUtils pulseUtils;
        SPDPointUtils pointUtils;

        if(!fileOpened)
        {
            throw SPDIOException("Output file not open, cannot write to the file.");
        }

        try
        {
            std::vector<SPDPoint*>::iterator iterPts;
            std::vector<SPDPulse*>::iterator iterInPls;

            // Set up LAS point
            LASpoint point;
            point.init(this->lasFileHeader, this->lasFileHeader->point_data_format, this->lasFileHeader->point_data_record_length, 0);

            if(plsIn->size() > 0)
            {
                for(iterInPls = plsIn->begin(); iterInPls != plsIn->end(); ++iterInPls)
                {
                    if((*iterInPls)->numberOfReturns)
                    {
                        for(iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
                        {
                            point.set_X((*iterPts)->x/LAS_SCALE_FACTOR);
                            point.set_Y((*iterPts)->y/LAS_SCALE_FACTOR);
                            // If exportZasH write point height to Z field
                            if(this->exportZasH)
                            {
                                point.set_Z((*iterPts)->height/LAS_SCALE_FACTOR);
                            }
                            else
                            {
                                point.set_Z((*iterPts)->z/LAS_SCALE_FACTOR);
                            }
                            point.set_intensity((*iterPts)->amplitudeReturn);
                            // Convert GPS time in s to ns for SPDLib (stored as 64 bit float)
                            point.set_gps_time((*iterPts)->gpsTime/1E9);
                            point.set_intensity((*iterPts)->amplitudeReturn);
                            point.set_return_number((*iterPts)->returnID);
                            point.set_number_of_returns((*iterInPls)->numberOfReturns);
                            //point.set_scan_angle_rank((*iterInPls)->zenith*180.0/3.141592653589793+0.5-180.0);
                            point.set_user_data((*iterPts)->widthReturn);

                            // Add RGB
                            point.rgb[0] = (*iterPts)->red;
                            point.rgb[1] = (*iterPts)->green;
                            point.rgb[2] = (*iterPts)->blue;

                            switch ((*iterPts)->classification)
                            {
                                case SPD_CREATED:
                                    if((*iterPts)->lowPoint == SPD_TRUE)
                                    {
                                        point.set_classification(7);
                                    }
                                    else if((*iterPts)->modelKeyPoint == SPD_TRUE)
                                    {
                                        point.set_classification(8);
                                    }
                                    else if((*iterPts)->overlap == SPD_TRUE)
                                    {
                                        point.set_classification(12);
                                    }
                                    else
                                    {
                                        point.set_classification(0);
                                    }
                                    break;
                                case SPD_UNCLASSIFIED:
                                    point.set_classification(1);
                                    break;
                                case SPD_GROUND:
                                    point.set_classification(2);
                                    break;
                                case SPD_LOW_VEGETATION:
                                    point.set_classification(3);
                                    break;
                                case SPD_MEDIUM_VEGETATION:
                                    point.set_classification(4);
                                    break;
                                case SPD_HIGH_VEGETATION:
                                    point.set_classification(5);
                                    break;
                                case SPD_BUILDING:
                                    point.set_classification(6);
                                    break;
                                case SPD_WATER:
                                    point.set_classification(9);
                                    break;
                                default:
                                    point.set_classification(0);
                                    break;
                            }

                            // write the modified point
                            this->lasWriter->write_point(&point);
                            // add it to the inventory
                            this->lasWriter->update_inventory(&point);
                        }
                    }
                    SPDPulseUtils::deleteSPDPulse(*iterInPls);
                }
                plsIn->clear();
            }

        }
        catch(SPDIOException &e)
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

    void SPDLASFileExporter::finaliseClose() 
    {
        if(!this->fileOpened)
        {
            throw SPDIOException("Output file not open, cannot finalise.");
        }

        if(!this->finalisedClosed)
        {
            // update the header
            this->lasWriter->update_header(this->lasFileHeader, TRUE);

            // Close writer
            this->lasWriter->close();
            delete this->lasWriter;
        }
        this->finalisedClosed = true;
    }

    bool SPDLASFileExporter::requireGrid()
    {
        return false;
    }

    bool SPDLASFileExporter::needNumOutPts()
    {
        return false;
    }

    SPDLASFileExporter::~SPDLASFileExporter()
    {
        if(this->fileOpened)
        {
            try
            {
                this->finaliseClose();
            }
            catch (SPDIOException &e)
            {
                std::cerr << "WARNING: " << e.what() << std::endl;
            }
        }
    }

}
