/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 30/11/2010.
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

#include <string>
#include <iostream>
#include <algorithm>

#include <spd/tclap/CmdLine.h>

#include "gdal_priv.h"
#include "ogrsf_frmts.h"
#include "ogr_spatialref.h"

#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDException.h"
#include "spd/SPDFile.h"
#include "spd/SPDFileReader.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[]) 
{
    std::cout.precision(12);
    
    //cout << "spdproj " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	//cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	//cout << "and you are welcome to redistribute it under certain conditions; See\n";
	//cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	//cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;
	
	try 
	{
        TCLAP::CmdLine cmd("Print and convert projection strings: spdproj", ' ', "1.0.0");
		
		TCLAP::ValueArg<std::string> proj4Arg("","proj4","Enter a proj4 string (to print WKT)",false,"", "string");
        TCLAP::ValueArg<std::string> proj4PrettyArg("","proj4pretty","Enter a proj4 string (to print Pretty WKT)",false,"", "string");
        TCLAP::ValueArg<std::string> imageArg("","image","Print the WKT string associated with the input image.",false,"", "string");
        TCLAP::ValueArg<std::string> imagePrettyArg("","imagepretty", "Print the WKT (to print Pretty WKT) string associated with the input image.",false,"", "string");
        TCLAP::ValueArg<std::string> spdArg("","spd","Print the WKT string associated with the input spd file.",false,"", "string");
        TCLAP::ValueArg<std::string> spdPrettyArg("","spdpretty", "Print the WKT (to print Pretty WKT) string associated with the input spd file.",false,"", "string");
        
        std::vector<TCLAP::Arg*> arguments;
        arguments.push_back(&proj4Arg);
        arguments.push_back(&proj4PrettyArg);
        arguments.push_back(&imageArg);
        arguments.push_back(&imagePrettyArg);
        arguments.push_back(&spdArg);
        arguments.push_back(&spdPrettyArg);
        
        cmd.xorAdd(arguments);
         
        cmd.parse( argc, argv );
        
        if(proj4PrettyArg.isSet())
        {
            OGRSpatialReference oSRS;
            oSRS.importFromProj4(proj4PrettyArg.getValue().c_str());
            char **wktspatialref = new char*[1];
            oSRS.exportToPrettyWkt(wktspatialref);
            std::cout << wktspatialref[0] << std::endl;
            OGRFree(wktspatialref);
        }
        else if(proj4Arg.isSet())
        {
            OGRSpatialReference oSRS;
            oSRS.importFromProj4(proj4Arg.getValue().c_str());
            char **wktspatialref = new char*[1];
            oSRS.exportToWkt(wktspatialref);
            std::cout << wktspatialref[0] << std::endl;
            OGRFree(wktspatialref);
        }
        else if(imageArg.isSet())
        {
            GDALAllRegister();
			GDALDataset *inGDALImage = (GDALDataset *) GDALOpenShared(imageArg.getValue().c_str(), GA_ReadOnly);
            if(inGDALImage == NULL)
            {
                std::string message = std::string("Could not open image ") + imageArg.getValue();
                throw spdlib::SPDException(message.c_str());
            }
            
            const char *wtkSpatialRef = inGDALImage->GetProjectionRef();
            
            std::cout << wtkSpatialRef << std::endl;
            
            GDALClose(inGDALImage);
			GDALDestroyDriverManager();
        }
        else if(imagePrettyArg.isSet())
        {
            GDALAllRegister();
			GDALDataset *inGDALImage = (GDALDataset *) GDALOpenShared(imagePrettyArg.getValue().c_str(), GA_ReadOnly);
            if(inGDALImage == NULL)
            {
                std::string message = std::string("Could not open image ") + imagePrettyArg.getValue();
                throw spdlib::SPDException(message.c_str());
            }
            
            const char *wtkSpatialRef = inGDALImage->GetProjectionRef();
            
            OGRSpatialReference ogrSpatial = OGRSpatialReference(wtkSpatialRef);
            
            char **wktPrettySpatialRef = new char*[1];
            ogrSpatial.exportToPrettyWkt(wktPrettySpatialRef);
            std::cout << wktPrettySpatialRef[0] << std::endl;
            OGRFree(wktPrettySpatialRef);

            GDALClose(inGDALImage);
			GDALDestroyDriverManager();            
        }
        else if(spdArg.isSet())
        {
            spdlib::SPDFile *spdFile = new spdlib::SPDFile(spdArg.getValue());
            spdlib::SPDFileReader spdReader;
            spdReader.readHeaderInfo(spdArg.getValue(), spdFile);
            std::cout << spdFile->getSpatialReference() << std::endl;
            delete spdFile;
        }
        else if(spdPrettyArg.isSet())
        {
            spdlib::SPDFile *spdFile = new spdlib::SPDFile(spdPrettyArg.getValue());
            spdlib::SPDFileReader spdReader;
            spdReader.readHeaderInfo(spdPrettyArg.getValue(), spdFile);
            
            OGRSpatialReference ogrSpatial = OGRSpatialReference(spdFile->getSpatialReference().c_str());
            
            char **wktPrettySpatialRef = new char*[1];
            ogrSpatial.exportToPrettyWkt(wktPrettySpatialRef);
            std::cout << wktPrettySpatialRef[0] << std::endl;
            OGRFree(wktPrettySpatialRef);
            
            delete spdFile;
        }
        else
        {
            throw spdlib::SPDException("No option given...");
        }
        
	}
	catch (TCLAP::ArgException &e) 
	{
		std::cerr << "Parse Error: " << e.what() << std::endl;
	}
    catch(spdlib::SPDException &e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
    }
}

