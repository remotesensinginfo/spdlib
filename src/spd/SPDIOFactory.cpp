/*
 *  SPDIOFactory.cpp
 *  spdlib_prj
 *
 *  Created by Pete Bunting on 13/10/2009.
 *  Copyright 2009 SPDLib. All rights reserved.
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

#include "spd/SPDIOFactory.h"

namespace spdlib
{

	SPDIOFactory::SPDIOFactory(): exporters(NULL), importers(NULL)
	{
		exporters = new std::list<SPDDataExporter*>();
		importers = new std::list<SPDDataImporter*>();
		this->registerAll();
	}

	SPDIOFactory::SPDIOFactory(const SPDIOFactory &ioFactory): exporters(NULL), importers(NULL)
	{
		this->importers = ioFactory.importers;
		this->exporters = ioFactory.exporters;
	}

	SPDDataExporter* SPDIOFactory::getExporter(std::string filetype) throw(SPDIOException)
	{
		SPDDataExporter *dataExporter = NULL;
		bool found = false;
		
		std::list<SPDDataExporter*>::iterator iterExport;
		for(iterExport = exporters->begin(); iterExport != exporters->end(); ++iterExport)
		{
			if((*iterExport)->isFileType(filetype))
			{
				dataExporter = (*iterExport)->getInstance();
				found = true;
				break;
			}
		}
		
		if(!found)
		{
			throw SPDIOException("Could not find suitable exporter");
		}
		
		return dataExporter;
	}

	SPDDataImporter* SPDIOFactory::getImporter(std::string filetype, bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold) throw(SPDIOException)
	{
		SPDDataImporter *dataImporter = NULL;
		bool found = false;
		
        try
        {
            std::list<SPDDataImporter*>::iterator iterImport;
            for(iterImport = importers->begin(); iterImport != importers->end(); ++iterImport)
            {
                if((*iterImport)->isFileType(filetype))
                {
                    dataImporter = (*iterImport)->getInstance(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
                    dataImporter->readSchema();
                    found = true;
                    break;
                }
            }
            
            if(!found)
            {
                throw SPDIOException("Could not find suitable importor");
            }
        }
        catch(SPDIOException &e)
        {
            throw e;
        }
		
		return dataImporter;
	}

	void SPDIOFactory::registerExporter(SPDDataExporter *exporter)
	{
		exporters->push_back(exporter);
	}

	void SPDIOFactory::registerImporter(SPDDataImporter *importer)
	{
		importers->push_back(importer);
	}

	SPDIOFactory& SPDIOFactory::operator=(const SPDIOFactory& ioFactory)
	{
		this->importers = ioFactory.importers;
		this->exporters = ioFactory.exporters;
		return *this;
	}

	void SPDIOFactory::registerAll()
	{
		this->importers->push_back(new SPDTextFileImporter(new SPDLineParserASCIIPulsePerRow()));
        this->importers->push_back(new SPDTextFileImporter(new SPDLineParserASCII()));
		this->importers->push_back(new SPDFileReader());
		this->importers->push_back(new SPDFullWaveformDatFileImporter());
		this->importers->push_back(new SPDDecomposedDatFileImporter());
		this->importers->push_back(new SPDLASFileImporter());
        this->importers->push_back(new SPDLASFileNoPulsesImporter());
        this->importers->push_back(new SPDLASFileImporterStrictPulses());
		this->importers->push_back(new SPDDecomposedCOOFileImporter());
        this->importers->push_back(new SPDASCIIMultiLineReader());
        this->importers->push_back(new SPDSALCADataBinaryImporter());
        this->importers->push_back(new SPDOptechFullWaveformASCIIImport());
        this->importers->push_back(new SPDARADecomposedDatFileImporter());
				
		this->exporters->push_back(new SPDSeqFileWriter());
        this->exporters->push_back(new SPDNonSeqFileWriter());
        this->exporters->push_back(new SPDNoIdxFileWriter());
		this->exporters->push_back(new SPDGeneralASCIIFileWriter());
		this->exporters->push_back(new SPDLASFileExporter());
        this->exporters->push_back(new SPDLAZFileExporter());
	}

	SPDIOFactory::~SPDIOFactory()
	{
		std::list<SPDDataImporter*>::iterator iterImport;
		for(iterImport = importers->begin(); iterImport != importers->end(); )
		{
			delete *iterImport;
			importers->erase(iterImport++);
		}
		delete importers;
		
		std::list<SPDDataExporter*>::iterator iterExport;
		for(iterExport = exporters->begin(); iterExport != exporters->end(); )
		{
			delete *iterExport;
			exporters->erase(iterExport++);
		}
		delete exporters;
	}
		
}


