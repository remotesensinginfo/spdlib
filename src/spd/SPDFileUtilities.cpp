/*
 *  SPDFileUtilities.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 09/03/2011.
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

#ifdef _MSC_VER
    // include it here so it doesn't need to be installed
    #include "spd/windows/dirent.h"
#else
    #include <dirent.h>
#endif
#include "spd/SPDFileUtilities.h"

namespace spdlib 
{
    SPDFileUtilities::SPDFileUtilities()
    {
        
    }
    
    boost::uint_fast16_t SPDFileUtilities::getDIRCount(std::string dir) 
	{
		DIR *dp;
		struct dirent *dirp;
		if((dp  = opendir(dir.c_str())) == NULL)
		{
			std::string message = std::string("Could not open ") + dir;
			throw SPDException(message);
		}
		
        std::vector<std::string> files;
		while ((dirp = readdir(dp)) != NULL)
		{
            if((std::string(dirp->d_name) != std::string(".")) & (std::string(dirp->d_name) != std::string("..")))
            {
                files.push_back(std::string(dirp->d_name));
            }
		}
		closedir(dp);
        return files.size();
	}
    
	void SPDFileUtilities::getDIRList(std::string dir, std::list<std::string> *files) 
	{
		DIR *dp;
		struct dirent *dirp;
		if((dp  = opendir(dir.c_str())) == NULL) 
		{
			std::string message = std::string("Could not open ") + dir;
			throw SPDException(message);
		}
		
		while ((dirp = readdir(dp)) != NULL) 
		{
			files->push_back(std::string(dirp->d_name));
		}
		closedir(dp);
	}
	
	void SPDFileUtilities::getDIRList(std::string dir, std::vector<std::string> *files) 
	{
		DIR *dp;
		struct dirent *dirp;
		if((dp  = opendir(dir.c_str())) == NULL) 
		{
			std::string message = std::string("Could not open ") + dir;
			throw SPDException(message);
		}
		
		while ((dirp = readdir(dp)) != NULL) 
		{
			files->push_back(std::string(dirp->d_name));
		}
		closedir(dp);
	}
	
	void SPDFileUtilities::getDIRList(std::string dir, std::string ext, std::list<std::string> *files, bool withpath) 
	{
		DIR *dp;
		struct dirent *dirp;
		if((dp  = opendir(dir.c_str())) == NULL) 
		{
			std::string message = std::string("Could not open ") + dir;
			throw SPDException(message);
		}
		
		std::string filename = "";
		
		while ((dirp = readdir(dp)) != NULL) 
		{
			filename = std::string(dirp->d_name);
			if(SPDFileUtilities::getExtension(filename) == ext)
			{
				if(withpath)
				{
					filename = dir + filename;
				}
				files->push_back(filename);
			}
		}
		closedir(dp);		
	}
	
	void SPDFileUtilities::getDIRList(std::string dir, std::string ext, std::vector<std::string> *files, bool withpath) 
	{
		DIR *dp;
		struct dirent *dirp;
		if((dp  = opendir(dir.c_str())) == NULL) 
		{
			std::string message = std::string("Could not open ") + dir;
			throw SPDException(message);
		}
		
		std::string filename = "";
		
		while ((dirp = readdir(dp)) != NULL) 
		{
			filename = std::string(dirp->d_name);
			if(SPDFileUtilities::getExtension(filename) == ext)
			{
				if(withpath)
				{
					filename = dir + filename;
				}
				files->push_back(filename);
			}
		}
		closedir(dp);		
	}
	
	std::string* SPDFileUtilities::getDIRList(std::string dir, std::string ext, boost::uint_fast32_t *numFiles, bool withpath) 
	{
		std::vector<std::string> *files = new std::vector<std::string>();
		DIR *dp;
		struct dirent *dirp;
		if((dp  = opendir(dir.c_str())) == NULL) 
		{
			std::string message = std::string("Could not open ") + dir;
			throw SPDException(message);
		}
		
		std::string filename = "";
		
		while ((dirp = readdir(dp)) != NULL) 
		{
			filename = std::string(dirp->d_name);
			if(SPDFileUtilities::getExtension(filename) == ext)
			{
				if(withpath)
				{
					filename = dir + filename;
				}
				files->push_back(filename);
			}
		}
		closedir(dp);
		
		*numFiles = files->size();
		std::string *outputFiles = new std::string[*numFiles];
		for(boost::uint_fast32_t i = 0; i < *numFiles; i++)
		{
			outputFiles[i] = dir + files->at(i);
		}
		delete files;
		
		return outputFiles;
	}
	
	std::string* SPDFileUtilities::getFilesInDIRWithName(std::string dir, std::string name, boost::uint_fast32_t *numFiles) 
	{
		std::vector<std::string> *files = new std::vector<std::string>();
		DIR *dp;
		struct dirent *dirp;
		if((dp  = opendir(dir.c_str())) == NULL) 
		{
			std::string message = std::string("Could not open ") + dir;
			throw SPDException(message);
		}
		
		std::string filename = "";
		
		while ((dirp = readdir(dp)) != NULL) 
		{
			filename = std::string(dirp->d_name);
			//std::cout << "Filename (" << name << "): " << filename << " (" << SPDFileUtilities::getFileNameNoExtension(filename) << ")"<< std::endl;
			if(SPDFileUtilities::getFileNameNoExtension(filename) == name)
			{
				files->push_back(filename);
			}
		}
		closedir(dp);
		
		*numFiles = files->size();
		std::string *outputFiles = new std::string[*numFiles];
		for(boost::uint_fast32_t i = 0; i < *numFiles; i++)
		{
			outputFiles[i] = dir + files->at(i);
		}
		delete files;
		
		return outputFiles;
	}
	
	std::string SPDFileUtilities::getFileName(std::string filepath)
	{
		//std::cout << filepath << std::endl;
		boost::uint_fast32_t strSize = filepath.size();
		boost::uint_fast32_t lastSlash = 0;
		for(boost::uint_fast32_t i = 0; i < strSize; i++)
		{
			if(filepath.at(i) == '/')
			{
				lastSlash = i + 1;
			}
		}
		std::string filename = filepath.substr(lastSlash);
		//std::cout << filename << std::endl;
		return filename;	
	}
	
	std::string SPDFileUtilities::getFileNameNoExtension(std::string filepath)
	{
		//std::cout << filepath << std::endl;
		boost::uint_fast32_t strSize = filepath.size();
		boost::uint_fast32_t lastSlash = 0;
		for(boost::uint_fast32_t i = 0; i < strSize; i++)
		{
			if(filepath.at(i) == '/')
			{
				lastSlash = i + 1;
			}
		}
		std::string filename = filepath.substr(lastSlash);
		//std::cout << filename << std::endl;
		
		strSize = filename.size();
		boost::uint_fast32_t lastpt = 0;
		for(boost::uint_fast32_t i = 0; i < strSize; i++)
		{
			if(filename.at(i) == '.')
			{
				lastpt = i;
			}
		}
		
		std::string layerName = filename.substr(0, lastpt);
		//std::cout << layerName << std::endl;
		return layerName;	
	}
	
	std::string SPDFileUtilities::removeExtension(std::string filepath)
	{
		boost::uint_fast32_t strSize = filepath.size();
		boost::uint_fast32_t lastpt = 0;
		for(boost::uint_fast32_t i = 0; i < strSize; i++)
		{
			if(filepath.at(i) == '.')
			{
				lastpt = i;
			}
		}
		
		std::string layerName = filepath.substr(0, lastpt);
		//std::cout << layerName << std::endl;
		return layerName;	
	}
	
	std::string SPDFileUtilities::getExtension(std::string filepath)
	{
		boost::uint_fast32_t strSize = filepath.size();
		boost::uint_fast32_t lastpt = 0;
		for(boost::uint_fast32_t i = 0; i < strSize; i++)
		{
			if(filepath.at(i) == '.')
			{
				lastpt = i;
			}
		}
		
		std::string extension = filepath.substr(lastpt);
		//std::cout << layerName << std::endl;
		return extension;	
	}
	
	std::string SPDFileUtilities::getFileDirectoryPath(std::string filepath)
	{
		boost::uint_fast32_t strSize = filepath.size();
		boost::uint_fast32_t lastSlash = 0;
		for(boost::uint_fast32_t i = 0; i < strSize; i++)
		{
			if(filepath.at(i) == '/')
			{
				lastSlash = i + 1;
			}
		}
		std::string path = filepath.substr(0, lastSlash);
		//std::cout << path << std::endl;
		return path;	
	}
	
	bool SPDFileUtilities::checkFilePresent(std::string file)
	{
		struct stat stFileInfo; 
		bool blnReturn; 
		boost::uint_fast32_t intStat; 
		
		intStat = stat(file.c_str(), &stFileInfo); 
		if(intStat == 0) 
		{  
			blnReturn = true; 
		} 
		else 
		{ 
			blnReturn = false; 
		}
		
		return blnReturn; 
	}
    
    bool SPDFileUtilities::checkDIR4SHP(std::string dir, std::string shp) 
	{
		std::string *dirList = NULL;
		boost::uint_fast32_t numFiles = 0;
		bool returnVal = false;
		
		try
		{
			dirList = SPDFileUtilities::getFilesInDIRWithName(dir, shp, &numFiles);
			if(numFiles > 0)
			{
				for(boost::uint_fast32_t i = 0; i < numFiles; i++)
				{
					if(SPDFileUtilities::getExtension(dirList[i]) == ".shp")
					{
						returnVal = true;
					}
				}
			}
		}
		catch(SPDException &e)
		{
			throw e;
		}
		delete[] dirList;
		
		return returnVal;
	}
	
	void SPDFileUtilities::deleteSHP(std::string dir, std::string shp) 
	{
		std::string *dirList = NULL;
		boost::uint_fast32_t numFiles = 0;
		
		try
		{
			dirList = SPDFileUtilities::getFilesInDIRWithName(dir, shp, &numFiles);
			if(numFiles > 0)
			{
				std::cout << "Deleting shapefile...\n";
				for(boost::uint_fast32_t i = 0; i < numFiles; ++i)
				{
					if(SPDFileUtilities::getExtension(dirList[i]) == ".shp")
					{
						std::cout << dirList[i];
						if( remove( dirList[i].c_str() ) != 0 )
						{
							throw SPDException("Could not delete file.");
						}
						std::cout << " deleted\n";
					}
					else if(SPDFileUtilities::getExtension(dirList[i]) == ".shx")
					{
						std::cout << dirList[i];
						if( remove( dirList[i].c_str() ) != 0 )
						{
							throw SPDException("Could not delete file.");
						}
						std::cout << " deleted\n";
					}
					else if(SPDFileUtilities::getExtension(dirList[i]) == ".sbx")
					{
						std::cout << dirList[i];
						if( remove( dirList[i].c_str() ) != 0 )
						{
							throw SPDException("Could not delete file.");
						}
						std::cout << " deleted\n";
					}
					else if(SPDFileUtilities::getExtension(dirList[i]) == ".sbn")
					{
						std::cout << dirList[i];
						if( remove( dirList[i].c_str() ) != 0 )
						{
							throw SPDException("Could not delete file.");
						}
						std::cout << " deleted\n";
					}
					else if(SPDFileUtilities::getExtension(dirList[i]) == ".dbf")
					{
						std::cout << dirList[i];
						if( remove( dirList[i].c_str() ) != 0 )
						{
							throw SPDException("Could not delete file.");
						}
						std::cout << " deleted\n";
					}
					else if(SPDFileUtilities::getExtension(dirList[i]) == ".prj")
					{
						std::cout << dirList[i];
						if( remove( dirList[i].c_str() ) != 0 )
						{
							throw SPDException("Could not delete file.");
						}
						std::cout << " deleted\n";
					}
					
				}
			}
		}
		catch(SPDException &e)
		{
			throw e;
		}
		delete[] dirList;
	}
    
    
    SPDFileUtilities::~SPDFileUtilities()
    {
        
    }
}

