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

#include "spd/SPDFileUtilities.h"

namespace spdlib 
{
    SPDFileUtilities::SPDFileUtilities()
    {
        
    }
    
	void SPDFileUtilities::getDIRList(string dir, list<string> *files) throw(SPDException)
	{
		DIR *dp;
		struct dirent *dirp;
		if((dp  = opendir(dir.c_str())) == NULL) 
		{
			string message = string("Could not open ") + dir;
			throw SPDException(message);
		}
		
		while ((dirp = readdir(dp)) != NULL) 
		{
			files->push_back(std::string(dirp->d_name));
		}
		closedir(dp);
	}
	
	void SPDFileUtilities::getDIRList(string dir, vector<string> *files) throw(SPDException)
	{
		DIR *dp;
		struct dirent *dirp;
		if((dp  = opendir(dir.c_str())) == NULL) 
		{
			string message = string("Could not open ") + dir;
			throw SPDException(message);
		}
		
		while ((dirp = readdir(dp)) != NULL) 
		{
			files->push_back(std::string(dirp->d_name));
		}
		closedir(dp);
	}
	
	void SPDFileUtilities::getDIRList(string dir, string ext, list<string> *files, bool withpath) throw(SPDException)
	{
		DIR *dp;
		struct dirent *dirp;
		if((dp  = opendir(dir.c_str())) == NULL) 
		{
			string message = string("Could not open ") + dir;
			throw SPDException(message);
		}
		
		string filename = "";
		
		while ((dirp = readdir(dp)) != NULL) 
		{
			filename = string(dirp->d_name);
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
	
	void SPDFileUtilities::getDIRList(string dir, string ext, vector<string> *files, bool withpath) throw(SPDException)
	{
		DIR *dp;
		struct dirent *dirp;
		if((dp  = opendir(dir.c_str())) == NULL) 
		{
			string message = string("Could not open ") + dir;
			throw SPDException(message);
		}
		
		string filename = "";
		
		while ((dirp = readdir(dp)) != NULL) 
		{
			filename = string(dirp->d_name);
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
	
	string* SPDFileUtilities::getDIRList(string dir, string ext, boost::uint_fast32_t *numFiles, bool withpath) throw(SPDException)
	{
		vector<string> *files = new vector<string>();
		DIR *dp;
		struct dirent *dirp;
		if((dp  = opendir(dir.c_str())) == NULL) 
		{
			string message = string("Could not open ") + dir;
			throw SPDException(message);
		}
		
		string filename = "";
		
		while ((dirp = readdir(dp)) != NULL) 
		{
			filename = string(dirp->d_name);
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
		string *outputFiles = new string[*numFiles];
		for(boost::uint_fast32_t i = 0; i < *numFiles; i++)
		{
			outputFiles[i] = dir + files->at(i);
		}
		delete files;
		
		return outputFiles;
	}
	
	string* SPDFileUtilities::getFilesInDIRWithName(string dir, string name, boost::uint_fast32_t *numFiles) throw(SPDException)
	{
		vector<string> *files = new vector<string>();
		DIR *dp;
		struct dirent *dirp;
		if((dp  = opendir(dir.c_str())) == NULL) 
		{
			string message = string("Could not open ") + dir;
			throw SPDException(message);
		}
		
		string filename = "";
		
		while ((dirp = readdir(dp)) != NULL) 
		{
			filename = string(dirp->d_name);
			//cout << "Filename (" << name << "): " << filename << " (" << SPDFileUtilities::getFileNameNoExtension(filename) << ")"<< endl;
			if(SPDFileUtilities::getFileNameNoExtension(filename) == name)
			{
				files->push_back(filename);
			}
		}
		closedir(dp);
		
		*numFiles = files->size();
		string *outputFiles = new string[*numFiles];
		for(boost::uint_fast32_t i = 0; i < *numFiles; i++)
		{
			outputFiles[i] = dir + files->at(i);
		}
		delete files;
		
		return outputFiles;
	}
	
	string SPDFileUtilities::getFileName(string filepath)
	{
		//cout << filepath << endl;
		boost::uint_fast32_t strSize = filepath.size();
		boost::uint_fast32_t lastSlash = 0;
		for(boost::uint_fast32_t i = 0; i < strSize; i++)
		{
			if(filepath.at(i) == '/')
			{
				lastSlash = i + 1;
			}
		}
		string filename = filepath.substr(lastSlash);
		//cout << filename << endl;
		return filename;	
	}
	
	string SPDFileUtilities::getFileNameNoExtension(string filepath)
	{
		//cout << filepath << endl;
		boost::uint_fast32_t strSize = filepath.size();
		boost::uint_fast32_t lastSlash = 0;
		for(boost::uint_fast32_t i = 0; i < strSize; i++)
		{
			if(filepath.at(i) == '/')
			{
				lastSlash = i + 1;
			}
		}
		string filename = filepath.substr(lastSlash);
		//cout << filename << endl;
		
		strSize = filename.size();
		boost::uint_fast32_t lastpt = 0;
		for(boost::uint_fast32_t i = 0; i < strSize; i++)
		{
			if(filename.at(i) == '.')
			{
				lastpt = i;
			}
		}
		
		string layerName = filename.substr(0, lastpt);
		//cout << layerName << endl;
		return layerName;	
	}
	
	string SPDFileUtilities::removeExtension(string filepath)
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
		
		string layerName = filepath.substr(0, lastpt);
		//cout << layerName << endl;
		return layerName;	
	}
	
	string SPDFileUtilities::getExtension(string filepath)
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
		
		string extension = filepath.substr(lastpt);
		//cout << layerName << endl;
		return extension;	
	}
	
	string SPDFileUtilities::getFileDirectoryPath(string filepath)
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
		string path = filepath.substr(0, lastSlash);
		//cout << path << endl;
		return path;	
	}
	
	bool SPDFileUtilities::checkFilePresent(string file)
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
    
    bool SPDFileUtilities::checkDIR4SHP(string dir, string shp) throw(SPDException)
	{
		string *dirList = NULL;
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
	
	void SPDFileUtilities::deleteSHP(string dir, string shp) throw(SPDException)
	{
		string *dirList = NULL;
		boost::uint_fast32_t numFiles = 0;
		
		try
		{
			dirList = SPDFileUtilities::getFilesInDIRWithName(dir, shp, &numFiles);
			if(numFiles > 0)
			{
				cout << "Deleting shapefile...\n";
				for(boost::uint_fast32_t i = 0; i < numFiles; ++i)
				{
					if(SPDFileUtilities::getExtension(dirList[i]) == ".shp")
					{
						cout << dirList[i];
						if( remove( dirList[i].c_str() ) != 0 )
						{
							throw SPDException("Could not delete file.");
						}
						cout << " deleted\n";
					}
					else if(SPDFileUtilities::getExtension(dirList[i]) == ".shx")
					{
						cout << dirList[i];
						if( remove( dirList[i].c_str() ) != 0 )
						{
							throw SPDException("Could not delete file.");
						}
						cout << " deleted\n";
					}
					else if(SPDFileUtilities::getExtension(dirList[i]) == ".sbx")
					{
						cout << dirList[i];
						if( remove( dirList[i].c_str() ) != 0 )
						{
							throw SPDException("Could not delete file.");
						}
						cout << " deleted\n";
					}
					else if(SPDFileUtilities::getExtension(dirList[i]) == ".sbn")
					{
						cout << dirList[i];
						if( remove( dirList[i].c_str() ) != 0 )
						{
							throw SPDException("Could not delete file.");
						}
						cout << " deleted\n";
					}
					else if(SPDFileUtilities::getExtension(dirList[i]) == ".dbf")
					{
						cout << dirList[i];
						if( remove( dirList[i].c_str() ) != 0 )
						{
							throw SPDException("Could not delete file.");
						}
						cout << " deleted\n";
					}
					else if(SPDFileUtilities::getExtension(dirList[i]) == ".prj")
					{
						cout << dirList[i];
						if( remove( dirList[i].c_str() ) != 0 )
						{
							throw SPDException("Could not delete file.");
						}
						cout << " deleted\n";
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

