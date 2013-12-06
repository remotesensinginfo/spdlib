/*
 *  SPDImportSALCAData2SPD.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 04/12/2013.
 *  Copyright 2013 SPDLib. All rights reserved.
 *
 *  Code within this file has been provided by 
 *  Steven Hancock for reading the SALCA data
 *  and sorting out the geometry. This has been 
 *  adapted and brought across into the SPD
 *  importer interface.
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

#include "spd/SPDImportSALCAData2SPD.h"

namespace spdlib
{
    
    SPDSALCADataBinaryImporter::SPDSALCADataBinaryImporter(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold):SPDDataImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold)
    {
        
    }
		
    SPDDataImporter* SPDSALCADataBinaryImporter::getInstance(bool convertCoords, std::string outputProjWKT,std::string schema,boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold)
    {
        return new SPDSALCADataBinaryImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
    }
    
    std::list<SPDPulse*>* SPDSALCADataBinaryImporter::readAllDataToList(std::string, SPDFile *spdFile)throw(SPDIOException)
    {
        std::list<SPDPulse*> *pulses = new std::list<SPDPulse*>();
        
        return pulses;
    }
    
    std::vector<SPDPulse*>* SPDSALCADataBinaryImporter::readAllDataToVector(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)
    {
        std::vector<SPDPulse*> *pulses = new std::vector<SPDPulse*>();
        
        return pulses;
    }
    
    void SPDSALCADataBinaryImporter::readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) throw(SPDIOException)
    {
        try
        {
            std::vector<std::pair<float, std::string> > *inputBinFiles = new std::vector<std::pair<float, std::string> >();
            SalcaHDRParams *inParams = this->readHeaderParameters(inputFile, inputBinFiles);

            std::cout << "Header Parameters: \n";
            std::cout << "\tmaxR = " << inParams->maxR << std::endl;
            std::cout << "\tnAz = " << inParams->nAz << std::endl;
            std::cout << "\tnZen = " << inParams->nZen << std::endl;
            std::cout << "\tazStep = " << inParams->azStep << std::endl;
            std::cout << "\tzStep = " << inParams->zStep << std::endl;
            std::cout << "\tmaxZen = " << inParams->maxZen << std::endl;
            std::cout << "\tazStart = " << inParams->azStart << std::endl;
            std::cout << "\tazSquint = " << inParams->azSquint << std::endl;
            std::cout << "\tzenSquint = " << inParams->zenSquint << std::endl;
            std::cout << "\tomega = " << inParams->omega << std::endl;
            
            if(inputBinFiles->size() != inParams->nAz)
            {
                throw SPDIOException("The number of specified azimuth values is not equal to the number of input files.");
            }
            std::vector<float> wavelengths;
            wavelengths.push_back(1063);
            wavelengths.push_back(1545);
            spdFile->setWavelengths(wavelengths);
            std::vector<float> bandWidths;
            bandWidths.push_back(1);
            bandWidths.push_back(1);
            spdFile->setBandwidths(bandWidths);
            spdFile->setNumOfWavelengths(2);
            spdFile->setReceiveWaveformDefined(SPD_TRUE);
            spdFile->setTemporalBinSpacing(0.5);
            
            
            int z = 0, band = 0;    /* Loop control */
            int numb = 0;           /* Number of zenith steps */
            int nBins = 0;          /* Range bins per file (one file per azimuth) */
            int sOffset = 0;        /* Bounds to avoid outgoing pulse */
            int start[2];           /* Array start */
            int end[2];             /* Array end bounds */
            int length = 0;         /* Total file length */
            int waveStart = 0;      /* Waveform start in data array */
            float zen = 0;          /* True zenith */
            float az = 0;           /* True azimuth */
            char *data = NULL;      /* Waveform data. Holds whole file. */
            char ringTest = 0;      /* Ringing and saturation indicator */
            int val = 0;
            SPDPulse *pulse = NULL;
            SPDPulseUtils plsUtils;
            size_t pulseID = 0;
            boost::uint_fast32_t scanline = 0;
            boost::uint_fast16_t scanlineIdx = 0;
            
            sOffset=7;   /* 1.05 m */
            
            numb=(int)(inParams->maxZen/0.059375); /* Number of zenith steps. Zenith res is hard wired for SALCA */
            nBins=(int)((150.0+inParams->maxR)/0.15);
            
            /* Band start and end bins for sampling */
            start[0]=sOffset;   /* 1.05 m to avoid the outgoing pulse */
            end[0]=(int)(inParams->maxR/0.15);
            start[1]=1000+sOffset;   /* 7 to avoid outgoing pulses */
            end[1]=nBins;
            
            /* Set up squint angles */
            this->translateSquint(inParams);
            this->setSquint(inParams, numb);
            
            for(std::vector<std::pair<float, std::string> >::iterator iterFiles = inputBinFiles->begin(); iterFiles != inputBinFiles->end(); ++iterFiles)
            {
                std::cout << (*iterFiles).second << " with azimuth = " << (*iterFiles).first << std::endl;
                /* Read binary data into an array */
                data=this->readData((*iterFiles).second, (*iterFiles).first, &numb, &nBins, &length, inParams);
                
                if(data)
                {
                    zen = 0;
                    for(z = 0; z < numb; ++z)
                    {
                        /* zenith loop */
                        //zen=inParams->zen[z]; /* This is the 'really' zenith calculated value as provided by Steven */
                        zen += 0.059375; /* TESTING: trying to simiplify with regular zenith intervals to see if get something closer to what I expect */
                        
                        az=inParams->azOff[z]+(float)(*iterFiles).first*inParams->azStep+inParams->azStart;
                        
                        for(band = 0; band < 2; ++band)
                        {
                            /* loop through wavebands */
                            /* determine shot start position */
                            waveStart=this->findStart(start[band],end[band],&ringTest,data,z*nBins);
                            if((band==0) && (waveStart<0))
                            {
                                waveStart=0; /* as the 1545 pulse is often lost off the end */
                            }
                            
                            pulse = new SPDPulse();
                            plsUtils.initSPDPulse(pulse);
                            pulse->pulseID = pulseID++;
                            
                            /*the wave runs from data[z*nBins+waveStart] to data[z*nBins+end[band]] */
                            //fprintf(stdout,"Wave az %f zen %d band %d runs from %d to %d\n",(*iterFiles).first,z,band,z*nBins+waveStart,z*nBins+end[band]);
                            pulse->numOfReceivedBins = (z*nBins+end[band]) - (z*nBins+waveStart);
                            pulse->received = new uint_fast32_t[pulse->numOfReceivedBins];
                            for(int n = z*nBins+waveStart, i = 0; n < z*nBins+end[band]+1; ++n, ++i)
                            {
                                val = data[n];
                                val += 128;
                                pulse->received[i] = val;
                            }
                            
                            pulse->sourceID = band;
                            if(band == 0)
                            {
                                pulse->wavelength = 1063;
                            }
                            else if(band == 1)
                            {
                                pulse->wavelength = 1545;
                            }
                            
                            if(this->defineOrigin)
                            {
                                pulse->x0 = this->originX;
                                pulse->y0 = this->originY;
                                pulse->z0 = this->originZ;
                            }
                            else
                            {
                                pulse->x0 = 0;
                                pulse->y0 = 0;
                                pulse->z0 = 0;
                            }
                            
                            pulse->azimuth = ((*iterFiles).first/180.0)*M_PI;
                            pulse->zenith = ((zen/180.0)*M_PI)+(M_PI/2);
                            //std::cout << z << "\t zen = " << zen << "\tpulse->zenith = " << pulse->zenith << std::endl;
                            //std::cout << z << "," << zen << std::endl;
                            pulse->scanline = scanlineIdx;
                            pulse->scanlineIdx = scanline;
                            pulse->receiveWaveNoiseThreshold = 30;
                            pulse->rangeToWaveformStart = 0;
                            
                            processor->processImportedPulse(spdFile, pulse);
                        }
                        ++scanlineIdx;
                    }
                    delete[] data;
                }
                else
                {
                    throw SPDIOException("The data file was not opened or data was not read in correctly.");
                }
                ++scanline;
                scanlineIdx = 0;
            }
            
            if(inParams->azOff != NULL)
            {
                delete[] inParams->azOff;
            }
            if(inParams->zen != NULL)
            {
                delete[] inParams->zen;
            }
            
            delete inParams;
            delete inputBinFiles;
        }
        catch (SPDIOException &e)
        {
            throw e;
        }
        catch (std::exception &e)
        {
            throw SPDIOException(e.what());
        }
    }
    
    bool SPDSALCADataBinaryImporter::isFileType(std::string fileType)
    {
        if(fileType == "SALCA")
		{
			return true;
		}
		return false;
    }
    
    void SPDSALCADataBinaryImporter::readHeaderInfo(std::string inputFile, SPDFile *spdFile) throw(SPDIOException)
    {
        
    }
    
    SalcaHDRParams* SPDSALCADataBinaryImporter::readHeaderParameters(std::string headerFilePath, std::vector<std::pair<float,std::string> > *fileList)throw(SPDIOException)
    {
        SalcaHDRParams *hdrParams = new SalcaHDRParams();
        try
        {
            if(boost::filesystem::exists(headerFilePath))
            {
                boost::filesystem::path filePath = boost::filesystem::path(headerFilePath);
                filePath = boost::filesystem::absolute(filePath).parent_path();
                boost::filesystem::path tmpFilePath;
                
                
                SPDTextFileLineReader lineReader;
                SPDTextFileUtilities txtUtils;
                std::vector<std::string> *tokens = new std::vector<std::string>();
                lineReader.openFile(headerFilePath);
                std::string line = "";
                std::string filePathStr = "";
                float azimuthVal = 0.0;
                
                bool readingHeader = false;
                bool readingFileList = false;
                
                while(!lineReader.endOfFile())
                {
                    line = lineReader.readLine();
                    boost::algorithm::trim(line);
                    
                    if((line != "") && (!txtUtils.lineStartWithHash(line)))
                    {
                        if(line == "[START HEADER]")
                        {
                            readingHeader = true;
                        }
                        else if(line == "[END HEADER]")
                        {
                            readingHeader = false;
                        }
                        else if(line == "[FILE LIST START]")
                        {
                            readingHeader = false;
                            readingFileList = true;
                        }
                        else if(line == "[FILE LIST END]")
                        {
                            readingFileList = false;
                        }
                        
                        if(readingHeader & (line != "[START HEADER]"))
                        {
                            tokens->clear();
                            txtUtils.tokenizeString(line, '=', tokens, true);

                            if(tokens->at(0) == "maxR")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'maxR\'.");
                                }
                                hdrParams->maxR = txtUtils.strtofloat(tokens->at(1));
                            }
                            else if(tokens->at(0) == "nAz")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'nAz\'.");
                                }
                                hdrParams->nAz = txtUtils.strto16bitUInt(tokens->at(1));
                            }
                            else if(tokens->at(0) == "nZen")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'nZen\'.");
                                }
                                hdrParams->nZen = txtUtils.strto16bitUInt(tokens->at(1));
                            }
                            else if(tokens->at(0) == "azStep")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'azStep\'.");
                                }
                                hdrParams->azStep = txtUtils.strtofloat(tokens->at(1));
                            }
                            else if(tokens->at(0) == "zStep")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'zStep\'.");
                                }
                                hdrParams->zStep = txtUtils.strtofloat(tokens->at(1));
                            }
                            else if(tokens->at(0) == "maxZen")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'maxZen\'.");
                                }
                                hdrParams->maxZen = txtUtils.strtofloat(tokens->at(1));
                            }
                            else if(tokens->at(0) == "azStart")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'azStart\'.");
                                }
                                hdrParams->azStart = txtUtils.strtofloat(tokens->at(1));
                            }
                            else if(tokens->at(0) == "omega")
                            {
                                if(tokens->size() != 2)
                                {
                                    throw SPDIOException("Failed to parser header line \'omega\'.");
                                }
                                hdrParams->omega = txtUtils.strtofloat(tokens->at(1));
                            }
                            else
                            {
                                std::cout << line << std::endl;
                                
                                throw SPDIOException("Could not parser the header.");
                            }
                            
                        }
                        else if(readingFileList & (line != "[FILE LIST START]"))
                        {
                            tokens->clear();
                            txtUtils.tokenizeString(line, ':', tokens, true);
                            if(tokens->size() != 2)
                            {
                                throw SPDIOException("Failed to parser file list (Structure should be \'azimuth:filename\').");
                            }
                            azimuthVal = txtUtils.strtofloat(tokens->at(0));
                            
                            tmpFilePath = filePath;
                            tmpFilePath /= boost::filesystem::path(tokens->at(1));
                            
                            fileList->push_back(std::pair<float, std::string>(azimuthVal, tmpFilePath.string()));
                        }
                    }
                }
                delete tokens;
                lineReader.closeFile();
                
            }
        }
        catch (SPDIOException &e)
        {
            delete hdrParams;
            throw e;
        }
        catch (std::exception &e)
        {
            delete hdrParams;
            throw SPDIOException(e.what());
        }
        
        return hdrParams;
    }
    
    /** Translate from nice squint angles to those used in equations */
    void SPDSALCADataBinaryImporter::translateSquint(SalcaHDRParams *options)
    {
        float sinAz=0,sinZen=0;
        
        sinZen=sin(options->zenSquint);
        sinAz=sin(options->azSquint);
        
        options->azSquint=atan2(sinAz,sinZen);
        options->zenSquint=atan2(sqrt(sinAz*sinAz+sinZen*sinZen),1.0);
    }/*translateSquint*/
    
    
    /** Precalculate squint angles */
    void SPDSALCADataBinaryImporter::setSquint(SalcaHDRParams *options, int numb)
    {
        float zen=0,az=0;
        float cZen=0,cAz=0;
        
        options->zen = new float[numb];
        options->azOff = new float[numb];
        
        az=0.0;
        for(int i = 0; i < numb; ++i)
        {
            zen=((float)(options->nZen/2)-(float)i)*options->zStep;
            this->squint(&(cZen),&(cAz),zen,az,options->zenSquint,options->azSquint,options->omega);
            options->zen[i]=cZen;
            options->azOff[i]=cAz;
        }
        /*zenith loop*/

    }/*setSquint*/
    
    /** Caluclate squint angle */
    void SPDSALCADataBinaryImporter::squint(float *cZen,float *cAz,float zM,float aM,float zE,float aE,float omega)
    {
        float inc = 0;  /* Angle of incidence */
        float *vect = NULL;
        /* Working variables */
        float mX=0,mY=0,mZ=0; /* Mirror vector */
        float lX=0,lY=0,lZ=0; /* Incoming laser vector */
        float rX=0,rY=0,rZ=0; /* Vector orthogonal to m and l */
        float thetaZ=0;       /* Angle to rotate to mirror surface about z axis */
        float thetaX=0;       /* Angle to rotate about x axis */
        float slope=0;        /* Rotation vector slope angle, for rounding issues */
        /*trig*/
        float coszE=0,sinzE=0;
        float cosaE=0,sinaE=0;
        float coszM=0,sinzM=0;
        float cosW=0,sinW=0;
        
        coszE=cos(zE);
        sinzE=sin(zE);
        cosaE=cos(aE);
        sinaE=sin(aE);
        cosW=cos(omega);
        sinW=sin(omega);
        coszM=cos(zM);
        sinzM=sin(zM);
        
        mX=cosW;        /* Mirror normal vector */
        mY=sinW*sinzM;
        mZ=sinW*coszM;
        lX=-1.0*coszE;  /* Laser Pointing vector */
        lY=sinaE*sinzE;
        lZ=cosaE*sinzE;
        rX=lY*mZ-lZ*mY; /* Cross product of mirror and laser */
        rY=lZ*mX-lX*mZ; /* ie The vector to rotate about */
        rZ=lX*mY-lY*mX;
        
        inc=acos(-1.0*mX*lX+mY*lY+mZ*lZ);   /* Angle of incidence. Reverse x to get acute angle */
        thetaZ=-1.0*atan2(rX,rY);
        thetaX=atan2(rZ,sqrt(rX*rX+rY*rY));
        
        vect = new float[3];
        vect[0]=lX;
        vect[1]=lY;
        vect[2]=lZ;
        
        /* To avoid rounding rotate to z or y axis as appropriate */
        slope=atan2(sqrt(rX*rX+rY*rY),fabs(rZ));
        if(fabs(slope)<(M_PI/4.0))
        {
            /* Rotate about z axis */
            thetaX=-1.0*atan2(sqrt(rX*rX+rY*rY),rZ);
            thetaZ=-1.0*atan2(rX,rY);
            this->rotateZ(vect,thetaZ);
            this->rotateX(vect,thetaX);
            this->rotateZ(vect,-2.0*inc);
            this->rotateX(vect,-1.0*thetaX);
            this->rotateZ(vect,-1.0*thetaZ);
        }
        else
        {
            /* Rotate about y axis */
            thetaZ=-1.0*atan2(rX,rY);
            thetaX=atan2(rZ,sqrt(rX*rX+rY*rY));
            this->rotateZ(vect,thetaZ);
            this->rotateX(vect,thetaX);
            this->rotateY(vect,-2.0*inc);
            this->rotateX(vect,-1.0*thetaX);
            this->rotateZ(vect,-1.0*thetaZ);
        }
        
        *cZen=atan2(sqrt(vect[0]*vect[0]+vect[1]*vect[1]),vect[2]);
        if(vect[1]!=0.0)
        {
            *cAz=atan2(vect[0],vect[1])+aM;
        }
        else
        {
            *cAz=aM;
        }
        
        delete[] vect;
    }/*squint*/
    
    /* Rotate about x axis */
    void SPDSALCADataBinaryImporter::rotateX(float *vect, float theta)
    {
        float temp[3];
        
        temp[0]=vect[0];
        temp[1]=vect[1]*cos(theta)+vect[2]*sin(theta);
        temp[2]=vect[2]*cos(theta)-vect[1]*sin(theta);
        
        for(int i = 0; i < 3; ++i)
        {
            vect[i]=temp[i];
        }
    }/*rotateX*/
    
    /* Rotate about y axis */
    void SPDSALCADataBinaryImporter::rotateY(float *vect, float theta)
    {
        float temp[3];
        
        temp[0]=vect[0]*cos(theta)-vect[1]*sin(theta);
        temp[1]=vect[1];
        temp[2]=vect[0]*sin(theta)+vect[2]*cos(theta);
        
        for(int i = 0; i < 3; ++i)
        {
            vect[i]=temp[i];
        }
    }/*rotateY*/
    
    /** Rotate about z axis */
    void SPDSALCADataBinaryImporter::rotateZ(float *vect, float theta)
    {
        float temp[3];
        
        temp[0]=vect[0]*cos(theta)+vect[1]*sin(theta);
        temp[1]=vect[1]*cos(theta)-vect[0]*sin(theta);
        temp[2]=vect[2];
        
        for(int i = 0; i < 3; ++i)
        {
            vect[i]=temp[i];
        }
    }/*rotateZ*/
    
    /** Read data into array */
    char* SPDSALCADataBinaryImporter::readData(std::string inFilePath, int i, int *numb, int *nBins, int *length, SalcaHDRParams *options) throw(SPDIOException)
    {
        char *data = NULL;
        FILE *ipoo = NULL;
        
        //fprintf(stdout,"%d of %d Reading %s\n",i,options->nAz,inFilePath.c_str());  /*progress indicator*/
        
        /* Open the input file */
        if((ipoo=fopen(inFilePath.c_str(),"rb"))==NULL)
        {
            std::string message = std::string("Could not open file: \'") + inFilePath + std::string("\'");
            throw SPDIOException(message);
        }
        
        /* Determine the file length */
        if(fseek(ipoo,(long)0,SEEK_END))
        {
            throw SPDIOException("Could not determine the length of the file.");
        }
        *length=ftell(ipoo);
        
        if(((*nBins)*(*numb))>(*length))
        {
            fprintf(stderr,"File size mismatch\n");
            exit(1);
        }
        data = new char[*length];
        
        /* Now we know how long, read the file*/
        if(fseek(ipoo,(long)0,SEEK_SET))
        {
            throw SPDIOException("Could not restart the seek read to the beginning of the file.");
        }
        
        /* Read the data into the data array */
        if(fread(&(data[0]),sizeof(char),*length,ipoo)!=*length)
        {
            throw SPDIOException("Failed to read the data - reason unknown.");
        }
        
        if(ipoo)
        {
            fclose(ipoo);
            ipoo=NULL;
        }
        
        return(data);
    }/*readData*/
    
    /** Find outgoing pulse and check saturation */
    int SPDSALCADataBinaryImporter::findStart(int start,int end,char *satTest, char *data,int offset)
    {
        int i=0,b=0,e=0;  /*loop control and bounds*/
        int place=0;      /*array index*/
        int waveStart=0;  /*outgoing pulse bin*/
        char max=0;       /*max intensity*/
        char satThresh=0; /*saturation threshold*/
        char thresh=0;
        int sOffset=0;
        
        sOffset=7;   /*1.05 m*/
        waveStart=-1; /*start-sOffset;*/
        
        thresh=-110;  /*noise threshold, chosen from histograms*/
        satThresh=127;
        max=-125;
        
        *satTest=0;  /*not saturated by default*/
        b=start-50;   /*4.5m, from histograms*/
        if(b < 0)
        {
            b=0;  /*truncate at 0*/
        }
        
        e=start+sOffset;
        if(e > end)
        {
            e=end;
        }
        
        max=-125;
        for(i = b; i < e; ++i)
        {
            /*loop around where we think the pulse might be*/
            place=offset+i;
            if((data[place]>thresh)&&(data[place]>max))
            {
                /*outgoing peak*/
                max=data[place];
                waveStart=i;
            }
            /*outgoing peak test*/
            if((max>-125) && (data[place]<=thresh))
            {
                break;
            }
        }/*range loop*/
        
        for(i = start; i < end; ++i)
        {
            /*loop through full waveform to test for saturation*/
            place=offset+i;
            if(data[place]>=satThresh)
            {
                /*saturated*/
                *satTest=1;
                break;
            }  
        }/*saturation test loop*/
        
        return(waveStart);
    }
    
    SPDSALCADataBinaryImporter::~SPDSALCADataBinaryImporter()
    {
        
    }

}


