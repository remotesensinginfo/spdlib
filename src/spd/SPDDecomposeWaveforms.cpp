/*
 *  SPDDecomposeWaveforms.cpp
 *
 *  Created by Pete Bunting on 06/03/2012.
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

#include "spd/SPDDecomposeWaveforms.h"

namespace spdlib
{

    SPDDecomposeWaveforms::SPDDecomposeWaveforms()
    {
        
    }
        
    void SPDDecomposeWaveforms::decomposeWaveforms(std::string inFilePath, std::string outFilePath, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize, SPDDecompOption decompOption, boost::uint_fast32_t intThreshold, bool thresholdSet, bool noiseSet, uint_fast32_t window, boost::uint_fast32_t decayThres, float decayVal) throw(SPDException)
    {
        try 
        {
            SPDFileReader spdReader;
            SPDFile *inSPDFile = new SPDFile(inFilePath);
            spdReader.readHeaderInfo(inFilePath, inSPDFile);
                        
            SPDInitDecomposition *zeroCrossingInit = new SPDInitDecompositionZeroCrossing(decayVal, decayThres);
            
            SPDDecomposePulse *decomp = NULL;
            if(decompOption == spd_decomp_all)
            {
                decomp = new SPDDecomposePulseAll(zeroCrossingInit, intThreshold, thresholdSet, noiseSet);
            }
            else if(decompOption == spd_decomp_indvid)
            {
                decomp = new SPDDecomposePulseIndividually(zeroCrossingInit, window, intThreshold, thresholdSet);
            }
            else
            {
                throw SPDException("Decomposition option is unknown");
            }
            
            if(inSPDFile->getFileType() == SPD_UPD_TYPE)
            {
                SPDFile *spdFileOut = new SPDFile(outFilePath);
                SPDDataExporter *exporter = new SPDNoIdxFileWriter();
                SPDDecomposePulseImportProcessor *processor = new SPDDecomposePulseImportProcessor(decomp, exporter, spdFileOut);
                
                spdReader.readAndProcessAllData(inFilePath, inSPDFile, processor);
                
                processor->completeFileAndClose(inSPDFile);
                delete spdFileOut;
            }
            else
            {
                SPDPulseProcessor *pulseDecompProcessor = new SPDDecomposePulseColumnProcessor(decomp);            
                SPDSetupProcessPulses processPulses = SPDSetupProcessPulses(blockXSize, blockYSize, true);
                processPulses.processPulsesWithOutputSPD(pulseDecompProcessor, inSPDFile, outFilePath);
                delete pulseDecompProcessor;
            }
            
            delete decomp;
            delete zeroCrossingInit;
            delete inSPDFile;
        } 
        catch (SPDException &e) 
        {
            throw e;
        }
    }
    
    SPDDecomposeWaveforms::~SPDDecomposeWaveforms()
    {
        
    }
    
    
    

    SPDDecomposePulseAll::SPDDecomposePulseAll(SPDInitDecomposition *findInitPts, boost::uint_fast32_t intThreshold, bool thresholdSet, bool noiseSet):SPDDecomposePulse()
    {
        this->findInitPts = findInitPts;
        this->intThreshold = intThreshold;
        this->thresholdSet = thresholdSet;
        this->noiseSet = noiseSet;
		
		this->mpConfigValues = new mp_config();
		mpConfigValues->ftol = 1e-10;
		mpConfigValues->xtol = 1e-10;
		mpConfigValues->gtol = 1e-10;
		mpConfigValues->epsfcn = MP_MACHEP0;
		mpConfigValues->stepfactor = 100.0;
		mpConfigValues->covtol = 1e-14;
		mpConfigValues->maxiter = 10;
		mpConfigValues->maxfev = 0;
		mpConfigValues->nprint = 1;
		mpConfigValues->douserscale = 0;
		mpConfigValues->nofinitecheck = 0;
		mpConfigValues->iterproc = 0;
		
		this->mpResultsValues = new mp_result();
    }
        
    void SPDDecomposePulseAll::decompose(SPDPulse *pulse, SPDFile *spdFile) throw(SPDProcessingException)
    {
        bool debug_info = false;
		try
		{
			//Set noise threshold
            if(thresholdSet)
            {
                pulse->receiveWaveNoiseThreshold = intThreshold;
            }
            
            // Find init peaks
			std::vector<boost::uint_fast32_t> *peaks = findInitPts->findInitPoints(pulse->received, pulse->numOfReceivedBins, pulse->receiveWaveNoiseThreshold);
			if(debug_info)
			{
				std::cout << "Pulse " << pulse->pulseID << std::endl;
				std::cout << "Peaks = " << peaks->size() << std::endl;
			}
			
			// Fit Gaussians
			if(peaks->size() > 0)
			{
				int numOfParams = (peaks->size() * 3) + 1;
				/*
				 * p[0] = noise
                 * p[1] = amplitude
				 * p[2] = time offset
				 * p[3] = width
				 */
				double *parameters = new double[numOfParams];
				mp_par *paramConstraints = new mp_par[numOfParams];
                
                parameters[0] = pulse->receiveWaveNoiseThreshold;
                if(noiseSet)
                {
                    paramConstraints[0].fixed = false;
                }
                else
                {
                    paramConstraints[0].fixed = true;
                }
				paramConstraints[0].limited[0] = true;
				paramConstraints[0].limited[1] = true;
				paramConstraints[0].limits[0] = 0.0;
				paramConstraints[0].limits[1] = pulse->receiveWaveNoiseThreshold+0.01; // We can't detect peaks above the noise threshold
				paramConstraints[0].parname = const_cast<char*>(std::string("Noise").c_str());;
				paramConstraints[0].step = 0.01;
				paramConstraints[0].relstep = 0;
				paramConstraints[0].side = 0;
				paramConstraints[0].deriv_debug = 0;                
                
				int idx = 0;
				for(unsigned int i = 0; i < peaks->size(); ++i)
				{
					idx = (i*3)+1;
					parameters[idx] = pulse->received[peaks->at(i)] - pulse->receiveWaveNoiseThreshold; // Amplitude / Intensity
					double ampVar = parameters[idx] * 0.1;
					/*if(ampVar > 10)
                     {
                     ampVar = 10;
                     }*/
					if(ampVar < 1)
					{
						ampVar = 1;
					}
					paramConstraints[idx].fixed = false;
					paramConstraints[idx].limited[0] = true;
					paramConstraints[idx].limited[1] = true;
					paramConstraints[idx].limits[0] = parameters[idx] - ampVar;
					paramConstraints[idx].limits[1] = parameters[idx] + ampVar;
					paramConstraints[idx].parname = const_cast<char*>(std::string("Amplitude").c_str());;
					paramConstraints[idx].step = 0;
					paramConstraints[idx].relstep = 0;
					paramConstraints[idx].side = 0;
					paramConstraints[idx].deriv_debug = 0;
					
					parameters[idx+1] = peaks->at(i); // Time
					paramConstraints[idx+1].fixed = false;
					paramConstraints[idx+1].limited[0] = true;
					paramConstraints[idx+1].limited[1] = true;
					paramConstraints[idx+1].limits[0] = parameters[idx+1] - 5;
					paramConstraints[idx+1].limits[1] = parameters[idx+1] + 5;
					paramConstraints[idx+1].parname = const_cast<char*>(std::string("Time").c_str());;
					paramConstraints[idx+1].step = 0;
					paramConstraints[idx+1].relstep = 0;
					paramConstraints[idx+1].side = 0;
					paramConstraints[idx+1].deriv_debug = 0;
					
					parameters[idx+2] = spdFile->getTemporalBinSpacing()/2.0;
					paramConstraints[idx+2].fixed = false;
					paramConstraints[idx+2].limited[0] = true;
					paramConstraints[idx+2].limited[1] = true;
					paramConstraints[idx+2].limits[0] = 0.01;
					paramConstraints[idx+2].limits[1] = 10;
					paramConstraints[idx+2].parname = const_cast<char*>(std::string("Width").c_str());;
					paramConstraints[idx+2].step = 0.01;
					paramConstraints[idx+2].relstep = 0;
					paramConstraints[idx+2].side = 0;
					paramConstraints[idx+2].deriv_debug = 0;
                    
				}
				
				if(debug_info)
				{
					std::cout << "Pulse noise = " << parameters[0] << std::endl << std::endl;
                    for(unsigned int i = 0; i < peaks->size(); ++i)
					{
						idx = (i*3)+1;
						std::cout << "Point " << i+1 << " amplitude = " << parameters[idx] << std::endl;
						std::cout << "Point " << i+1 << " time = " << parameters[idx+1] << std::endl;
						std::cout << "Point " << i+1 << " width = " << parameters[idx+2] << std::endl << std::endl;
					}
				}
				
                // Determine number of real values
                unsigned int rCount = 0;
				for(boost::uint_fast16_t i = 0; i < pulse->numOfReceivedBins; ++i)
				{
					if(pulse->received[i] > 0)
                    {
                        rCount += 1;
                    }
				}
                
                // Contruct waveform for decomposition
                PulseWaveform *waveformData = new PulseWaveform();
				/*waveformData->time = new double[pulse->numOfReceivedBins];
                 waveformData->intensity = new double[pulse->numOfReceivedBins];
                 waveformData->error = new double[pulse->numOfReceivedBins];*/
				waveformData->time = new double[rCount];
				waveformData->intensity = new double[rCount];
				waveformData->error = new double[rCount];                
				boost::uint_fast16_t j = 0;
                for(boost::uint_fast16_t i = 0; i < pulse->numOfReceivedBins; ++i)
				{
					if(pulse->received[i] > 0)
                    {
    					waveformData->time[j] = (double)i;
	    				/*waveformData->intensity[j] = (pulse->received[i]*pulse->receiveWaveGain)+pulse->receiveWaveOffset;*/
                        waveformData->intensity[j] = (double)pulse->received[i];
		    			waveformData->error[j] = (double)1;
                        j += 1;
                    }
				}
				
				// Zero results structure...
				this->mpResultsValues->bestnorm = 0;
				this->mpResultsValues->orignorm = 0;
				this->mpResultsValues->niter = 0;
				this->mpResultsValues->nfev = 0;
				this->mpResultsValues->status = 0;
				this->mpResultsValues->npar = 0;
				this->mpResultsValues->nfree = 0;
				this->mpResultsValues->npegged = 0;
				this->mpResultsValues->nfunc = 0;
				this->mpResultsValues->resid = 0;
				this->mpResultsValues->xerror = 0;
				this->mpResultsValues->covar = 0; // Not being retrieved
				
				/*
				 * int m     - number of data points
				 * int npar  - number of parameters
				 * double *xall - parameters values (initial values and then best fit values)
				 * mp_par *pars - Constrains
				 * mp_config *config - Configuration parameters
				 * void *private_data - Waveform data structure
				 * mp_result *result - diagnostic info from function
				 */
				/*int returnCode = mpfit(gaussianSum, pulse->numOfReceivedBins, numOfParams, parameters, paramConstraints, this->mpConfigValues, waveformData, this->mpResultsValues);*/
				int returnCode = mpfit(gaussianSum, rCount, numOfParams, parameters, paramConstraints, this->mpConfigValues, waveformData, this->mpResultsValues);
				
				if(debug_info)
				{
					if(returnCode == MP_OK_CHI)
					{
						std::cout << "mpfit - Convergence in chi-square value.\n";
					}
					else if(returnCode == MP_OK_PAR)
					{
						std::cout << "mpfit - Convergence in parameter value.\n";
					}
					else if(returnCode == MP_OK_BOTH)
					{
						std::cout << "mpfit - Convergence in chi-square and parameter value.\n";
					}
					else if(returnCode == MP_OK_DIR)
					{
						std::cout << "mpfit - Convergence in orthogonality.\n";
					}
					else if(returnCode == MP_MAXITER)
					{
						std::cout << "mpfit - Maximum number of iterations reached.\n";
					}
					else if(returnCode == MP_FTOL)
					{
						std::cout << "mpfit - ftol is too small; cannot make further improvements.\n";
					}
					else if(returnCode == MP_XTOL)
					{
						std::cout << "mpfit - xtol is too small; cannot make further improvements.\n";
					}
					else if(returnCode == MP_XTOL)
					{
						std::cout << "mpfit - gtol is too small; cannot make further improvements.\n";
					}
					else 
					{
						std::cout << "An error has probably occurred - wait for exception...\n";
					}
                    
					
					std::cout << "Run Results (MPFIT version: " << this->mpResultsValues->version << "):\n";
					std::cout << "Final Chi-Squaured = " << this->mpResultsValues->bestnorm << std::endl;
					std::cout << "Start Chi-Squaured = " << this->mpResultsValues->orignorm << std::endl;
					std::cout << "Num Iterations = " << this->mpResultsValues->niter << std::endl;
					std::cout << "Num Func Evals = " << this->mpResultsValues->nfev << std::endl;
					std::cout << "Status Fit Code = " << this->mpResultsValues->status << std::endl;
					std::cout << "Num Params = " << this->mpResultsValues->npar << std::endl;
					std::cout << "Num Free Params = " << this->mpResultsValues->nfree << std::endl;
					std::cout << "Num Pegged Params = " << this->mpResultsValues->npegged << std::endl;
					std::cout << "Num Residuals Params = " << this->mpResultsValues->nfunc << std::endl << std::endl;
				}
				
				if((returnCode == MP_OK_CHI) | (returnCode == MP_OK_PAR) |
                   (returnCode == MP_OK_BOTH) | (returnCode == MP_OK_DIR) |
                   (returnCode == MP_MAXITER) | (returnCode == MP_FTOL)
                   | (returnCode == MP_XTOL) | (returnCode == MP_XTOL))
				{
					// MP Fit completed.. On on debug_info for more information. 
				}
				else if(returnCode == MP_ERR_INPUT)
				{
					throw SPDException("mpfit - Check inputs.");
				}
				else if(returnCode == MP_ERR_NAN)
				{
					throw SPDException("mpfit - Sum of Gaussians function produced NaN value.");
				}
				else if(returnCode == MP_ERR_FUNC)
				{
					throw SPDException("mpfit - No Sum of Gaussians function was supplied.");
				}
				else if(returnCode == MP_ERR_NPOINTS)
				{
					throw SPDException("mpfit - No data points were supplied.");
				}
				else if(returnCode == MP_ERR_NFREE)
				{
					throw SPDException("mpfit - No parameters are free - i.e., nothing to optimise!");
				}
				else if(returnCode == MP_ERR_MEMORY)
				{
					throw SPDException("mpfit - memory allocation error - may have run out!");
				}
				else if(returnCode == MP_ERR_INITBOUNDS)
				{
					throw SPDException("mpfit - Initial parameter values inconsistant with constraints.");
				}
				else if(returnCode == MP_ERR_PARAM)
				{
					throw SPDException("mpfit - An error has occur with an input parameter.");
				}
				else if(returnCode == MP_ERR_DOF)
				{
					throw SPDException("mpfit - Not enough degrees of freedom.");
				}
				else 
				{
					std::cout << "Return code is :" << returnCode << " - this can not been defined!\n";
				}
				
				if(debug_info)
				{
					std::cout << "Pulse noise = " << parameters[0] << std::endl << std::endl;
                    for(unsigned int i = 0; i < peaks->size(); ++i)
					{
						idx = (i*3)+1;
						std::cout << "Point " << i+1 << " amplitude = " << parameters[idx] << std::endl;
						std::cout << "Point " << i+1 << " time = " << parameters[idx+1] << std::endl;
						std::cout << "Point " << i+1 << " width = " << parameters[idx+2] << std::endl << std::endl;
					}
				}
				
				double tmpX = 0;
				double tmpY = 0;
				double tmpZ = 0;
				float peakTime = 0;
				
				SPDPointUtils ptUtils;
				//pulse->pts = new std::vector<SPDPoint*>();
				pulse->pts->reserve(peaks->size());
				pulse->numberOfReturns = peaks->size();
                pulse->receiveWaveNoiseThreshold = parameters[0];
                
				for(unsigned int i = 0; i < peaks->size(); ++i)
				{
					idx = (i*3)+1;
                    peakTime = parameters[idx+1] * spdFile->getTemporalBinSpacing();
					SPDPoint *pt = new SPDPoint();
					ptUtils.initSPDPoint(pt);
					pt->returnID = i;
					pt->gpsTime = pulse->gpsTime + peakTime;
					SPDConvertToCartesian(pulse->zenith, pulse->azimuth, (pulse->rangeToWaveformStart + (SPD_SPEED_OF_LIGHT_NS * (peakTime/2.0))), pulse->x0, pulse->y0, pulse->z0, &tmpX, &tmpY, &tmpZ);
					pt->x = tmpX;
					pt->y = tmpY;
					pt->z = tmpZ;
					pt->range = SPD_SPEED_OF_LIGHT_NS * (peakTime/2);
					pt->amplitudeReturn = parameters[idx];
					pt->widthReturn = ((parameters[idx+2] * spdFile->getTemporalBinSpacing()) * 10.0) * (2.0*sqrt(2.0*log(2.0)));
					pt->classification = SPD_UNCLASSIFIED;
                    
                    if(peakTime < 0)
                    {
                        pt->waveformOffset = 0;
                    }
                    else
                    {
                        try 
                        {
                            pt->waveformOffset = boost::numeric_cast<boost::uint_fast32_t>(peakTime * 1000.0);
                        }
                        catch(boost::numeric::negative_overflow& e) 
                        {
                            throw SPDIOException(e.what());
                        }
                        catch(boost::numeric::positive_overflow& e) 
                        {
                            throw SPDIOException(e.what());
                        }
                        catch(boost::numeric::bad_numeric_cast& e) 
                        {
                            throw SPDIOException(e.what());
                        }
					}
                    
					pulse->pts->push_back(pt);
				}
				
				delete[] waveformData->time;
				delete[] waveformData->intensity;
				delete[] waveformData->error;
				delete waveformData;
				
				delete[] parameters;
				delete[] paramConstraints;
			}			
			
			delete peaks;
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
		catch (SPDException &e) 
		{
			throw SPDIOException(e.what());
		}
    }
    
    SPDDecomposePulseAll::~SPDDecomposePulseAll()
    {
        delete this->mpConfigValues;
        delete this->mpResultsValues;
    }
    
    
    
    
    
    SPDDecomposePulseIndividually::SPDDecomposePulseIndividually(SPDInitDecomposition *findInitPts, boost::uint_fast16_t waveFitWindow, boost::uint_fast32_t intThreshold, bool thresholdSet)
    {
        this->findInitPts = findInitPts;
		this->waveFitWindow = waveFitWindow;
        this->intThreshold = intThreshold;
        this->thresholdSet = thresholdSet;
		
		this->mpConfigValues = new mp_config();
		mpConfigValues->ftol = 1e-10;
		mpConfigValues->xtol = 1e-10;
		mpConfigValues->gtol = 1e-10;
		mpConfigValues->epsfcn = MP_MACHEP0;
		mpConfigValues->stepfactor = 100.0;
		mpConfigValues->covtol = 1e-14;
		mpConfigValues->maxiter = 5;
		mpConfigValues->maxfev = 0;
		mpConfigValues->nprint = 1;
		mpConfigValues->douserscale = 0;
		mpConfigValues->nofinitecheck = 0;
		mpConfigValues->iterproc = 0;
		
		this->mpResultsValues = new mp_result();
    }
    
    void SPDDecomposePulseIndividually::decompose(SPDPulse *pulse, SPDFile *spdFile) throw(SPDProcessingException)
    {
        bool debug_info = false;
		try
		{
			//Set noise threshold
            if(thresholdSet)
            {
                pulse->receiveWaveNoiseThreshold = intThreshold;
            }
            
            // Find init peaks
			std::vector<boost::uint_fast32_t> *peaks = findInitPts->findInitPoints(pulse->received, pulse->numOfReceivedBins, pulse->receiveWaveNoiseThreshold);
            if(debug_info)
			{
				std::cout << "Pulse " << pulse->pulseID << std::endl;
				std::cout << "Peaks = " << peaks->size() << std::endl;
                std::cout << "pulse->waveNoiseThreshold = " << pulse->receiveWaveNoiseThreshold << std::endl << std::endl;
			}
			
			// Fit Gaussians
			if(peaks->size() > 0)
			{
				std::vector<SPDPoint*> *outPoints = new std::vector<SPDPoint*>();
				double *waveform = new double[pulse->numOfReceivedBins];
				
				for(boost::uint_fast16_t i = 0; i < pulse->numOfReceivedBins; ++i)
				{
					/*waveform[i] = (((double)pulse->received[i])*pulse->receiveWaveGain)+pulse->receiveWaveOffset;*/
                    waveform[i] = (double)pulse->received[i];
				}
				
				PulseWaveform *waveformData = NULL;
				
				boost::uint_fast16_t waveformIdxStart = 0;
				boost::uint_fast16_t waveformIdxEnd = 0;
				boost::uint_fast16_t waveformPeakIdx = 0;
				boost::uint_fast16_t waveformSampleLength = 0;
				boost::uint_fast32_t maxPeakIdx = 0;
				double maxPeakInt = 0;
				boost::uint_fast32_t maxPeaksVectIdx = 0;
				int numOfParams = 4;
				float peakTime = 0;
                float timeDiff = 0;
                boost::uint_fast16_t peakCount = 0;
				
				double tmpX = 0;
				double tmpY = 0;
				double tmpZ = 0;
				
				SPDPointUtils ptUtils;
				
				double *parameters = new double[numOfParams];
				mp_par *paramConstraints = new mp_par[numOfParams];
				
				while(peaks->size() > 0)
				{
					//std::cout << "peaks->size() = " << peaks->size() << std::endl;
					
					// Find the peak with the highest intensity
					for(unsigned int i = 0; i < peaks->size(); ++i)
					{
						if(i == 0)
						{
							maxPeakIdx = peaks->at(i);
							maxPeakInt = waveform[peaks->at(i)];
							maxPeaksVectIdx = i;
						}
						else if(waveform[peaks->at(i)] > maxPeakIdx)
						{
							maxPeakIdx = peaks->at(i);
							maxPeakInt = waveform[peaks->at(i)];
							maxPeaksVectIdx = i;
						}
					}
                    
                    //std::cout << "maxPeakInt = " << maxPeakInt << std::endl;
					
					// Remove peak from list
					std::vector<boost::uint_fast32_t>::iterator iterPeaks = peaks->begin();
					iterPeaks += maxPeaksVectIdx;
					peaks->erase(iterPeaks);
					
					// Define region either side of the peak (for fitting)
					if(((int_fast32_t)maxPeakIdx) - ((int_fast32_t)waveFitWindow) < 0)
					{
						waveformIdxStart = 0;
						waveformPeakIdx = waveFitWindow - maxPeakIdx;
					}
					else 
					{
						waveformIdxStart = maxPeakIdx - waveFitWindow;
						waveformPeakIdx = waveFitWindow;
					}
					
					if(((int_fast32_t)maxPeakIdx) + ((int_fast32_t)waveFitWindow+1) >= pulse->numOfReceivedBins)
					{
						waveformIdxEnd = pulse->numOfReceivedBins-1;
						waveformPeakIdx = waveFitWindow;
					}
					else 
					{
						waveformIdxEnd = maxPeakIdx + waveFitWindow + 1;
						waveformPeakIdx = waveFitWindow;
					}
					
					waveformSampleLength = waveformIdxEnd - waveformIdxStart;
					
					parameters[1] = maxPeakInt - pulse->receiveWaveNoiseThreshold; // Amplitude / Intensity
					double ampVar = parameters[1] * 0.1;
					
                    
                    
					if(ampVar < 1)
					{
						ampVar = 1;
					}
					
					parameters[0] = pulse->receiveWaveNoiseThreshold;
					paramConstraints[0].fixed = true;
					paramConstraints[0].limited[0] = true;
					paramConstraints[0].limited[1] = true;
					paramConstraints[0].limits[0] = 0.0;
					paramConstraints[0].limits[1] = pulse->receiveWaveNoiseThreshold+0.01; // Can't detect peaks below the noise threshold
					paramConstraints[0].parname = const_cast<char*>(std::string("Noise").c_str());
					paramConstraints[0].step = 0.01;
					paramConstraints[0].relstep = 0;
					paramConstraints[0].side = 0;
					paramConstraints[0].deriv_debug = 0;
                    
                    paramConstraints[1].fixed = false;
					paramConstraints[1].limited[0] = true;
					paramConstraints[1].limited[1] = true;
					paramConstraints[1].limits[0] = parameters[1] - ampVar;
					paramConstraints[1].limits[1] = parameters[1] + ampVar;
					paramConstraints[1].parname = const_cast<char*>(std::string("Amplitude").c_str());;
					paramConstraints[1].step = 0;
					paramConstraints[1].relstep = 0;
					paramConstraints[1].side = 0;
					paramConstraints[1].deriv_debug = 0;
					
					parameters[2] = waveformPeakIdx; // Time
					paramConstraints[2].fixed = false;
					paramConstraints[2].limited[0] = true;
					paramConstraints[2].limited[1] = true;
					paramConstraints[2].limits[0] = parameters[2] - 2;
					paramConstraints[2].limits[1] = parameters[2] + 2;
					paramConstraints[2].parname = const_cast<char*>(std::string("Time").c_str());;
					paramConstraints[2].step = 0;
					paramConstraints[2].relstep = 0;
					paramConstraints[2].side = 0;
					paramConstraints[2].deriv_debug = 0;
					
					parameters[3] = spdFile->getTemporalBinSpacing()/2.0;
					paramConstraints[3].fixed = false;
					paramConstraints[3].limited[0] = true;
					paramConstraints[3].limited[1] = true;
					paramConstraints[3].limits[0] = 0.01;
					paramConstraints[3].limits[1] = 10;
					paramConstraints[3].parname = const_cast<char*>(std::string("Width").c_str());
					paramConstraints[3].step = 0.01;
					paramConstraints[3].relstep = 0;
					paramConstraints[3].side = 0;
					paramConstraints[3].deriv_debug = 0;
                    
					
					waveformData = new PulseWaveform();
					waveformData->time = new double[waveformSampleLength];
					waveformData->intensity = new double[waveformSampleLength];
					waveformData->error = new double[waveformSampleLength];
					for(boost::uint_fast16_t waveIdx = waveformIdxStart, i = 0; waveIdx < waveformIdxEnd; ++i, ++waveIdx)
					{
						waveformData->time[i] = (double)i;
						waveformData->intensity[i] = (double)waveform[waveIdx];
						waveformData->error[i] = (double)1;
					}
					
					mpResultsValues->bestnorm = 0;
					mpResultsValues->orignorm = 0;
					mpResultsValues->niter = 0;
					mpResultsValues->nfev = 0;
					mpResultsValues->status = 0;
					mpResultsValues->npar = 0;
					mpResultsValues->nfree = 0;
					mpResultsValues->npegged = 0;
					mpResultsValues->nfunc = 0;
					mpResultsValues->resid = 0;
					mpResultsValues->xerror = 0;
					mpResultsValues->covar = 0; // Not being retrieved
					
					/*
					 * int m     - number of data points
					 * int npar  - number of parameters
					 * double *xall - parameters values (initial values and then best fit values)
					 * mp_par *pars - Constrains
					 * mp_config *config - Configuration parameters
					 * void *private_data - Waveform data structure
					 * mp_result *result - diagnostic info from function
					 */
					int returnCode = mpfit(gaussianSum, waveformSampleLength, numOfParams, parameters, paramConstraints, mpConfigValues, waveformData, mpResultsValues);
					if((returnCode == MP_OK_CHI) | (returnCode == MP_OK_PAR) |
					   (returnCode == MP_OK_BOTH) | (returnCode == MP_OK_DIR) |
					   (returnCode == MP_MAXITER) | (returnCode == MP_FTOL)
					   | (returnCode == MP_XTOL) | (returnCode == MP_XTOL))
					{
						// MP Fit completed.. On on debug_info for more information. 
                        if(debug_info)
                        {
                            if(returnCode == MP_OK_CHI)
                            {
                                std::cout << "mpfit - Convergence in chi-square value.\n";
                            }
                            else if(returnCode == MP_OK_PAR)
                            {
                                std::cout << "mpfit - Convergence in parameter value.\n";
                            }
                            else if(returnCode == MP_OK_BOTH)
                            {
                                std::cout << "mpfit - Convergence in chi-square and parameter value.\n";
                            }
                            else if(returnCode == MP_OK_DIR)
                            {
                                std::cout << "mpfit - Convergence in orthogonality.\n";
                            }
                            else if(returnCode == MP_MAXITER)
                            {
                                std::cout << "mpfit - Maximum number of iterations reached.\n";
                            }
                            else if(returnCode == MP_FTOL)
                            {
                                std::cout << "mpfit - ftol is too small; cannot make further improvements.\n";
                            }
                            else if(returnCode == MP_XTOL)
                            {
                                std::cout << "mpfit - xtol is too small; cannot make further improvements.\n";
                            }
                            else if(returnCode == MP_XTOL)
                            {
                                std::cout << "mpfit - gtol is too small; cannot make further improvements.\n";
                            }
                            else 
                            {
                                std::cout << "An error has probably occurred - wait for exception...\n";
                            }
                            
                            
                            std::cout << "Run Results (MPFIT version: " << this->mpResultsValues->version << "):\n";
                            std::cout << "Final Chi-Squaured = " << this->mpResultsValues->bestnorm << std::endl;
                            std::cout << "Start Chi-Squaured = " << this->mpResultsValues->orignorm << std::endl;
                            std::cout << "Num Iterations = " << this->mpResultsValues->niter << std::endl;
                            std::cout << "Num Func Evals = " << this->mpResultsValues->nfev << std::endl;
                            std::cout << "Status Fit Code = " << this->mpResultsValues->status << std::endl;
                            std::cout << "Num Params = " << this->mpResultsValues->npar << std::endl;
                            std::cout << "Num Free Params = " << this->mpResultsValues->nfree << std::endl;
                            std::cout << "Num Pegged Params = " << this->mpResultsValues->npegged << std::endl;
                            std::cout << "Num Residuals Params = " << this->mpResultsValues->nfunc << std::endl << std::endl;
                        }
					}
					else if(returnCode == MP_ERR_INPUT)
					{
						throw SPDException("mpfit - Check inputs.");
					}
					else if(returnCode == MP_ERR_NAN)
					{
						throw SPDException("mpfit - Sum of Gaussians function produced NaN value.");
					}
					else if(returnCode == MP_ERR_FUNC)
					{
						throw SPDException("mpfit - No Sum of Gaussians function was supplied.");
					}
					else if(returnCode == MP_ERR_NPOINTS)
					{
						throw SPDException("mpfit - No data points were supplied.");
					}
					else if(returnCode == MP_ERR_NFREE)
					{
						throw SPDException("mpfit - No parameters are free - i.e., nothing to optimise!");
					}
					else if(returnCode == MP_ERR_MEMORY)
					{
						throw SPDException("mpfit - memory allocation error - may have run out!");
					}
					else if(returnCode == MP_ERR_INITBOUNDS)
					{
						throw SPDException("mpfit - Initial parameter values inconsistant with constraints.");
					}
					else if(returnCode == MP_ERR_PARAM)
					{
						throw SPDException("mpfit - An error has occur with an input parameter.");
					}
					else if(returnCode == MP_ERR_DOF)
					{
						throw SPDException("mpfit - Not enough degrees of freedom.");
					}
					else 
					{
						std::cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}
					
					if(debug_info)
                    {
                        std::cout << "Fitted Noise: " << parameters[0] << std::endl;
                        std::cout << "Fitted Ampitude: " << parameters[1] << std::endl;
                        std::cout << "Fitted Time: " << parameters[2] << std::endl;
                        std::cout << "Fitted Width: " << parameters[3] << std::endl << std::endl;
                    }
                    
					// Create Point
					timeDiff = ((float)parameters[2]) - ((float)waveformPeakIdx);
					peakTime = (((float)maxPeakIdx) + timeDiff) * spdFile->getTemporalBinSpacing();
                    
                    if(debug_info)
                    {
                        std::cout << "Time Diff: " << timeDiff << std::endl;
                        std::cout << "Peak Time: " << peakTime << std::endl;
                    }
					
					SPDPoint *pt = new SPDPoint();
					ptUtils.initSPDPoint(pt);
					pt->returnID = peakCount++;
					pt->gpsTime = pulse->gpsTime + peakTime;
                    
                    if(debug_info)
                    {
                        std::cout << "pulse->zenith = " << pulse->zenith << std::endl;
                        std::cout << "pulse->azimuth = " << pulse->azimuth << std::endl;
                        
                        std::cout << "pulse->rangeToWaveformStart = " << pulse->rangeToWaveformStart << std::endl;
                        std::cout << "peakTime = " << peakTime << std::endl;
                        std::cout << "pulse->x0 = " << pulse->x0 << std::endl;
                        std::cout << "pulse->y0 = " << pulse->y0 << std::endl;
                        std::cout << "pulse->z0 = " << pulse->z0 << std::endl;
                    }
                    
					SPDConvertToCartesian(pulse->zenith, pulse->azimuth, (pulse->rangeToWaveformStart + (SPD_SPEED_OF_LIGHT_NS * (peakTime/2.0))), pulse->x0, pulse->y0, pulse->z0, &tmpX, &tmpY, &tmpZ);
					if(debug_info)
                    {
                        std::cout << "Pt Location [" << tmpX << ", " << tmpY << ", " << tmpZ << "]\n";
                    }
                    pt->x = tmpX;
					pt->y = tmpY;
					pt->z = tmpZ;
					pt->range = SPD_SPEED_OF_LIGHT_NS * (peakTime/2);
                    if(debug_info)
                    {
                        std::cout << "pt->range = " << pt->range << std::endl;
                    }
                    pt->classification = SPD_UNCLASSIFIED;
					
					if((parameters[1] < 0) | (waveform[maxPeakIdx] < 0))
					{
						pt->amplitudeReturn = 0;
					}
					else 
					{
						/*if(waveform[maxPeakIdx] < 0)
                         {
                         std::cout << "Pulse " << pulse->pulseID << " has ampulitude = " << parameters[1] << " on return." << std::endl;
                         std::cout << "Start index = " << maxPeakIdx << " has amp = " << waveform[maxPeakIdx] << std::endl;
                         }*/
						
						pt->amplitudeReturn = parameters[1];
					}
                    
					pt->widthReturn = ((parameters[3] * spdFile->getTemporalBinSpacing()) * 10.0) * (2.0*sqrt(2.0*log(2.0)));
					
                    if(peakTime < 0)
                    {
                        pt->waveformOffset = 0;
                    }
                    else
                    {
                        try 
                        {
                            pt->waveformOffset = boost::numeric_cast<boost::uint_fast32_t>(peakTime * 1000.0);
                        }
                        catch(boost::numeric::negative_overflow& e) 
                        {
                            throw SPDIOException(e.what());
                        }
                        catch(boost::numeric::positive_overflow& e) 
                        {
                            throw SPDIOException(e.what());
                        }
                        catch(boost::numeric::bad_numeric_cast& e) 
                        {
                            throw SPDIOException(e.what());
                        }
                    }
					outPoints->push_back(pt);
					
					// Remove Fitted Gaussian from waveform
					double fittedPeakIdxArray = ((float)maxPeakIdx) + timeDiff;
					
					for(boost::uint_fast16_t i = 0; i < pulse->numOfReceivedBins; ++i)
					{
						waveform[i] -= (parameters[1] * exp((-1.0)*
                                                            (
                                                             pow(i - fittedPeakIdxArray,2)
                                                             /
                                                             (2.0 * pow(parameters[3], 2))
                                                             )));
					}
					
					delete[] waveformData->time;
					delete[] waveformData->intensity;
					delete[] waveformData->error;
					delete waveformData;
				}
				
				// Add points to pulse
                std::sort(outPoints->begin(), outPoints->end(), cmpSPDPointTime);
				
				//pulse->pts = new std::vector<SPDPoint*>();
				pulse->pts->reserve(outPoints->size());
				pulse->numberOfReturns = outPoints->size();
                //pulse->waveNoiseThreshold = parameters[0]
				boost::uint_fast16_t idCount = 0;
				for(std::vector<SPDPoint*>::iterator iterPts = outPoints->begin(); iterPts != outPoints->end();)
				{
					(*iterPts)->returnID = idCount++;
					pulse->pts->push_back(*iterPts);
					iterPts = outPoints->erase(iterPts);
				}
				
				delete[] parameters;
				delete[] paramConstraints;
				delete[] waveform;
				delete outPoints;
			}			
			
			delete peaks;
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
		catch (SPDException &e) 
		{
			throw SPDIOException(e.what());
		}
    }
        
    SPDDecomposePulseIndividually::~SPDDecomposePulseIndividually()
    {
        delete this->mpConfigValues;
        delete this->mpResultsValues;
    }





}
