/*
 *  SPDMathsUtils.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 22/06/2011.
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

#include "spd/SPDMathsUtils.h"

namespace spdlib {
	
    
    SPDMathsUtils::SPDMathsUtils()
    {
        
    }
    
    void SPDMathsUtils::applySavitzkyGolaySmoothing(float *dataValuesY, float *dataValuesX, boost::uint_fast32_t numValues, boost::uint_fast16_t winHSize, boost::uint_fast16_t order, bool removeLTZeros) throw(SPDProcessingException)
    {
        int numCols = 2;
		int numRows = 0;
		int startVal = 0;
		
		SPDPolyFit polyFit;
		gsl_vector *coefficients = NULL;
		gsl_matrix *inputValues = NULL;
        
        float *outputValues = new float[numValues];
		
		for(boost::uint_fast32_t i = 0; i < numValues; ++i)
		{
			if((((int_fast32_t)i)-winHSize) < 0)
			{
				numRows = (winHSize + 1) + i;
				startVal = 0;
			}
			else if((i+winHSize) >= numValues)
			{
				numRows = (winHSize + 1) + (numValues - i);
				startVal = i - winHSize;
			}
			else 
			{
				numRows = (winHSize * 2) + 1;
				startVal = i - winHSize;
			}
            
			inputValues = gsl_matrix_alloc (numRows,numCols);
			for(int j = 0; j < numRows; ++j)
			{
				gsl_matrix_set (inputValues, j, 0, dataValuesX[(startVal+j)]);
				gsl_matrix_set (inputValues, j, 1, dataValuesY[(startVal+j)]);
			}
			
			coefficients = polyFit.PolyfitOneDimensionQuiet(order, inputValues);	
			
			double yPredicted = 0;
			for(int j = 0; j < order ; j++)
			{
				double xPow = pow(dataValuesX[i], j); // x^n;
				double coeff = gsl_vector_get(coefficients, j); // a_n
				double coeffXPow = coeff * xPow; // a_n * x^n				
				yPredicted = yPredicted + coeffXPow;
			}
            if((dataValuesY[i] == 0) & ((yPredicted > 100000) | (yPredicted < -100000)))
            {
                outputValues[i] = 0;
            }
            else
            {
                outputValues[i] = yPredicted;
            }
			
			gsl_matrix_free(inputValues);
			gsl_vector_free(coefficients);
		}
        
        for(boost::uint_fast32_t i = 0; i < numValues; ++i)
		{
            if(removeLTZeros)
            {
                if(outputValues[i] < 0)
                {
                    outputValues[i] = 0;
                }
            }
            dataValuesY[i] = outputValues[i];
        }
        delete[] outputValues;

    }
    
    std::vector<GaussianDecompReturnType*>* SPDMathsUtils::fitGaussianMixture(SPDInitDecomposition *initDecomp, float minimumGaussianGap, float *dataValues, float *dataIntervals, boost::uint_fast32_t nVals, float intThreshold) throw(SPDProcessingException)
    {
        if(nVals < 5)
        {
            throw SPDProcessingException("Less than 5 values were passed to the fitGaussianMixture function, insuficiant for fitting.");
        }
        
        std::vector<GaussianDecompReturnType*> *gaussianPeaks = new std::vector<GaussianDecompReturnType*>();
        
        /*for(boost::uint_fast32_t i = 0; i < nVals; ++i)
        {
            if(i == 0)
            {
                std::cout << dataValues[i];
            }
            else
            {
                std::cout << "," << dataValues[i];
            }
        }
        std::cout << std::endl;
        
        for(boost::uint_fast32_t i = 0; i < nVals; ++i)
        {
            if(i == 0)
            {
                std::cout << dataIntervals[i];
            }
            else
            {
                std::cout << "," << dataIntervals[i];
            }
        }
        std::cout << std::endl;*/
        
        try 
        {
            std::vector<boost::uint_fast32_t> *peaks = new std::vector<boost::uint_fast32_t>();
            std::vector<boost::uint_fast32_t> *initPeaks = initDecomp->findInitPoints(dataValues, nVals, intThreshold);
                        
            //std::cout << "There are " << initPeaks->size() << " initial peaks.\n";
            
            /*for(unsigned int i = 0; i < initPeaks->size(); ++i)
            {
                peaks->push_back(initPeaks->at(i));
            }*/
            
            if(initPeaks->size() == 1)
            {
                for(unsigned int i = 0; i < initPeaks->size(); ++i)
                {
                    peaks->push_back(initPeaks->at(i));
                }
            }
			else if(initPeaks->size() > 1)
			{
                if(minimumGaussianGap > 0)
                {
                    float intervalDiff = 0;
                    for(unsigned int i = 0; i < initPeaks->size()-1; ++i)
                    {
                        //std::cout << "dataIntervals[" << initPeaks->at(i) << "]: " << dataIntervals[initPeaks->at(i)] << std::endl;
                        //std::cout << "dataIntervals[" << initPeaks->at(i+1) << "]: " << dataIntervals[initPeaks->at(i+1)] << std::endl;
                        intervalDiff = dataIntervals[initPeaks->at(i+1)] - dataIntervals[initPeaks->at(i)];
                        //std::cout << "Diff: " << intervalDiff << std::endl;
                        if(intervalDiff <= minimumGaussianGap)
                        {
                            //std::cout << "dataValues[" << initPeaks->at(i) << "]: " << dataValues[initPeaks->at(i)] << std::endl;
                            //std::cout << "dataValues[" << initPeaks->at(i+1) << "]: " << dataValues[initPeaks->at(i+1)] << std::endl;
                            if(dataValues[initPeaks->at(i)] >= dataValues[initPeaks->at(i+1)])
                            {
                                peaks->push_back(initPeaks->at(i));
                                //std::cout << "Keep " << initPeaks->at(i) << std::endl;
                            }
                            else
                            {
                                peaks->push_back(initPeaks->at(i+1));
                                //std::cout << "Keep " << initPeaks->at(i+1) << std::endl;
                            }
                            ++i;
                        }
                        else
                        {
                            peaks->push_back(initPeaks->at(i));
                        }
                    }
                    if(intervalDiff > minimumGaussianGap)
                    {
                        peaks->push_back(initPeaks->at(initPeaks->size()-1));
                    }
                }
                else
                {
                    for(unsigned int i = 0; i < initPeaks->size(); ++i)
                    {
                        peaks->push_back(initPeaks->at(i));
                    }
                }
            }
            
            //std::cout << "There are " << peaks->size() << " peaks.\n";
            
            if(peaks->size() > 0)
            {
                mp_config *mpConfigValues = new mp_config();
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
                
                mp_result *mpResultsValues = new mp_result();
                
				int numOfParams = peaks->size() * 3;
				/*
				 * p[0] = amplitude
				 * p[1] = time offset
				 * p[2] = width
				 */
				double *parameters = new double[numOfParams];
				mp_par *paramConstraints = new mp_par[numOfParams];
				int idx = 0;
				for(unsigned int i = 0; i < peaks->size(); ++i)
				{
					idx = i*3;
					parameters[idx] = dataValues[peaks->at(i)]; // Amplitude / Intensity
					double ampVar = parameters[idx] * 0.1;
					if(ampVar > 10)
					{
						ampVar = 10;
					}
					else if(ampVar < 1)
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
					
					parameters[idx+1] = dataIntervals[peaks->at(i)]; // Time
					paramConstraints[idx+1].fixed = false;
					paramConstraints[idx+1].limited[0] = true;
					paramConstraints[idx+1].limited[1] = true;
					paramConstraints[idx+1].limits[0] = parameters[idx+1] - 5;
					paramConstraints[idx+1].limits[1] = parameters[idx+1] + 5;
					paramConstraints[idx+1].parname = const_cast<char*>(std::string("Height").c_str());;
					paramConstraints[idx+1].step = 0;
					paramConstraints[idx+1].relstep = 0;
					paramConstraints[idx+1].side = 0;
					paramConstraints[idx+1].deriv_debug = 0;
					
					parameters[idx+2] = 0.5;
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
                    
                    
                    //std::cout << "Peak: " << i << std::endl;
                    //std::cout << "Amp: " << parameters[idx] << std::endl;
                    //std::cout << "Height: " << parameters[idx+1] << std::endl;
                    //std::cout << "Width: " << parameters[idx+2] << std::endl;
				}
				
				PulseWaveform *waveformData = new PulseWaveform();
				waveformData->time = new double[nVals];
				waveformData->intensity = new double[nVals];
				waveformData->error = new double[nVals];
				for(boost::uint_fast16_t i = 0; i < nVals; ++i)
				{
					waveformData->time[i] = dataIntervals[i];
					waveformData->intensity[i] = dataValues[i];
					waveformData->error[i] = 1;
				}
				
				// Zero results structure...
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
                //std::cout << "Called mpfit\n";
				int returnCode = mpfit(gaussianSumNoNoise, nVals, numOfParams, parameters, paramConstraints, mpConfigValues, waveformData, mpResultsValues);
				//std::cout << "mpfit returned\n";
                if((returnCode == MP_OK_CHI) | (returnCode == MP_OK_PAR) |
                   (returnCode == MP_OK_BOTH) | (returnCode == MP_OK_DIR) |
                   (returnCode == MP_MAXITER) | (returnCode == MP_FTOL)
                   | (returnCode == MP_XTOL) | (returnCode == MP_XTOL))
				{
					// MP Fit completed..
                    
                    /* DEBUG INFO...
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
                     
                     
                     std::cout << "Run Results (MPFIT version: " << mpResultsValues->version << "):\n";
                     std::cout << "Final Chi-Squaured = " << mpResultsValues->bestnorm << std::endl;
                     std::cout << "Start Chi-Squaured = " << mpResultsValues->orignorm << std::endl;
                     std::cout << "Num Iterations = " << mpResultsValues->niter << std::endl;
                     std::cout << "Num Func Evals = " << mpResultsValues->nfev << std::endl;
                     std::cout << "Status Fit Code = " << mpResultsValues->status << std::endl;
                     std::cout << "Num Params = " << mpResultsValues->npar << std::endl;
                     std::cout << "Num Free Params = " << mpResultsValues->nfree << std::endl;
                     std::cout << "Num Pegged Params = " << mpResultsValues->npegged << std::endl;
                     std::cout << "Num Residuals Params = " << mpResultsValues->nfunc << std::endl << std::endl;
                     */
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
                    std::cout << "Returned values may have errors associated with...\n";
					std::cout << "Return code was :" << returnCode << " - this can not been defined!\n";
				}
                
                for(unsigned int i = 0; i < peaks->size(); ++i)
				{
                    idx = i*3;
                    GaussianDecompReturnType *gausReturn = new GaussianDecompReturnType();
                    gausReturn->gaussianAmplitude = parameters[idx];
                    gausReturn->gaussianWidth = parameters[idx+2] * (2.0*sqrt(2.0*log(2.0)));
                    gausReturn->axisInterval = parameters[idx+1];
                    //std::cout << "Peak: " << i << std::endl;
                    //std::cout << "Amp: " << parameters[idx] << std::endl;
                    //std::cout << "Height: " << parameters[idx+1] << std::endl;
                    //std::cout << "Width: " << parameters[idx+2] << std::endl;
                    //std::cout << "FWHM: " << gausReturn->gaussianWidth << std::endl;
                    
                    gaussianPeaks->push_back(gausReturn);
                }
                			
				delete[] waveformData->time;
				delete[] waveformData->intensity;
				delete[] waveformData->error;
				delete waveformData;
                
                delete mpConfigValues;
                delete mpResultsValues;
				
				delete[] parameters;
				delete[] paramConstraints;
			}
            
			delete peaks;
            delete initPeaks;
            
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
        catch (SPDException &e) 
        {
            throw SPDProcessingException(e.what());
        }
        
        return gaussianPeaks;
    }
    
    void SPDMathsUtils::decomposeSingleGaussian(boost::uint_fast32_t *waveform, boost::uint_fast16_t waveformLength, boost::uint_fast16_t waveFitWindow, float waveformTimeInterval, float *transAmp, float *transWidth, float *peakTime) throw(SPDProcessingException)
    {
        try 
        {
            boost::uint_fast32_t maxInt = 0;
            boost::uint_fast16_t maxIdx = 0;
            
            // Find max intensity - this will be the starting point of the decomposition.
            for(boost::uint_fast16_t i = 0; i < waveformLength; ++i)
            {
                if(i == 0)
                {
                    maxInt = waveform[i];
                    maxIdx = i;
                }
                else if(waveform[i] > maxInt)
                {
                    maxInt = waveform[i];
                    maxIdx = i;
                }
            }
            
            boost::uint_fast16_t waveformIdxStart = 0;
            boost::uint_fast16_t waveformIdxEnd = 0;
            boost::uint_fast16_t waveformPeakIdx = 0;
            
            if(((int_fast32_t)maxIdx) - ((int_fast32_t)waveFitWindow) < 0)
            {
                waveformIdxStart = 0;
                waveformPeakIdx = waveFitWindow - maxIdx;
            }
            else 
            {
                waveformIdxStart = maxIdx - waveFitWindow;
                waveformPeakIdx = waveFitWindow;
            }
            
            if(((int_fast32_t)maxIdx) + ((int_fast32_t)waveFitWindow+1) >= waveformLength)
            {
                waveformIdxEnd = waveformLength-1;
                waveformPeakIdx = waveFitWindow;
            }
            else 
            {
                waveformIdxEnd = maxIdx + waveFitWindow + 1;
                waveformPeakIdx = waveFitWindow;
            }
            
            boost::uint_fast16_t waveformSampleLength = waveformIdxEnd - waveformIdxStart;
            
            int numOfParams = 3; // i.e., 1 peak
            /*
             * p[0] = amplitude
             * p[1] = time offset
             * p[2] = width
             */
            double *parameters = new double[numOfParams];
            mp_par *paramConstraints = new mp_par[numOfParams];
            
            parameters[0] = maxInt; // Amplitude / Intensity
            double ampVar = parameters[0] * 0.1;
            if(ampVar > 10)
            {
                ampVar = 10;
            }
            else if(ampVar < 1)
            {
                ampVar = 1;
            }
            paramConstraints[0].fixed = false;
            paramConstraints[0].limited[0] = true;
            paramConstraints[0].limited[1] = true;
            paramConstraints[0].limits[0] = parameters[0] - ampVar;
            paramConstraints[0].limits[1] = parameters[0] + ampVar;
            paramConstraints[0].parname = const_cast<char*>(std::string("Amplitude").c_str());;
            paramConstraints[0].step = 0;
            paramConstraints[0].relstep = 0;
            paramConstraints[0].side = 0;
            paramConstraints[0].deriv_debug = 0;
            
            parameters[1] = waveformPeakIdx; // Time
                                             //std::cout << "Peak Time (Full): " << maxIdx << std::endl;
                                             //std::cout << "Peak Time (Local): " << waveformPeakIdx << std::endl;
            paramConstraints[1].fixed = false;
            paramConstraints[1].limited[0] = true;
            paramConstraints[1].limited[1] = true;
            paramConstraints[1].limits[0] = parameters[1] - 2;
            paramConstraints[1].limits[1] = parameters[1] + 2;
            paramConstraints[1].parname = const_cast<char*>(std::string("Time").c_str());;
            paramConstraints[1].step = 0;
            paramConstraints[1].relstep = 0;
            paramConstraints[1].side = 0;
            paramConstraints[1].deriv_debug = 0;
            
            parameters[2] = 0.5;
            paramConstraints[2].fixed = false;
            paramConstraints[2].limited[0] = true;
            paramConstraints[2].limited[1] = true;
            paramConstraints[2].limits[0] = 0.01;
            paramConstraints[2].limits[1] = 10;
            paramConstraints[2].parname = const_cast<char*>(std::string("Width").c_str());
            paramConstraints[2].step = 0.01;
            paramConstraints[2].relstep = 0;
            paramConstraints[2].side = 0;
            paramConstraints[2].deriv_debug = 0;
            
            PulseWaveform *waveformData = new PulseWaveform();
            waveformData->time = new double[waveformSampleLength];
            waveformData->intensity = new double[waveformSampleLength];
            waveformData->error = new double[waveformSampleLength];
            for(boost::uint_fast16_t waveIdx = waveformIdxStart, i = 0; waveIdx < waveformIdxEnd; ++i, ++waveIdx)
            {
                waveformData->time[i] = i;
                waveformData->intensity[i] = waveform[waveIdx];
                waveformData->error[i] = 1;
            }
            
            // Create and initise configure and results structures...			
            mp_config *mpConfigValues = new mp_config();
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
            
            mp_result *mpResultsValues = new mp_result();
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
            int returnCode = mpfit(gaussianSumNoNoise, waveformSampleLength, numOfParams, parameters, paramConstraints, mpConfigValues, waveformData, mpResultsValues);
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
            
            float timeDiff = ((float)parameters[1]) - ((float)waveformPeakIdx);
            
            *transAmp = parameters[0];
            *peakTime = (((float)maxIdx) + timeDiff) * waveformTimeInterval;
            *transWidth = (parameters[2] * 10) * (2.0*sqrt(2.0*log(2.0)));
            
            //std::cout << "Output peak time (local): " << parameters[1] << std::endl;
            //std::cout << "Output peak time (Global): " << *peakTime << std::endl << std::endl;
            
            delete mpConfigValues;
            delete mpResultsValues;
            
            delete[] waveformData->time;
            delete[] waveformData->intensity;
            delete[] waveformData->error;
            delete waveformData;
            
            delete[] parameters;
            delete[] paramConstraints;
        }
        catch (SPDException &e) 
        {
            throw e;
        }
    }
    
    bool SPDMathsUtils::rectangleIntersection(double xMin1, double xMax1, double yMin1, double yMax1, double xMin2, double xMax2, double yMin2, double yMax2)
    {
        //std::cout << "1 = [" << xMin1 << ", " << xMax1 << "][" << yMin1 << ", " << yMax1 << "]" << std::endl;
        //std::cout << "2 = [" << xMin2 << ", " << xMax2 << "][" << yMin2 << ", " << yMax2 << "]" << std::endl;
        
        double xMin = 0;
        double xMax = 0;
        double yMin = 0;
        double yMax = 0;
        
        if(xMin1 > xMin2)
        {
            xMin = xMin1;
        }
        else
        {
            xMin = xMin2;
        }
        
        if(yMin1 > yMin2)
        {
            yMin = yMin1;
        }
        else
        {
            yMin = yMin2;
        }
        
        if(xMax1 < xMax2)
        {
            xMax = xMax1;
        }
        else
        {
            xMax = xMax2;
        }
        
        if(yMax1 < yMax2)
        {
            yMax = yMax1;
        }
        else
        {
            yMax = yMax2;
        }
        
        //std::cout << "X = " << xMin << ", " << xMax << ":\t" << xMax - xMin << std::endl;
        //std::cout << "Y = " << yMin << ", " << yMax << ":\t" << yMax - yMin << std::endl << std::endl;
        
        bool intersect = true;
        if(xMax - xMin <= 0)
        {
            intersect = false;
        }
        
        if(yMax - yMin <= 0)
        {
            intersect = false;
        }
        
        return intersect;
    }
    
    SPDMathsUtils::~SPDMathsUtils()
    {
        
    }
    
    
    SPDInitDecompositionZeroCrossingSimple::SPDInitDecompositionZeroCrossingSimple(float decay):SPDInitDecomposition()
	{
		this->decay = decay;
	}
	
	std::vector<boost::uint_fast32_t>* SPDInitDecompositionZeroCrossingSimple::findInitPoints(boost::uint_fast32_t *waveform, boost::uint_fast16_t waveformLength, float intThreshold) throw(SPDException)
	{
		std::vector<boost::uint_fast32_t> *pts = new std::vector<boost::uint_fast32_t>();
		if(waveformLength > 0)
		{
			boost::uint_fast16_t waveformLengthLess1 = waveformLength - 1;
			float forGrad = 0;
			float backGrad = 0;
			bool firstPeak = true;
			boost::uint_fast32_t peakInt = 0;
			boost::uint_fast32_t peakTime = 0;
			boost::uint_fast32_t timeDiff = 0;
			float calcThreshold = 0;
			for(boost::uint_fast16_t i = 1; i < waveformLengthLess1; ++i)
			{
				forGrad = ((float)waveform[i]) - ((float)waveform[i-1]);
				backGrad = ((float)waveform[i+1]) - ((float)waveform[i]);
				if((forGrad > 0) & (backGrad < 0))
				{
					if(((float)waveform[i]) > intThreshold)
					{
						if(firstPeak)
						{
							pts->push_back(i);
							peakInt = ((float)waveform[i]);
							peakTime = i;
							firstPeak = false;
						}
						else 
						{
							timeDiff = i - peakTime;
							calcThreshold = (peakInt-intThreshold) * pow(E, ((double)((timeDiff/decay)*(-1)))) + intThreshold;
							if(((float)waveform[i]) > calcThreshold)
							{
								pts->push_back(i);
								peakInt = ((float)waveform[i]);
								peakTime = i;
							}
						}
                        
					}
				}
			}
		}
		
		return pts;
	}
    
    std::vector<boost::uint_fast32_t>* SPDInitDecompositionZeroCrossingSimple::findInitPoints(float *waveform, boost::uint_fast16_t waveformLength, float intThreshold) throw(SPDException)
	{
		std::vector<boost::uint_fast32_t> *pts = new std::vector<boost::uint_fast32_t>();
		if(waveformLength > 0)
		{
			boost::uint_fast16_t waveformLengthLess1 = waveformLength - 1;
			float forGrad = 0;
			float backGrad = 0;
			bool firstPeak = true;
			boost::uint_fast32_t peakInt = 0;
			boost::uint_fast32_t peakTime = 0;
			boost::uint_fast32_t timeDiff = 0;
			float calcThreshold = 0;
			for(boost::uint_fast16_t i = 1; i < waveformLengthLess1; ++i)
			{
				forGrad = (waveform[i]) - (waveform[i-1]);
				backGrad = (waveform[i+1]) - (waveform[i]);
				if((forGrad > 0) & (backGrad < 0))
				{
					if(waveform[i] > intThreshold)
					{
						if(firstPeak)
						{
							pts->push_back(i);
							peakInt = waveform[i];
							peakTime = i;
							firstPeak = false;
						}
						else 
						{
							timeDiff = i - peakTime;
							calcThreshold = (peakInt-intThreshold) * pow(E, ((double)((timeDiff/decay)*(-1)))) + intThreshold;
							if(waveform[i] > calcThreshold)
							{
								pts->push_back(i);
								peakInt = waveform[i];
								peakTime = i;
							}
						}
                        
					}
				}
			}
		}
		
		return pts;
	}
	
	SPDInitDecompositionZeroCrossingSimple::~SPDInitDecompositionZeroCrossingSimple()
	{
		
	}
	
	SPDInitDecompositionZeroCrossing::SPDInitDecompositionZeroCrossing(float decay, boost::uint_fast32_t intDecayThres)
	{
		this->decay = decay;
		this->intDecayThres = intDecayThres;
	}
	
	std::vector<boost::uint_fast32_t>* SPDInitDecompositionZeroCrossing::findInitPoints(boost::uint_fast32_t *waveform, boost::uint_fast16_t waveformLength, float intThreshold) throw(SPDException)
	{
		std::vector<boost::uint_fast32_t> *pts = new std::vector<boost::uint_fast32_t>();
		if(waveformLength > 0)
		{
			boost::uint_fast16_t waveformLengthLess1 = waveformLength - 1;
			float *gradients = new float[waveformLengthLess1];
			/*
            std::cout << std::endl;
            for(boost::uint_fast16_t i = 0; i < waveformLength; ++i)
            {
                if(i == 0)
                {
                    std::cout << ((((float)waveform[i])*gain)+offset);
                }
                else 
                {
                    std::cout << "," << ((((float)waveform[i])*gain)+offset);
                }
            }
            std::cout << std::endl;
			*/
			for(boost::uint_fast16_t i = 0; i < waveformLengthLess1; ++i)
			{
				gradients[i] = ((float)waveform[i+1]) - ((float)waveform[i]);
				/*if(i == 0)
                {
                    std::cout << gradients[i];
                }
                else 
                {
                    std::cout << "," << gradients[i];
                }*/
			}
			//std::cout << std::endl << std::endl;
			
			bool firstPeak = true;
			boost::uint_fast32_t peakInt = 0;
			boost::uint_fast32_t peakTime = 0;
			boost::uint_fast32_t timeDiff = 0;
			float calcThreshold = 0;
			
			for(boost::uint_fast16_t i = 0; i < waveformLengthLess1-1; ++i)
			{
				for(boost::uint_fast16_t j = i+1; j < waveformLengthLess1; ++j)
				{
					if((!compare_double(gradients[j], 0)) & this->zeroCrossing(gradients[i], gradients[j]))
					{
						//std::cout << "Crossing: " << i+1 << " - amplitude = " << waveform[i+1] << std::endl;
						if(((float)waveform[i+1]) > intThreshold)
						{
							//std::cout << "Under threshold\n";
							if(firstPeak)
							{
								//std::cout << "push (first) = " << i << " = " << waveform[i] << std::endl;
								pts->push_back(i+1);
								peakInt = ((float)waveform[i+1]);
								peakTime = i+1;
								firstPeak = false;
							}
							else 
							{
								timeDiff = i - peakTime;
								if(peakInt < intDecayThres)
								{
									calcThreshold = 0;
								}
								else 
								{
									calcThreshold = peakInt * pow(E, ((double)((timeDiff/decay)*(-1))));
									//std::cout << "calcThreshold = " << calcThreshold << std::endl;
									//std::cout << "peakInt = " << peakInt << std::endl;
									//std::cout << "timeDiff = " << timeDiff << std::endl;
									//std::cout << "decay = " << decay << std::endl;
									//std::cout << "intThreshold = " << intThreshold << std::endl;
								}
								
								if(calcThreshold < intThreshold)
								{
									//std::cout << "push (calcThreshold < intThreshold) = " << i << " = " << ((((float)waveform[i+1])*gain)+offset) << std::endl;
									pts->push_back(i+1);
									peakInt = ((float)waveform[i+1]);
									peakTime = i+1;
								}
								else if(waveform[i] > calcThreshold)
								{
									//std::cout << "push (waveform[i] > calcThreshold) = " << i << " = " << ((((float)waveform[i+1])*gain)+offset) << std::endl;
									pts->push_back(i+1);
									peakInt = ((float)waveform[i+1]);
									peakTime = i+1;
								}
							}
						}
						break;
					}
					else if(!compare_double(gradients[j], 0))
					{
						break;
					}
                    
				}
			}
            
            delete[] gradients;
		}
		
		return pts;
	}
    
    std::vector<boost::uint_fast32_t>* SPDInitDecompositionZeroCrossing::findInitPoints(float *waveform, boost::uint_fast16_t waveformLength, float intThreshold) throw(SPDException)
	{
		std::vector<boost::uint_fast32_t> *pts = new std::vector<boost::uint_fast32_t>();
		if(waveformLength > 0)
		{
			boost::uint_fast16_t waveformLengthLess1 = waveformLength - 1;
			float *gradients = new float[waveformLengthLess1];
			/*std::cout << std::endl;
             for(boost::uint_fast16_t i = 0; i < waveformLength; ++i)
             {
             if(i == 0)
             {
             std::cout << waveform[i];
             }
             else 
             {
             std::cout << "," << waveform[i];
             }
             }
             std::cout << std::endl;*/
			
			for(boost::uint_fast16_t i = 0; i < waveformLengthLess1; ++i)
			{
				gradients[i] = (waveform[i+1]) - (waveform[i]);
				/*if(i == 0)
                 {
                 std::cout << gradients[i];
                 }
                 else 
                 {
                 std::cout << "," << gradients[i];
                 }*/
			}
			//std::cout << std::endl << std::endl;
			
			bool firstPeak = true;
			boost::uint_fast32_t peakInt = 0;
			boost::uint_fast32_t peakTime = 0;
			boost::uint_fast32_t timeDiff = 0;
			float calcThreshold = 0;
			
			for(boost::uint_fast16_t i = 0; i < waveformLengthLess1-1; ++i)
			{
				for(boost::uint_fast16_t j = i+1; j < waveformLengthLess1; ++j)
				{
					if((!compare_double(gradients[j], 0)) & this->zeroCrossing(gradients[i], gradients[j]))
					{
						//std::cout << "Crossing: " << i+1 << " - amplitude = " << waveform[i+1] << std::endl;
						if(waveform[i+1] > intThreshold)
						{
							//std::cout << "Under threshold\n";
							if(firstPeak)
							{
								//std::cout << "push (first) = " << i << " = " << waveform[i] << std::endl;
								pts->push_back(i+1);
								peakInt = waveform[i+1];
								peakTime = i+1;
								firstPeak = false;
							}
							else 
							{
								timeDiff = i - peakTime;
								if(peakInt < intDecayThres)
								{
									calcThreshold = 0;
								}
								else 
								{
									calcThreshold = peakInt * pow(E, ((double)((timeDiff/decay)*(-1))));
									//std::cout << "calcThreshold = " << calcThreshold << std::endl;
									//std::cout << "peakInt = " << peakInt << std::endl;
									//std::cout << "timeDiff = " << timeDiff << std::endl;
									//std::cout << "decay = " << decay << std::endl;
									//std::cout << "intThreshold = " << intThreshold << std::endl;
								}
								
								if(calcThreshold < intThreshold)
								{
									//std::cout << "push (calcThreshold < intThreshold) = " << i << " = " << waveform[i] << std::endl;
									pts->push_back(i+1);
									peakInt = waveform[i+1];
									peakTime = i+1;
								}
								else if(waveform[i] > calcThreshold)
								{
									//std::cout << "push (waveform[i] > calcThreshold) = " << i << " = " << waveform[i] << std::endl;
									pts->push_back(i+1);
									peakInt = waveform[i+1];
									peakTime = i+1;
								}
							}
						}
						break;
					}
					else if(!compare_double(gradients[j], 0))
					{
						break;
					}
                    
				}
			}
            
            delete[] gradients;
		}
		
		return pts;
	}
	
	bool SPDInitDecompositionZeroCrossing::zeroCrossing(float grad1, float grad2)
	{
		if((grad1 > 0) & (grad2 < 0))
		{
			return true;
		}
		return false;
	}
    
	SPDInitDecompositionZeroCrossing::~SPDInitDecompositionZeroCrossing()
	{
		
	}
    
    
    SPDInitDecompositionZeroCrossingNoRinging::SPDInitDecompositionZeroCrossingNoRinging()
	{

	}
	
	std::vector<boost::uint_fast32_t>* SPDInitDecompositionZeroCrossingNoRinging::findInitPoints(boost::uint_fast32_t *waveform, boost::uint_fast16_t waveformLength, float intThreshold) throw(SPDException)
	{
		std::vector<boost::uint_fast32_t> *pts = new std::vector<boost::uint_fast32_t>();
		if(waveformLength > 0)
		{
			boost::uint_fast16_t waveformLengthLess1 = waveformLength - 1;
			float *gradients = new float[waveformLengthLess1];
			
			for(boost::uint_fast16_t i = 0; i < waveformLengthLess1; ++i)
			{
				gradients[i] = ((float)waveform[i+1]) - ((float)waveform[i]);
			}
			
			for(boost::uint_fast16_t i = 0; i < waveformLengthLess1-1; ++i)
			{
				for(boost::uint_fast16_t j = i+1; j < waveformLengthLess1; ++j)
				{
					if((!compare_double(gradients[j], 0)) & this->zeroCrossing(gradients[i], gradients[j]))
					{
						if(((float)waveform[i+1]) > intThreshold)
						{
							pts->push_back(i+1);
						}
						break;
					}
					else if(!compare_double(gradients[j], 0))
					{
						break;
					}
                    
				}
			}
            delete[] gradients;
		}
		
		return pts;
	}
    
    std::vector<boost::uint_fast32_t>* SPDInitDecompositionZeroCrossingNoRinging::findInitPoints(float *waveform, boost::uint_fast16_t waveformLength, float intThreshold) throw(SPDException)
	{
		std::vector<boost::uint_fast32_t> *pts = new std::vector<boost::uint_fast32_t>();
		if(waveformLength > 0)
		{
			boost::uint_fast16_t waveformLengthLess1 = waveformLength - 1;
			float *gradients = new float[waveformLengthLess1];
			
			for(boost::uint_fast16_t i = 0; i < waveformLengthLess1; ++i)
			{
				gradients[i] = waveform[i+1] - waveform[i];
			}
			
			for(boost::uint_fast16_t i = 0; i < waveformLengthLess1-1; ++i)
			{
				for(boost::uint_fast16_t j = i+1; j < waveformLengthLess1; ++j)
				{
					if((!compare_double(gradients[j], 0)) & this->zeroCrossing(gradients[i], gradients[j]))
					{
						if(waveform[i+1] > intThreshold)
						{
							pts->push_back(i+1);
						}
						break;
					}
					else if(!compare_double(gradients[j], 0))
					{
						break;
					}
                    
				}
			}
            delete[] gradients;
		}
		
		return pts;
	}
	
	bool SPDInitDecompositionZeroCrossingNoRinging::zeroCrossing(float grad1, float grad2)
	{
		if((grad1 > 0) & (grad2 < 0))
		{
			return true;
		}
		return false;
	}
    
	SPDInitDecompositionZeroCrossingNoRinging::~SPDInitDecompositionZeroCrossingNoRinging()
	{
		
	}
    
    
    
    SPDSingularValueDecomposition::SPDSingularValueDecomposition()
	{
		
	}
	
	void SPDSingularValueDecomposition::ComputeSVDgsl(gsl_matrix *inA)
	{
		this->inA = inA;
		/// Calculates SVD for matrix in GSL format using GSL libarary
		
		outV = gsl_matrix_alloc (inA->size2, inA->size2);
		outS = gsl_vector_alloc (inA->size2);
		gsl_vector *out_work = gsl_vector_alloc (inA->size2);
		
		svdCompute = gsl_linalg_SV_decomp(inA, outV, outS, out_work);
	}
	
	void SPDSingularValueDecomposition::SVDLinSolve(gsl_vector *outX, gsl_vector *inB)
	{
		// Solves linear equation using SVD
		/** This uses the gsl_linalg_SV_solve function to calculate the coefficients 
		 for a linear equation. The number of coefficients are determined by the output
		 gsl_vector outX.
		 */ 
		this->inA = inA;
		this->outV = outV;
		this->outS = outS;
		svdSolve = gsl_linalg_SV_solve (inA, outV, outS, inB, outX);
		gsl_matrix_free(outV);
		gsl_vector_free(outS);
	}
    
    SPDSingularValueDecomposition::~SPDSingularValueDecomposition()
	{
		
	}
    
    
    
	SPDPolyFit::SPDPolyFit()
	{
	}
	
	gsl_vector* SPDPolyFit::PolyfitOneDimensionQuiet(int order, gsl_matrix *inData)
	{
		/// Fit one-dimensional n-1th order polynomial
		/**
		 * A gsl_matrix containing the independent and dependent variables is passed in the form: \n
		 * x, y. \n
		 * A gsl_vector is returned containing the coeffients. \n
		 * Polynomial coefficients are obtained using a least squares fit. \n
		 */ 
		
		// Set up matrix of powers
		gsl_matrix *indVarPow;
		gsl_vector *depVar;
		gsl_vector *outCoefficients;
		
		indVarPow = gsl_matrix_alloc(inData->size1, order); // Matrix to hold powers of x
		depVar = gsl_vector_alloc(inData->size1); // Vector to hold y term
		outCoefficients = gsl_vector_alloc(order); // Vector to hold output coefficients and ChiSq
		
		for(unsigned int i = 0; i < inData->size1; i++)
		{
			// Populate devVar vector with y values
			double yelement = gsl_matrix_get(inData, i, 1);
			gsl_vector_set(depVar, i, yelement);
			// Populate indVarPow with x^n
			for(int j = 0; j < order; j++)
			{
				double xelement = gsl_matrix_get(inData, i, 0);
				double xelementtPow = pow(xelement, (j));
				gsl_matrix_set(indVarPow, i, j, xelementtPow);
			}
		}
		
		// Perform Least Squared Fit
		gsl_multifit_linear_workspace *workspace;
		workspace = gsl_multifit_linear_alloc(inData->size1, order);
		gsl_matrix *cov;
		double chisq;
		cov = gsl_matrix_alloc(order, order);
		gsl_multifit_linear(indVarPow, depVar, outCoefficients, cov, &chisq, workspace);
		
		
		/*
         std::cout << "----------------------------------------------------------------------------" << std::endl;
         std::cout << "coefficients are : ";
         vectorUtils.printGSLVector(outCoefficients); 
         std::cout << " chisq = " << chisq << std::endl;
         std::cout << "----------------------------------------------------------------------------" << std::endl;
         std::cout << std::endl;
         */
		
		// Clean up
		gsl_multifit_linear_free(workspace);
		gsl_matrix_free(indVarPow);
		gsl_vector_free(depVar);
		gsl_matrix_free(cov);
		
		return outCoefficients;
	}
	
	gsl_vector* SPDPolyFit::PolyfitOneDimension(int order, gsl_matrix *inData)
	{
		/// Fit one-dimensional n-1th order polynomial
		/**
		 * A gsl_matrix containing the independent and dependent variables is passed in the form: \n
		 * x, y. \n
		 * A gsl_vector is returned containing the coeffients. \n
		 * Polynomial coefficients are obtained using a least squares fit. \n
		 */ 
		
		// Set up matrix of powers
		gsl_matrix *indVarPow;
		gsl_vector *depVar;
		gsl_vector *outCoefficients;
		
		indVarPow = gsl_matrix_alloc(inData->size1, order); // Matrix to hold powers of x
		depVar = gsl_vector_alloc(inData->size1); // Vector to hold y term
		outCoefficients = gsl_vector_alloc(order); // Vector to hold output coefficients and ChiSq
		
		for(unsigned int i = 0; i < inData->size1; i++)
		{
			// Populate devVar vector with y values
			double yelement = gsl_matrix_get(inData, i, 1);
			gsl_vector_set(depVar, i, yelement);
			// Populate indVarPow with x^n
			for(int j = 0; j < order; j++)
			{
				double xelement = gsl_matrix_get(inData, i, 0);
				double xelementtPow = pow(xelement, (j));
				gsl_matrix_set(indVarPow, i, j, xelementtPow);
			}
		}
		
		// Perform Least Squared Fit
		gsl_multifit_linear_workspace *workspace;
		workspace = gsl_multifit_linear_alloc(inData->size1, order);
		gsl_matrix *cov;
		double chisq;
		cov = gsl_matrix_alloc(order, order);
		gsl_multifit_linear(indVarPow, depVar, outCoefficients, cov, &chisq, workspace);
		
		/*
        std::cout << "----------------------------------------------------------------------------" << std::endl;
		std::cout << "coefficients are : ";
		vectorUtils.printGSLVector(outCoefficients); 
		std::cout << " chisq = " << chisq << std::endl;
		std::cout << "----------------------------------------------------------------------------" << std::endl;
		std::cout << std::endl;
		*/
        
		// Clean up
		gsl_multifit_linear_free(workspace);
		gsl_matrix_free(indVarPow);
		gsl_vector_free(depVar);
		gsl_matrix_free(cov);
		
		return outCoefficients;
	}
	
	gsl_vector* SPDPolyFit::PolyfitOneDimensionSVD(int order, gsl_matrix *inData)
	{	
		/// Fit one-dimensional n-1th order polynomial
		/**
		 * A gsl_matrix containing the independent and dependent variables is passed in the form: \n
		 * x, y. \n
		 * A gsl_vector is returned containing the coeffients. \n
		 * A coefficients are obtained using SVD. \n
		 * Use this version when there are equal number of coefficients and variables as it will solve the equations. \n
		 * Otherwise use PolyfitOneDimension to estimate coefficents using Least squares fitting. \n
		 */
		
		// Set up matrix of powers
		gsl_matrix *indVarPow;
		gsl_vector *depVar;
		gsl_vector *outCoefficients;
		
		indVarPow = gsl_matrix_alloc(inData->size1, order); // Matrix to hold powers of x
		depVar = gsl_vector_alloc(inData->size1); // Vector to hold y term
		outCoefficients = gsl_vector_alloc(order); // Vector to hold output coefficients and ChiSq
		
		for(unsigned int i = 0; i < inData->size1; i++)
		{
			// Populate devVad vector with y values
			double yelement = gsl_matrix_get(inData, i, 1);
			gsl_vector_set(depVar, i, yelement);
			
			// Populate indVarPow with x^n
			for(int j = 0; j < order; j++)
			{
				double xelement = gsl_matrix_get(inData, i, 0);
				double xelementtPow = pow(xelement, (j));
				gsl_matrix_set(indVarPow, i, j, xelementtPow);
			}
		}
		
		// Calculate SVD
		SPDSingularValueDecomposition svd;
		svd.ComputeSVDgsl(indVarPow);
		
		// Solve Equation
		svd.SVDLinSolve(outCoefficients, depVar);
		//std::cout << "coefficents are : ";
		//vectorUtils.printGSLVector(outCoefficients);
		
		// Clean up
		gsl_matrix_free(indVarPow);
		gsl_vector_free(depVar);
		
		return outCoefficients;
	}
	
	gsl_matrix* SPDPolyFit::PolyTestOneDimension(int order, gsl_matrix *inData, gsl_vector *coefficients)
	{
		/// Tests one dimensional polynomal equation, outputs measured and predicted values to a matrix.
		
		gsl_matrix *measuredVpredictted;
		measuredVpredictted = gsl_matrix_alloc(inData->size1, 2); // Set up matrix to hold measured and predicted y values.
		
		for(unsigned int i = 0; i < inData->size1; i++) // Loop through inData
		{
			double xVal;
			double yMeasured;
			double yPredicted;
			
			xVal = gsl_matrix_get(inData, i, 0); // Get x value
			yMeasured = gsl_matrix_get(inData, i, 1); // Get measured y value.
			yPredicted = 0;
			
			for(int j = 0; j < order ; j++)
			{
				double xPow = pow(xVal, j); // x^n;
				double coeff = gsl_vector_get(coefficients, j); // a_n
				double coeffXPow = coeff * xPow; // a_n * x^n				
				yPredicted = yPredicted + coeffXPow;
			}
			
			//std::cout << "measured = " << yMeasured << " predicted = " << yPredicted << std::endl;
			
			gsl_matrix_set(measuredVpredictted, i, 0, yMeasured);
			gsl_matrix_set(measuredVpredictted, i, 1, yPredicted);
		}
		
		//matrixUtils.printGSLMatrix(measuredVpredictted);
		
		this->calcRSquaredGSLMatrix(measuredVpredictted);
		
		return measuredVpredictted;
	}
	
	gsl_matrix* SPDPolyFit::PolyfitTwoDimension(int numX, int numY, int orderX, int orderY, gsl_matrix *inData)
	{
		/// Fit n-1th order two dimensional polynomal equation.
		/**
		 * Using least squares, two sets of fits are performed to obtain a two dimensional polynomal equation \n
		 * of the form z(x,y) = a_0(y) + a_1(y)*x + a_2(y)*x^2 + ... + a_n(y)*x^n. \n
		 * where a_n(y) = b_0 + b_1*y + b_2*y^2 + .... + b_n*y^n \n
		 * Data is inputted using mtxt format with data stored: x, y, z. \n
		 * For example: \n
		 * x_1, y_1, z_11 \n
		 * x_2, y_1, z_12 \n
		 * x_3, y_1, z_13 \n
		 * x_1, y_2, z_21 \n
		 * x_2, y_2, z_22 \n
		 * x_3, y_2, z_23 \n
		 * Where the number of x terms (numX) is 3 and the number of y terms (numY) is 2. \n
		 * The b coefficients are outputted as a gsl_matrix with the errors stored in the last column. \n
		 */
		
		gsl_matrix *aCoeff;
		gsl_matrix *bCoeff;
		
		aCoeff = gsl_matrix_alloc(numY, orderX+1);
		bCoeff = gsl_matrix_alloc(orderX, orderY+1);
		
		// PERFORM FIRST SET OF FITS
		//gsl_vector *indVar;
		gsl_matrix *indVarPow;
		gsl_vector *depVar;
		gsl_vector *tempAcoeff;
		gsl_vector *indVar2;
		indVarPow = gsl_matrix_alloc(numX, orderX); // Set up matrix to hold powers of x term for each fit
		indVar2 = gsl_vector_alloc(numY); // Set up vector to hold y values for each fit
		depVar = gsl_vector_alloc(numX); // Set up vector to hold z values for each fit
		tempAcoeff = gsl_vector_alloc(orderX); // Set up vector to hold output coefficients for each fit
		
		double errorA = 0;
		int indexY = 0;
        
		for(int y = 0; y < numY; y++)
		{
			// Populate matrix
			indexY = y * numX;
			//std::cout << "solving set " << y + 1 << "...." << std::endl;
			double yelement = gsl_matrix_get(inData, indexY, 1);
			gsl_vector_set(indVar2, y, yelement); // Add y values to indVar2 vector
			
			// Create matrix of powers for x term
			for(int i = 0; i < numX; i++)
			{
				double melement = gsl_matrix_get(inData, indexY+i, 0);
				double melementDep = gsl_matrix_get(inData, indexY+i, 2);
				gsl_vector_set(depVar, i, melementDep); // Fill dependent variable vector
				
				for(int j = 0; j < orderX; j++)
				{
					double melementPow = pow(melement, (j));
					gsl_matrix_set(indVarPow, i, j, melementPow);
				}
			}
			
			//std::cout << "Starting to solve " << std::endl;
			
			// Solve
			gsl_multifit_linear_workspace *workspace;
			workspace = gsl_multifit_linear_alloc(numX, orderX);
			gsl_matrix *cov;
			double chisq;
			cov = gsl_matrix_alloc(orderX, orderX);
			gsl_multifit_linear(indVarPow, depVar, tempAcoeff, cov, &chisq, workspace); // Perform least squares fit
                                                                                        //std::cout << "solved!" << std::endl;
                                                                                        //vectorUtils.printGSLVector(tempAcoeff);
			
			// Add coefficents to Matrix
			for(int k = 0; k < orderX; k++)
			{
				double coeffElement = gsl_vector_get(tempAcoeff, k);
				gsl_matrix_set(aCoeff, y, k, coeffElement);
			}
			// ChiSq
			gsl_matrix_set(aCoeff, y, orderX, chisq);
			
			errorA = errorA + chisq;
			
		}
		
		errorA = errorA / numY; // Calculate average ChiSq
		std::cout << "-----------------------------" << std::endl;
		std::cout << "First set of fits complete!" << std::endl;
		std::cout << " Average ChiSq = " << errorA << std::endl;
		std::cout << std::endl;
		//matrixUtils.printGSLMatrix(aCoeff);
		//matrixUtils.saveGSLMatrix2GridTxt(aCoeff, "/users/danclewley/Documents/Temp/L_HH_aCoeff");
		
		//Clean up
		gsl_vector_free(tempAcoeff);
		gsl_vector_free(depVar);
		gsl_matrix_free(indVarPow);
		
		// PERFORM SECOND SET OF FITS
		gsl_matrix *indVar2Pow; // Set up matrix to hold powers of y term for each fit
		gsl_vector *depVar2; // Set up vector to hold a coefficeints for each fit
		gsl_vector *tempBcoeff; // Set up matrix to hold B coefficients for each fit
		indVar2Pow = gsl_matrix_alloc(numY, orderY);
		depVar2 = gsl_vector_alloc(numY);
		tempBcoeff = gsl_vector_alloc(orderY);
		double errorB  = 0;
		
		// Create matrix of powers for y term.
		for(int i = 0; i < numY; i++)
		{
			double melement = gsl_vector_get(indVar2, i);
			
			for(int j = 0; j < orderY; j++)
			{
				double melementPow = pow(melement, (j));
				gsl_matrix_set(indVar2Pow, i, j, melementPow);
			}
		}
		
		// Loop through fits
		for(int i = 0; i < orderX; i++)
		{
			for(int j = 0; j < numY; j++)
			{
				double melement = gsl_matrix_get(aCoeff, j, i);
				gsl_vector_set(depVar2, j, melement);
			}
			// Solve
			gsl_multifit_linear_workspace *workspace;
			workspace = gsl_multifit_linear_alloc(numY, orderY);	
			gsl_matrix *cov;
			double chisq;
			cov = gsl_matrix_alloc(orderY, orderY);
			gsl_multifit_linear(indVar2Pow, depVar2, tempBcoeff, cov, &chisq, workspace);
			gsl_multifit_linear_free(workspace);
			gsl_matrix_free(cov);
			//matrixUtils.printGSLMatrix(indVar2Pow);
			//vectorUtils.printGSLVector(depVar2);
			
			// Add coefficents to Matrix
			for(int k = 0; k < orderY; k++)
			{
				double coeffElement = gsl_vector_get(tempBcoeff, k);
				gsl_matrix_set(bCoeff, i, k, coeffElement);
			}
			
			// ChiSq
			//std::cout << "chisq = "<< chisq << std::endl;
			errorB = errorB + chisq;
			gsl_matrix_set(bCoeff, i, orderY, chisq);
		}
		
		errorB = errorB / orderX; // Calculate average ChiSq
		std::cout << "Second set of fits complete!" << std::endl;
		std::cout << " Average ChiSq = " << errorB << std::endl;
		std::cout << "-----------------------------" << std::endl;
		std::cout << std::endl;
		//std::cout << "Coefficients are : " << std::endl;
		//matrixUtils.printGSLMatrix(bCoeff);
		
		// Clean up
		gsl_matrix_free(indVar2Pow);
		gsl_vector_free(depVar2);
		gsl_vector_free(tempBcoeff);
		gsl_matrix_free(aCoeff);
		
		return bCoeff;
	}
	
	gsl_matrix* SPDPolyFit::PolyTestTwoDimension(int orderX, int orderY, gsl_matrix *inData, gsl_matrix *coefficients)
	{
		/// Tests one dimensional polynomal equation, outputs measured and predicted values to a matrix.
		
		gsl_matrix *measuredVpredictted;
		measuredVpredictted = gsl_matrix_alloc(inData->size1, 2); // Set up matrix to hold measured and predicted y values.
		
		for(unsigned int i = 0; i < inData->size1; i++) // Loop through inData
		{
			double xVal;
			double yVal;
			double zMeasured;
			double zPredicted;
			
			xVal = gsl_matrix_get(inData, i, 0); // Get x value
			yVal = gsl_matrix_get(inData, i, 1); // Get y value
			zMeasured = gsl_matrix_get(inData, i, 2); // Get measured z value.
			zPredicted = 0;
			for(int x = 0; x < orderX; x ++) 
			{
				double xPow = pow(xVal, x); // x^n;
				
				double aCoeff = 0.0; 
				
				for(int y = 0; y < orderY ; y++) // Calculate a_n(y)
				{
					double yPow = pow(yVal, y); // y^n;
					double bcoeff = gsl_matrix_get(coefficients, x, y); // b_n
					double bcoeffYPow = bcoeff * yPow; // b_n * y^n				
					aCoeff = aCoeff + bcoeffYPow;
				}
				double acoeffXPow = xPow * aCoeff;
				zPredicted = zPredicted + acoeffXPow;
			}
			
			//std::cout << "measured = " << yMeasured << " predicted = " << yPredicted << std::endl;
			
			gsl_matrix_set(measuredVpredictted, i, 0, zMeasured);
			gsl_matrix_set(measuredVpredictted, i, 1, zPredicted);
		}
		
		this->calcRSquaredGSLMatrix(measuredVpredictted);
		
		return measuredVpredictted;
	}
	
	gsl_matrix* SPDPolyFit::PolyfitThreeDimension(int numX, int numY, int numZ, int orderX, int orderY, int orderZ, gsl_matrix *inData)
	{
		/// Fit n-1th order three dimensional polynomal equation.
		/**
		 * Using least squares, three sets of fits are performed to obtain a three dimensional polynomal equation \n
		 * of the form z(x,y) = a_0(y) + a_1(y)*x + a_2(y)*x^2 + ... + a_n(y)*x^n. \n
		 * where a_n(y) = b_0 + b_1*y + b_2*y^2 + .... + b_n*y^n \n
		 * Data is inputted using mtxt format with data stored: x, y, z. \n
		 * For example: \n
		 * x_1, y_1, z_1 \n
		 * x_2, y_1, z_1 \n
		 * x_3, y_1, z_1 \n
		 * x_1, y_2, z_1 \n
		 * x_2, y_2, z_1 \n
		 * x_3, y_2, z_1 \n
		 * x_1, y_1, z_2 \n
		 * x_2, y_1, z_2 \n
		 * x_3, y_1, z_2 \n
		 * x_1, y_2, z_2 \n
		 * x_2, y_2, z_2 \n
		 * x_3, y_2, z_2 \n
		 * Where the number of x terms (numX) is 3, the number of y terms (numY) is 2 and the number of z terms (numZ) is 2\n
		 * The b coefficients are outputted as a gsl_matrix with the errors stored in the last column. \n
		 */
		
		std::cout << "Order X, Y, X = " << orderX << ", " << orderY << ", " << orderZ << std::endl;
		
		gsl_matrix *aCoeff, *bCoeff, *cCoeff, *indVarXPow;
		
		gsl_vector *depVarX, *depVarY, *depVarZ;
		gsl_vector *tempAcoeff, *tempBcoeff, *tempCcoeff;
		gsl_vector *indVarY, *indVarZ;
		gsl_matrix *indVarYPow, *indVarZPow; 
		
		aCoeff = gsl_matrix_alloc(numY, orderX+1); // Set up matrix to hold a coefficients (and chi squared)
		bCoeff = gsl_matrix_alloc(orderX * numZ, orderY+1); // Set up matrix to hold b coefficients (and chi squared)
		cCoeff = gsl_matrix_alloc(orderX * orderY, orderZ+1); // Set up matrix to hold b coefficients (and chi squared)
		
		indVarXPow = gsl_matrix_alloc(numX, orderX); // Set up matrix to hold powers of x term for each fit
		indVarY = gsl_vector_alloc(numY); // Set up vector to hold y values for each fit
		indVarZ = gsl_vector_alloc(numZ); // Set up vector to hold z values for each fit
		depVarX = gsl_vector_alloc(numX); // Set up vector to hold independent values for each fit
		tempAcoeff = gsl_vector_alloc(orderX); // Set up vector to hold output coefficients for each fit
		
		indVarYPow = gsl_matrix_alloc(numY, orderY);  // Set up matrix to hold powers of y term for each fit
		depVarY = gsl_vector_alloc(numY); // Set up vector to hold dependent varieble (a coefficents) for second set of fits
		tempBcoeff = gsl_vector_alloc(orderY); // Set up matrix to hold B coefficients for each fit
		
		indVarZPow = gsl_matrix_alloc(numZ, orderZ); // Set up matrix to hold powers of z
		depVarZ = gsl_vector_alloc(numZ); // Set up vector to hold dependent varieble (b coefficents) for third set of fits
		tempCcoeff = gsl_vector_alloc(orderZ); // Set up vectro to hold the c coefficents from each fit
		
		/*********************************
		 *  PERFORM FIRST SET OF FITS    *
		 *********************************/
		
		int indexZ = 0;
		double errorA = 0;
		double errorB  = 0;
		
		for(int z = 0; z < numZ; z++)
		{
			indexZ = z * (numX * numY); // Index moving through z variable
			
			
			int indexY = 0;
			
			double zelement = gsl_matrix_get(inData, indexZ, 2);
			gsl_vector_set(indVarZ, z, zelement); // Add z values to indVarZ vector
			
			for(int y = 0; y < numY; y++)
			{
				// Populate matrix
				indexY = (y * numX) + (indexZ);  // Index is the starting point for each y term
				double yelement = gsl_matrix_get(inData, indexY, 1);
				gsl_vector_set(indVarY, y, yelement); // Add y values to indVarY vector
				
				// Create matrix of powers for x term
				for(int i = 0; i < numX; i++)
				{
					double melement = gsl_matrix_get(inData, indexY+i, 0);
					double melementDep = gsl_matrix_get(inData, indexY+i, 3);
					gsl_vector_set(depVarX, i, melementDep); // Fill dependent variable vector
					
					for(int j = 0; j < orderX; j++)
					{
						double melementPow = pow(melement, j);
						gsl_matrix_set(indVarXPow, i, j, melementPow);
					}
				}
				
				/*std::cout << "indVarX = " << std::endl;
				 matrixUtils.printGSLMatrix(indVarXPow);
				 std::cout << "depVarX = " << std::endl;
				 vectorUtils.printGSLVector(depVarX);
				 std::cout << std::endl;*/
				
				// Solve
				gsl_multifit_linear_workspace *workspace;
				workspace = gsl_multifit_linear_alloc(numX, orderX);
				gsl_matrix *cov;
				double chisq;
				cov = gsl_matrix_alloc(orderX, orderX);
				gsl_multifit_linear(indVarXPow, depVarX, tempAcoeff, cov, &chisq, workspace); // Perform least squares fit
                                                                                              //std::cout << "a Coeff = " << std::endl;
                                                                                              //vectorUtils.printGSLVector(tempAcoeff);
				
				// Add coefficents to Matrix
				for(int k = 0; k < orderX; k++)
				{
					double coeffElement = gsl_vector_get(tempAcoeff, k);
					gsl_matrix_set(aCoeff, y, k, coeffElement);
				}
				// ChiSq
				gsl_matrix_set(aCoeff, y, orderX, chisq);
				//std::cout << "chiSq for first set of fits  = " << chisq << std::endl;
				errorA = errorA + chisq;
			}
			
			/*********************************
			 *  PERFORM SECOND SET OF FITS    *
			 **********************************/
			
			// Create matrix of powers for y term.
			for(int i = 0; i < numY; i++)
			{
				double melement = gsl_vector_get(indVarY, i);
				
				for(int j = 0; j < orderY; j++)
				{
					double melementPow = pow(melement, (j));
					gsl_matrix_set(indVarYPow, i, j, melementPow);
				}
			}
			
			// Loop through fits
			for(int i = 0; i < orderX; i++)
			{
				for(int j = 0; j < numY; j++)
				{
					double melement = gsl_matrix_get(aCoeff, j, i);
					gsl_vector_set(depVarY, j, melement);
				}
				
				/*
				 std::cout << "indVarY = " << std::endl;
				 vectorUtils.printGSLVector(indVarY);
				 std::cout << "depVarY = " << std::endl;
				 vectorUtils.printGSLVector(depVarY);
				 std::cout << std::endl;
				 */
				
				// Solve
				gsl_multifit_linear_workspace *workspace;
				workspace = gsl_multifit_linear_alloc(numY, orderY);
				gsl_matrix *cov;
				double chisq;
				cov = gsl_matrix_alloc(orderY, orderY);
				gsl_multifit_linear(indVarYPow, depVarY, tempBcoeff, cov, &chisq, workspace);
				gsl_multifit_linear_free(workspace);
				gsl_matrix_free(cov);
				
				// Add coefficents to Matrix
				for(int k = 0; k < orderY; k++)
				{
					double coeffElement = gsl_vector_get(tempBcoeff, k);
					gsl_matrix_set(bCoeff, i + (z * orderY), k, coeffElement);
				}
				// ChiSq
				
				errorB = errorB + chisq;
				gsl_matrix_set(bCoeff, i + (z * orderY), orderY, chisq);
			}
			
		}
		
		errorA = errorA / (numY * numZ); // Calculate average ChiSq
		errorB = errorB / (orderY * numZ); // Calculate average ChiSq	
		
		std::cout << "----------------------------------------------" << std::endl;
		std::cout << "First and second set of fits complete " << std::endl;
		std::cout << " Average ChiSq for first set of fits = " << errorA << std::endl;
		std::cout << " Average ChiSq for second set of fits = " << errorB << std::endl;
		
		/*
		 std::cout << "==============================================================" << std::endl;
		 matrixUtils.printGSLMatrix(bCoeff);
		 std::cout << "==============================================================" << std::endl;
		 */
		
		/*std::cout << "indVarZ = " << std::endl;
		 vectorUtils.printGSLVector(indVarZ);
		 std::cout << "coeffB = " << std::endl;
		 matrixUtils.printGSLMatrix(bCoeff);
		 std::cout << std::endl;*/
		
		/***************************************************
		 * PERFORM THIRD SET OF FITS                       
		 * 
		 * Coefficients are in the form:
		 * z1: 
		 * b0_0 b0_1 b0_2 ... b0_n
		 * b1_0 b1_1 b1_2 ... b0_n
		 *  .    .     .        .
		 * bn_0 bn_1 bn_2 ... bn_n
		 * z2 
		 * b0_0 b0_1 b0_2 ... b0_n
		 * b1_0 b1_1 b1_2 ... b0_n
		 *  .    .     .        .
		 * bn_0 bn_1 bn_2 ... bn_n 
		 *
		 *
		 ***************************************************/
		
		// Create matrix of powers for z term.
		for(int i = 0; i < numZ; i++)
		{
			double melement = gsl_vector_get(indVarZ, i);
			for(int j = 0; j < orderZ; j++)
			{
				double melementPow = pow(melement, j);
				gsl_matrix_set(indVarZPow, i, j, melementPow);
			}
		}
		
		// Loop through fits
		
		double errorC = 0;
		int c = 0;
		
		for (int a = 0; a < orderX; a++) // a \/ Go through b terms that make up a coefficients
		{
			for(int i = 0; i < orderY; i++) // b - > Loop through powers of b coefficients for a_i
			{
				for(int j = 0; j < numZ; j++) // b_n \/ Get b_ij coefficients for each value of z
				{
					double melement = gsl_matrix_get(bCoeff, a + (j * orderY), i);
					gsl_vector_set(depVarZ, j, melement);
				}
				
				/*
				 std::cout << "IndVar " << std::endl;
				 vectorUtils.printGSLVector(indVarZ);
				 std::cout << "DepVar " << std::endl;
				 vectorUtils.printGSLVector(depVarZ);
				 */
				
				// Solve
				gsl_multifit_linear_workspace *workspace;
				workspace = gsl_multifit_linear_alloc(numZ, orderZ);
				gsl_matrix *cov;
				double chisq;//
				cov = gsl_matrix_alloc(orderZ, orderZ);
				gsl_multifit_linear(indVarZPow, depVarZ, tempCcoeff, cov, &chisq, workspace);
				gsl_multifit_linear_free(workspace);
				gsl_matrix_free(cov);
				
				// Add coefficents to Matrix
				for(int k = 0; k < orderZ; k++)
				{
					double coeffElement = gsl_vector_get(tempCcoeff, k);
					gsl_matrix_set(cCoeff, c, k, coeffElement);
				}
				
				// ChiSq
				//std::cout << "chisq = "<< chisq << std::endl;
				errorC = errorC + chisq;
				gsl_matrix_set(cCoeff, c, orderZ, chisq);
				c++;
			}
		}
		
		errorC = errorC / c; // Calculate average ChiSq
		std::cout << "Third set of fits complete " << std::endl;
		std::cout << " Average ChiSq for first set of fits = " << errorC << std::endl;
		std::cout << "----------------------------------------------" << std::endl;
		std::cout << std::endl;
		
		// Clean up
		
		gsl_matrix_free(aCoeff);
		gsl_matrix_free(bCoeff);
		
		gsl_matrix_free(indVarXPow);
		gsl_vector_free(depVarX);
		gsl_vector_free(tempAcoeff);
		gsl_vector_free(indVarY);
		gsl_vector_free(indVarZ);
		
		gsl_matrix_free(indVarYPow);
		gsl_vector_free(depVarY); 
		gsl_vector_free(tempBcoeff); 
		
		gsl_matrix_free(indVarZPow); 
		gsl_vector_free(depVarZ); 
		gsl_vector_free(tempCcoeff); 
		
		//matrixUtils.printGSLMatrix(cCoeff);
		
		return cCoeff;
	}
	
	gsl_matrix* SPDPolyFit::PolyTestThreeDimension(int orderX, int orderY, int orderZ, gsl_matrix *inData, gsl_matrix *coefficients)
	{
		/// Tests a three dimensional polynomal equation, outputs measured and predicted values to a matrix.
		
		gsl_matrix *measuredVpredictted;
		measuredVpredictted = gsl_matrix_alloc(inData->size1, 2); // Set up matrix to hold measured and predicted y values.
        
		for(unsigned int i = 0; i < inData->size1; i++) // Loop through inData
		{
			double xVal, yVal, zVal;
			double xPow, yPow, zPow;
			double fMeasured, fPredicted;
			double bcoeffPowY, cCoeffPowZ;
			long double cCoeff;
			
			xVal = gsl_matrix_get(inData, i, 0); // Get x value
			yVal = gsl_matrix_get(inData, i, 1); // Get y value
			zVal = gsl_matrix_get(inData, i, 2); // Get z value
			
			fMeasured = gsl_matrix_get(inData, i, 3); // Get measured f value.
			fPredicted = 0.0;
			for(int x = 0; x < orderX; x ++) 
			{
				bcoeffPowY = 0.0; 
				for(int y = 0; y < orderY; y++)
				{
					cCoeffPowZ = 0.0;
					//std::cout << "cCoeff = ";
					for(int z = 0; z < orderZ; z++)
					{     
						zPow = pow(zVal, z);
						//cCoeff = gsl_matrix_get(coefficients, c, z);
						cCoeff = gsl_matrix_get(coefficients, y + (x * orderX), z);
						//std::cout << cCoeff;
						cCoeffPowZ = cCoeffPowZ + (cCoeff * zPow);
					}
					//std::cout << std::endl;
					yPow = pow(yVal, y); // y^n;
					bcoeffPowY = bcoeffPowY + (cCoeffPowZ * yPow); // c_n * y^n
				}
				xPow = pow(xVal, x); // dielectric^n;
				fPredicted = fPredicted + (bcoeffPowY * xPow);
			}
			
			//std::cout << "measured = " << yMeasured << " predicted = " << yPredicted << std::endl;
			
			gsl_matrix_set(measuredVpredictted, i, 0, fMeasured);
			gsl_matrix_set(measuredVpredictted, i, 1, fPredicted);
		}
		
		this->calcRSquaredGSLMatrix(measuredVpredictted);
		
		//matrixUtils.printGSLMatrix(measuredVpredictted);
		return measuredVpredictted;
	}
	
	void SPDPolyFit::calcRSquaredGSLMatrix(gsl_matrix *dataXY)
	{
		double sumX = 0;
		double sumY = 0;
		
		// Calc mean
		for(unsigned int i = 0; i < dataXY->size1; i++)
		{
			sumX = sumX + gsl_matrix_get(dataXY,i , 0);
			sumY = sumY + gsl_matrix_get(dataXY,i , 1);
		}
		
		double xMean = sumX / dataXY->size1;
		double yMean = sumY / dataXY->size1;
		
		double xMeanSq = xMean * xMean;
		double yMeanSq = yMean * yMean;
		double xyMean = xMean * yMean;
		
		double ssXX = 0;
		double ssYY = 0;
		double ssXY = 0;
		
		for(unsigned int i = 0; i < dataXY->size1; i++)
		{
			
			double dataX = gsl_matrix_get(dataXY,i , 0);
			double dataY = gsl_matrix_get(dataXY,i , 1);
			
			ssXX = ssXX + ((dataX * dataX) - xMeanSq);
			ssYY = ssYY + ((dataY * dataY) - yMeanSq);
			ssXY = ssXY + ((dataX * dataY) - xyMean);
		}
		
		double rSq = (ssXY * ssXY ) / (ssXX * ssYY);
		
		std::cout << "**************************" << std::endl;
		std::cout << "  R squared = " << rSq << std::endl;
		std::cout << "**************************" << std::endl;
	}
	
	double SPDPolyFit::calcRSquaredGSLMatrixQuiet(gsl_matrix *dataXY)
	{
		double sumX = 0;
		double sumY = 0;
		
		// Calc mean
		for(unsigned int i = 0; i < dataXY->size1; i++)
		{
			sumX = sumX + gsl_matrix_get(dataXY,i , 0);
			sumY = sumY + gsl_matrix_get(dataXY,i , 1);
		}
		
		double xMean = sumX / dataXY->size1;
		double yMean = sumY / dataXY->size1;
		
		double xMeanSq = xMean * xMean;
		double yMeanSq = yMean * yMean;
		double xyMean = xMean * yMean;
		
		double ssXX = 0;
		double ssYY = 0;
		double ssXY = 0;
		
		for(unsigned int i = 0; i < dataXY->size1; i++)
		{
			
			double dataX = gsl_matrix_get(dataXY,i , 0);
			double dataY = gsl_matrix_get(dataXY,i , 1);
			
			ssXX = ssXX + ((dataX * dataX) - xMeanSq);
			ssYY = ssYY + ((dataY * dataY) - yMeanSq);
			ssXY = ssXY + ((dataX * dataY) - xyMean);
		}
		
		double rSq = (ssXY * ssXY ) / (ssXX * ssYY);
		
		return rSq;
	}
	
	void SPDPolyFit::calcRMSErrorGSLMatrix(gsl_matrix *dataXY)
	{
		double sqSum = 0;
		
		// Calc mean
		for(unsigned int i = 0; i < dataXY->size1; i++)
		{
			sqSum = sqSum + pow((gsl_matrix_get(dataXY,i , 0) - gsl_matrix_get(dataXY,i , 1)),2);
		}
		
		double sqMean = sqSum / double(dataXY->size1);
		
		double rmse = sqrt(sqMean);
		
		std::cout << "**************************" << std::endl;
		std::cout << "  RMSE = " << rmse << std::endl;
		std::cout << "**************************" << std::endl;
	}
	
	void SPDPolyFit::calcMeanErrorGSLMatrix(gsl_matrix *dataXY)
	{
		double sum = 0;
		
		// Calc mean
		for(unsigned int i = 0; i < dataXY->size1; i++)
		{
			sum = sum + (gsl_matrix_get(dataXY,i , 0) - gsl_matrix_get(dataXY,i , 1));
		}
		
		double meanError = sum / double(dataXY->size1);
		
		std::cout << "**************************" << std::endl;
		std::cout << "  Mean Error = " << meanError << std::endl;
		std::cout << "**************************" << std::endl;
	}
	
	double SPDPolyFit::calcRMSErrorGSLMatrixQuiet(gsl_matrix *dataXY)
	{
		double sqSum = 0;
		
		// Calc mean
		for(unsigned int i = 0; i < dataXY->size1; i++)
		{
			sqSum = sqSum + pow((gsl_matrix_get(dataXY,i , 0) - gsl_matrix_get(dataXY,i , 1)),2);
		}
		
		double sqMean = sqSum / double(dataXY->size1);
		
		double rmse = sqrt(sqMean);
		return rmse;
	}
	
	double SPDPolyFit::calcMeanErrorGSLMatrixQuiet(gsl_matrix *dataXY)
	{
		double sum = 0;
		
		// Calc mean
		for(unsigned int i = 0; i < dataXY->size1; i++)
		{
			sum = sum + (gsl_matrix_get(dataXY,i , 0) - gsl_matrix_get(dataXY,i , 1));
		}
		
		double meanError = sum / double(dataXY->size1);
		return meanError;
	}
	
	SPDPolyFit::~SPDPolyFit()
	{
	}	
	
}



