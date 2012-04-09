/*
 *  SPDCommon.h
 *  spdlib
 *
 *  Created by Pete Bunting on 27/11/2010.
 *  Copyright 2010 SPDLib. All rights reserved.
 *
 *  This file is part of SPDLib.
 *
 *  Permission is hereby granted, free of charge, to any person 
 *  obtaining a copy of this software and associated documentation 
 *  files (the "Software"), to deal in the Software without restriction, 
 *  including without limitation the rights to use, copy, modify, 
 *  merge, publish, distribute, sublicense, and/or sell copies of the 
 *  Software, and to permit persons to whom the Software is furnished 
 *  to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be 
 *  included in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 *  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
 *  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR 
 *  ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
 *  CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
 *  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef SPDCommon_H
#define SPDCommon_H

#define _USE_MATH_DEFINES
#include <math.h>

#include <boost/cstdint.hpp>

namespace spdlib
{
    #ifndef NAN
        #define NAN nan("")
    #endif
    
    /**
     * Compression deflate parameter for SPD Writers.
     */
    static const boost::int_fast16_t SPD_DEFLATE( 1 );    
    
	static const boost::int_fast16_t SPD_FALSE( 0 );
	static const boost::int_fast16_t SPD_TRUE( 1 );
	
	/**
	 * Anything undefined value will be set to zero.
	 */
	static const boost::uint_fast16_t SPD_UNDEFINED( 0 );

	/**
	 * Define the known classes for classification.
	 */
	static const boost::uint_fast16_t SPD_UNCLASSIFIED( 1 );
	static const boost::uint_fast16_t SPD_CREATED( 2 );
	static const boost::uint_fast16_t SPD_GROUND( 3 );
	static const boost::uint_fast16_t SPD_LOW_VEGETATION( 4 );
	static const boost::uint_fast16_t SPD_MEDIUM_VEGETATION( 5 );
	static const boost::uint_fast16_t SPD_HIGH_VEGETATION( 6 );
	static const boost::uint_fast16_t SPD_BUILDING( 7 );
	static const boost::uint_fast16_t SPD_WATER( 8 );
	static const boost::uint_fast16_t SPD_TRUNK( 9 );
	static const boost::uint_fast16_t SPD_FOILAGE( 10 );
	static const boost::uint_fast16_t SPD_BRANCH( 11 );
	static const boost::uint_fast16_t SPD_WALL( 12 );
    
    static const boost::uint_fast16_t SPD_ALL_CLASSES( 100 );
    static const boost::uint_fast16_t SPD_ALL_CLASSES_TOP( 101 );
    static const boost::uint_fast16_t SPD_VEGETATION_TOP( 102 );
    static const boost::uint_fast16_t SPD_VEGETATION( 103 );
    static const boost::uint_fast16_t SPD_NOT_GROUND( 104 );

    /**
	 * Define the returnID values.
	 */
    static const boost::uint_fast16_t SPD_ALL_RETURNS( 100 );
    static const boost::uint_fast16_t SPD_FIRST_RETURNS( 101 );
    static const boost::uint_fast16_t SPD_LAST_RETURNS( 102 );
    static const boost::uint_fast16_t SPD_NOTFIRST_RETURNS( 103 );
    static const boost::uint_fast16_t SPD_FIRSTLAST_RETURNS( 104 );
    
	/**
	 * Define the flight line edge parameters.
	 */
	static const boost::uint_fast16_t SPD_WITH_SCAN( 1 );
	static const boost::uint_fast16_t SPD_SCAN_END( 2 );

	/**
	 * Define the scan direction.
	 */
	static const boost::uint_fast16_t SPD_NEGATIVE( 1 );
	static const boost::uint_fast16_t SPD_POSITIVE( 2 );
	
	
	/**
	 * Define the indexing of the file coordinate type.
	 */
	static const boost::uint_fast16_t SPD_NO_IDX( 0 );
	static const boost::uint_fast16_t SPD_CARTESIAN_IDX( 1 );
	static const boost::uint_fast16_t SPD_SPHERICAL_IDX( 2 );
	static const boost::uint_fast16_t SPD_CYLINDRICAL_IDX( 3 );
    static const boost::uint_fast16_t SPD_POLAR_IDX( 4 );
    static const boost::uint_fast16_t SPD_SCAN_IDX( 5 );
	
	/**
	 * Define the PointType
	 */
	static const boost::uint_fast16_t SPD_DISCRETE_PT( 1 );
	static const boost::uint_fast16_t SPD_DECOMPOSED_PT( 2 );
	static const boost::uint_fast16_t SPD_WAVEFORM_PT( 3 );
    
    /**
	 * Define the File type (SPD_SEQ SPD_NONSEQ UPD)
	 */
	static const boost::uint_fast16_t SPD_SEQ_TYPE( 1 );
	static const boost::uint_fast16_t SPD_NONSEQ_TYPE( 2 );
	static const boost::uint_fast16_t SPD_UPD_TYPE( 3 );
    
    /**
     * Define values to differiciate between Height and Z
     */
    static const boost::uint_fast16_t SPD_USE_Z( 1 );
	static const boost::uint_fast16_t SPD_USE_HEIGHT( 2 );
    
    /**
     * Define values to differiciate between selecting the lowest or highest point
     */
    static const boost::uint_fast16_t SPD_SELECT_LOWEST( 1 );
	static const boost::uint_fast16_t SPD_SELECT_HIGHEST( 2 );
    
    /**
     * Define the available options for the number of bytes used to store the waveform
     */
    static const boost::uint_fast16_t SPD_8_BIT_WAVE( 8 );
    static const boost::uint_fast16_t SPD_16_BIT_WAVE( 16 );
    static const boost::uint_fast16_t SPD_32_BIT_WAVE( 32 );
    
    enum SPDDataType
    {
        spd_int,
        spd_uint,
        spd_float,
        spd_double
    };
	
	/**
	 Speed of light used for calculated (Metres per nano-second)     
     Nominal refractive index calculated from http://emtoolbox.nist.gov/Wavelength/Ciddor.asp
        Vacuum Wavelength: 1550 Nanometers [nm] 
        Air Temperature: 20 Degrees Celsius 
        Atmospheric Pressure: 101.325 Kilopascals [kPa] 
        Air Humidity: 66 Relative Humidity, Percent 
        Carbon Dioxide Content: 450 Micromole per Mole [parts per million, ppm] 
        Wavelength in Ambient Air: 1549.584698 Nanometers [nm] 
        Refractive Index of Air: 1.000268008 
        Uncertainty of Calculated Index: 0.000000024
	 */
	static const double SPD_REFRACTIVE_INDEX_AIR(1.000268008);
    static const double SPD_SPEED_OF_LIGHT_NS(0.299792458 / SPD_REFRACTIVE_INDEX_AIR);
	
	/**
	 * Value of 'e'
	 */
	static const double E(M_E);
	
	/**
	 * Methods for deciding what values are used for indexing 
	 * the points using cartesian coordinate system.
	 */
	static const boost::uint_fast16_t SPD_FIRST_RETURN( 1 );
	static const boost::uint_fast16_t SPD_LAST_RETURN( 2 );
	static const boost::uint_fast16_t SPD_START_OF_RECEIVED_WAVEFORM( 3 );
	static const boost::uint_fast16_t SPD_END_OF_RECEIVED_WAVEFORM( 4 );
	static const boost::uint_fast16_t SPD_ORIGIN(5);
	static const boost::uint_fast16_t SPD_MAX_INTENSITY(6);
	static const boost::uint_fast16_t SPD_IDX_UNCHANGED(7);
    
	
	inline void SPDConvertToSpherical(double origX, double origY, double origZ, double ptX, double ptY, double ptZ, double *zenith, double *azimuth, double *range)
	{
		double tempX = ptX - origX;
		double tempY = ptY - origY;
		double tempZ = ptZ - origZ;
		
        /*
        Old version:
		*range = sqrt((tempX * tempX) + (tempY * tempY) + (tempZ * tempZ));
		*zenith = acos(tempZ/(*range));
		*azimuth = atan2(tempY, tempX);
        */
        
        /**
        r >= 0
        0 <= theta < pi
        0 <= phi < 2pi
        0 phi is true north
        0 theta is the vertical "up" view
        ALS scan angle = pi - zenith
        */
		*range = sqrt((tempX * tempX) + (tempY * tempY) + (tempZ * tempZ));
		*zenith = acos(tempZ/(*range));
		double tempAzimuth = atan2(tempX, tempY);
        if (tempAzimuth < 0)
        {
            *azimuth = (2.0 * M_PI) + tempAzimuth;
        }
        else
        {
            *azimuth = tempAzimuth; 
        }
	}
	
	inline void SPDConvertToCartesian(double zenith, double azimuth, double range, double origX, double origY, double origZ, double *ptX, double *ptY, double *ptZ)
	{
		/*
        Old version:
        *ptX = origX + (range * sin(zenith) * cos(azimuth));
		*ptY = origY + (range * sin(zenith) * sin(azimuth));
		*ptZ = origZ + (range * cos(zenith));
        */
        
        /**
        We're using a positive, or "right-handed" cartesian coordinate system
        y is positive looking true north, x is positive looking east, z is "up"
        TLS will typically have positive z, and ALS negative z (from 0,0,0 origin)
        */        
        *ptX = origX + (range * sin(zenith) * sin(azimuth));
		*ptY = origY + (range * sin(zenith) * cos(azimuth));
		*ptZ = origZ + (range * cos(zenith));        
                
	}
	
    struct WeibullFitVals
    {
        double *heights;
        double *binVals;
        double *error;
    };
    
    /*
	 * int m     - number of data points
	 * int n     - number of parameters
	 * double *p - array of n parameters
	 * double *deviates - array of m deviates to be returned by myfunct()
	 * double **derivs - used for user-computed derivatives (see below)
	 * (= 0  when automatic finite differences are computed)
	 */
    inline int weibullFit(int m, int n, double *p, double *deviates, double **derivs, void *data)
    {
        WeibullFitVals *fitData = (WeibullFitVals*) data;
        
        if(n != 2) // input parameters could be just beta and alpha
        {
            return -1;
        }
        
        /*
         * p[0] = alpha
         * p[1] = beta
         */
        
        double part1 = p[0]/pow(p[1], p[0]);
        double part2 = 0;
        double part3 = 0;
        
		float predVal = 0;
		for(int i = 0; i < m; ++i)
		{
			predVal = 0;
			
            part2 = pow(fitData->heights[i], p[0]-1);
            
            part3 = exp((-1)*(pow((fitData->heights[i]/p[1]),p[0])));
            
            predVal = part1 * part2 * part3;
            
			deviates[i] = (fitData->binVals[i] - predVal) / fitData->error[i];
		}
		
		return 0;
    }
    
	struct PulseWaveform 
	{
		double *time;
		double *intensity;
		double *error;
	};

	/*
	 * int m     - number of data points
	 * int n     - number of parameters
	 * double *p - array of n parameters
	 * double *deviates - array of m deviates to be returned by myfunct()
	 * double **derivs - used for user-computed derivatives (see below)
	 * (= 0  when automatic finite differences are computed)
	 */
	inline int gaussianSum(int m, int n, double *p, double *deviates, double **derivs, void *data)
	{
		PulseWaveform *pulse = (PulseWaveform*) data;
		int numPeaks = (n-1)/3;
		float predVal = 0;
		int idx = 0;
		for(int i = 0; i < m; ++i)
		{
			predVal = 0;
			/*
			 * p[0] = noise
             * p[1] = amplitude
			 * p[2] = time offset
			 * p[3] = width
			 */
			for(int j = 0; j < numPeaks; ++j)
			{
				idx = (j * 3) + 1;
				/*cout << "noise: " << p[0] << endl;
                 cout << "Peak " << j << endl;
				 cout << "Amplitude: " << p[idx] << endl;
				 cout << "Time Offset: " << p[idx+1] << endl;
				 cout << "width: " << p[idx+2] << endl << endl;*/
				
				predVal += (p[idx] * exp((-1.0)*
										 (
										  pow(pulse->time[i] - p[idx+1], 2)
										  /
										  (2.0 * pow(p[idx+2], 2))
										   )));
			}
			predVal += p[0];
            //cout << "PredVal = " << predVal << endl;
			//cout << "pulse->intensity[" << i << "] = " << pulse->intensity[i] << endl;
			deviates[i] = (pulse->intensity[i] - predVal) / pulse->error[i];
			//cout << "PredVal - pulse->amplitude[" << i << "] = " << deviates[i] << endl;
		}
		
		return 0;
	}

	/*
	 * int m     - number of data points
	 * int n     - number of parameters
	 * double *p - array of n parameters
	 * double *deviates - array of m deviates to be returned by myfunct()
	 * double **derivs - used for user-computed derivatives (see below)
	 * (= 0  when automatic finite differences are computed)
	 */
	inline int gaussianSumNoNoise(int m, int n, double *p, double *deviates, double **derivs, void *data)
	{
		PulseWaveform *pulse = (PulseWaveform*) data;
		int numPeaks = n/3;
		float predVal = 0;
		int idx = 0;
		for(int i = 0; i < m; ++i)
		{
			predVal = 0;
			/*
			 * p[0] = amplitude
			 * p[1] = time offset
			 * p[2] = width
			 */
			for(int j = 0; j < numPeaks; ++j)
			{
				idx = j * 3;
				/*cout << "Peak " << j << endl;
				 cout << "Amplitude: " << p[idx] << endl;
				 cout << "Time Offset: " << p[idx+1] << endl;
				 cout << "width: " << p[idx+2] << endl;*/
				
				predVal += (p[idx] * exp((-1.0)*
										 (
										  pow(pulse->time[i] - p[idx+1],2)
										  /
										  (2.0 * pow(p[idx+2], 2))
										   )));
			}
            //cout << "PredVal = " << predVal << endl;
			//cout << "pulse->intensity[" << i << "] = " << pulse->intensity[i] << endl;
			deviates[i] = (pulse->intensity[i] - predVal) / pulse->error[i];
			//cout << "PredVal - pulse->amplitude[" << i << "] = " << deviates[i] << endl;
		}
		
		return 0;
	}
	
	inline bool compare_float(float f1, float f2)
	{
		double precision = 0.000000000000000000000001;
		if (((f1 - precision) < f2) && 
			((f1 + precision) > f2))
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	
	inline bool compare_double(double d1, double d2)
	{
		double precision = 0.000000000000000000000001;
		if (((d1 - precision) < d2) && 
			((d1 + precision) > d2))
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	
	inline bool compare_doublefloat(double d1, float f2)
	{
		double precision = 0.000000000000000000000001;
		if (((d1 - precision) < f2) && 
			((d1 + precision) > f2))
		{
			return true;
		}
		else
		{
			return false;
		}
	}
}

#endif
