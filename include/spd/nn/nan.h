/******************************************************************************
 *
 * File:           nan.h
 *
 * Created:        18/10/2001
 *
 * Author:         Pavel Sakov
 *                 CSIRO Marine Research
 *
 * Purpose:        NaN definition
 *
 * Description:    Should cover machines with 64 bit doubles or other machines
 *                 with GCC
 *
 * Revisions:      None
 *
 *****************************************************************************/

#ifndef SPD_NN_NAN_H
#define SPD_NN_NAN_H

namespace spdlib{ namespace nn{

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)

static const double NaN = 0.0 / 0.0;

#elif defined(_WIN32)

static unsigned _int64 lNaN = ((unsigned _int64) 1 << 63) - 1;

#define NaN (*(double*)&lNaN)

#else

static const long long lNaN = ((unsigned long long) 1 << 63) - 1;

#define NaN (*(double*)&lNaN)

#endif

}}

#endif
