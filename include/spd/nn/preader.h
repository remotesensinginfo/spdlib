/******************************************************************************
 *
 * File:           preader.h
 *
 * Created:        29/05/2006
 *
 * Author:         Pavel Sakov
 *                 CSIRO Marine Research
 *
 * Purpose:        A header file with preader.c
 *
 * Revisions:      None
 *
 *****************************************************************************/

#ifndef SPD_NN_PREADER_H
#define SPD_NN_PREADER_H

namespace spdlib{ namespace nn{

struct preader;
typedef struct preader preader;

preader* preader_create1(double xmin, double xmax, double ymin, double ymax, int nx, int ny);
preader* preader_create2(char* fname);
point* preader_getpoint(preader* pr);
void preader_destroy(preader* pr);
    
}}

#endif
