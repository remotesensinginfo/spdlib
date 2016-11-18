#ifndef SPD_NN_CONFIG_H
#define SPD_NN_CONFIG_H

namespace spdlib{ namespace nn{
#if defined(_WIN32)
#define isnan _isnan
#define copysign _copysign
#define rint (int)
#define M_PI 3.14159265358979323846
#define TRILIBRARY
#define NO_TIMER
#endif
}}
#endif

