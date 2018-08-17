/*
 *  Copyright (C) 2003,2005 by Jarno Elonen
 *
 *  TPSDemo is Free Software / Open Source with a very permissive
 *  license:
 *
 *  Permission to use, copy, modify, distribute and sell this software
 *  and its documentation for any purpose is hereby granted without fee,
 *  provided that the above copyright notice appear in all copies and
 *  that both that copyright notice and this permission notice appear
 *  in supporting documentation.  The authors make no representations
 *  about the suitability of this software for any purpose.
 *  It is provided "as is" without express or implied warranty.
 *
 *
 *  April 2010: This file was subsequentely included within the MCC-LiDAR
 *              project version 1.0 rc3.
 *
 *  May 2011: The file was incorrporated into SPDLib by Pete Bunting
 *            and the namespace updated to spdlib::tps to avoid confusion
 *            with other distributions. Additionally, the code was updated
 *            such that double precision values are used throughout.
 *
 */

#ifndef SPD_SPLINE_H
#define SPD_SPLINE_H

#include <exception>
#include <vector>
#include <math.h>
// alloca.h not needed? Breaks Windows builds
//#include <alloca.h>

#include <boost/numeric/ublas/matrix.hpp>

#include "spd/tps/linalg3d.h"
#include "spd/tps/ludecomposition.h"

#include "spd/SPDCommon.h"

namespace spdlib{ namespace tps
{
    struct SingularMatrixError : std::runtime_error
  {
      SingularMatrixError() : std::runtime_error("Singular matrix occured while computing thin plate spline")
    {
    }
  };

  //---------------------------------------------------------------------------

  class Spline
  {
    public:
      // Throws SingularMatrixError if a singular matrix is detected.
      Spline(const std::vector<Vec> & control_pts, double regularization);

      double interpolate_height(double x, double z) const;
      double compute_bending_energy() const;

    private:
      unsigned p;
      const std::vector< Vec > & control_points;
      boost::numeric::ublas::matrix<double> mtx_v;
      boost::numeric::ublas::matrix<double> mtx_orig_k;
  };
}}

#endif
