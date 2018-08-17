/*
 *  One file long C++ library of linear algebra primitives for
 *  simple 3D programs
 *
 *  Copyright (C) 2001-2003 by Jarno Elonen
 *
 *  Permission to use, copy, modify, distribute and sell this software
 *  and its documentation for any purpose is hereby granted without fee,
 *  provided that the above copyright notice appear in all copies and
 *  that both that copyright notice and this permission notice appear
 *  in supporting documentation.  The authors make no representations
 *  about the suitability of this software for any purpose.
 *  It is provided "as is" without express or implied warranty.
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

#ifndef SPD_GEOM_H
#define SPD_GEOM_H

//#include <cmath>
#include <math.h>

namespace spdlib{ namespace tps
{

    #define EPSILON 0.00001f
    #define PI 3.1415926
    #define Deg2Rad(Ang) ((double)( Ang * PI / 180.0 ))
    #define Rad2Deg(Ang) ((double)( Ang * 180.0 / PI ))

    // =========================================
    // 3-vector
    // =========================================
    class Vec
    {
    public:

      // Position
      double x, y, z;

      // Default constructor
      Vec()
      : x( 0 ), y( 0 ), z( 0 ) {}

      // Element constructor
      Vec( double x, double y, double z )
      : x( x ), y( y ), z( z ) {}

      // Copy constructor
      Vec( const Vec& a )
      : x( a.x ), y( a.y ), z( a.z ) {}

      // Norm (len^2)
      inline double norm() const { return x*x + y*y + z*z; }

      // Length of the vector
      inline double len() const { return (double)sqrt(norm()); }

      Vec &operator += ( const Vec &src ) { x += src.x; y += src.y; z += src.z; return *this; }
      Vec operator + ( const Vec &src ) const { Vec tmp( *this ); return ( tmp += src ); }
      Vec &operator -= ( const Vec &src ) { x -= src.x; y -= src.y; z -= src.z; return *this; }
      Vec operator - ( const Vec &src ) const { Vec tmp( *this ); return ( tmp -= src ); }

      Vec operator - () const { return Vec(-x,-y,-z); }

      Vec &operator *= ( const double src ) { x *= src; y *= src; z *= src;  return *this; }
      Vec operator * ( const double src ) const { Vec tmp( *this ); return ( tmp *= src ); }
      Vec &operator /= ( const double src ) { x /= src; y /= src; z /= src; return *this; }
      Vec operator / ( const double src ) const { Vec tmp( *this ); return ( tmp /= src ); }

      bool operator == ( const Vec& b) const { return ((*this)-b).norm() < EPSILON; }
      //bool operator == ( const Vec& b) const { return x==b.x && y==b.y && z==b.z; }
    };

      // Left hand double multplication
      inline Vec operator * ( const double src, const Vec& v ) { Vec tmp( v ); return ( tmp *= src ); }

      // Dot product
      inline double dot( const Vec& a, const Vec& b )
      { return a.x*b.x + a.y*b.y + a.z*b.z; }

      // Cross product
      inline Vec cross( const Vec &a, const Vec &b )
      { return Vec( a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x ); }


    // =========================================
    // 4 x 4 matrix
    // =========================================
    class Mtx
    {
    public:

      // 4x4, [[0 1 2 3] [4 5 6 7] [8 9 10 11] [12 13 14 15]]
      double data[ 16 ];

      // Creates an identity matrix
      Mtx()
      {
        for ( int i = 0; i < 16; ++i )
          data[ i ] = 0;
        data[ 0 + 0 ] = data[ 4 + 1 ] = data[ 8 + 2 ] = data[ 12 + 3 ] = 1;
      }

      // Returns the transpose of this matrix
      Mtx transpose() const
      {
        Mtx m;
        int idx = 0;
        for ( int row = 0; row < 4; ++row )
          for ( int col = 0; col < 4; ++col, ++idx )
            m.data[ idx ] = data[ row + ( col * 4 ) ];
        return m;
      }

      // Operators
      double operator () ( unsigned column, unsigned row )
      { return data[ column + ( row * 4 ) ]; }
    };

    // Creates a scale matrix
    Mtx scale( const Vec &scale );

    // Creates a translation matrix
    Mtx translate( const Vec &moveAmt );

    // Creates an euler rotation matrix (by X-axis)
    Mtx rotateX( double ang );

    // Creates an euler rotation matrix (by Y-axis)
    Mtx rotateY( double ang );

    // Creates an euler rotation matrix (by Z-axis)
    Mtx rotateZ( double ang );

    // Creates an euler rotation matrix (pitch/head/roll (x/y/z))
    Mtx rotate( double pitch, double head, double roll );

    // Creates an arbitraty rotation matrix
    Mtx makeRotationMatrix( const Vec &dir, const Vec &up );

    // Transforms a vector by a matrix
    inline Vec operator * ( const Vec& v, const Mtx& m )
    {
      return Vec(
        m.data[ 0 ] * v.x + m.data[ 1 ] * v.y + m.data[ 2 ] * v.z + m.data[ 3 ],
        m.data[ 4 ] * v.x + m.data[ 5 ] * v.y + m.data[ 6 ] * v.z + m.data[ 7 ],
        m.data[ 8 ] * v.x + m.data[ 9 ] * v.y + m.data[ 10 ] * v.z + m.data[ 11 ] );
    }

    // Multiplies a matrix by another matrix
    Mtx operator * ( const Mtx& a, const Mtx& b );

    // =========================================
    // Plane
    // =========================================
    class Plane
    {
    public:
      enum PLANE_EVAL
      {
        EVAL_COINCIDENT,
        EVAL_IN_BACK_OF,
        EVAL_IN_FRONT_OF,
        EVAL_SPANNING
      };

      Vec normal;
      double d;

      // Default constructor
      Plane(): normal( 0,1,0 ), d( 0 ) {}

      // Vector form constructor
      //   normal = normalized normal of the plane
      //   pt = any point on the plane
      Plane( const Vec& normal, const Vec& pt )
        : normal( normal ), d( dot( -normal, pt )) {}

      // Copy constructor
      Plane( const Plane& a )
        : normal( a.normal ), d( a.d ) {}

      // Classifies a point (<0 == back, 0 == on plane, >0 == front)
      double classify( const Vec& pt ) const
      {
        double f = dot( normal, pt ) + d;
        return ( f > -EPSILON && f < EPSILON ) ? 0 : f;
      }
    };
}}

#endif
