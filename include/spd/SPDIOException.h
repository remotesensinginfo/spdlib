 /*
  *  SPDIOException.h
  *  spdlib
  *
  *  Created by Pete Bunting on 28/11/2010.
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

#ifndef SPDIOException_H
#define SPDIOException_H

#include <string>

#include "spd/SPDException.h"

// mark all exported classes/functions with DllExport to have
// them exported by Visual Studio
#undef DllExport
#ifdef _MSC_VER
    #ifdef libspdio_EXPORTS
        #define DllExport   __declspec( dllexport )
    #else
        #define DllExport   __declspec( dllimport )
    #endif
#else
    #define DllExport
#endif

namespace spdlib
{
	class DllExport SPDIOException : public SPDException
	{
	public:
		SPDIOException();
		SPDIOException(const char *message);
		SPDIOException(std::string message);
	};
}


#endif


