/*
 *  SPDTextFileException.h
 *  spdlib_prj
 *
 *  Created by Pete Bunting on 09/10/2009.
 *  Copyright 2009 SPDLib. All rights reserved.
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
 */


#ifndef SPDTextFileException_H
#define SPDTextFileException_H

#include <string>

#include "spd/SPDException.h"

namespace spdlib
{
	class DllExport SPDTextFileException : public SPDException
	{
	public:
		SPDTextFileException();
		SPDTextFileException(const char *message);
		SPDTextFileException(std::string message);
	};
}


#endif






