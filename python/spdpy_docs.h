/*
 *  spdpy_docs.h
 *
 *  Functions which provide access to SPD/UPD files
 *  from within Python.
 *
 *  Created by Pete Bunting on 04/03/2011.
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
 */

#ifndef SPDPy_docs_H
#define SPDPy_docs_H

static const string SPDFILE_CLASS_DOC = "Class representing the SPD file header.";

static const string SPDNONIDXWRITER_CLASS_DOC = "SPD non indexed file writer.";

static const string SPDSEQIDXWRITER_CLASS_DOC = \
"SPD sequecial index file writer.\n\n \
Below is an example ofhow to use:\n\n \
import spdpy\n \
spdFileOut = spdpy.createSPDFile(\"test.spd\")\n \
spdFileOut.setNumBinsX(20)\n \
spdFileOut.setNumBinsY(20)\n \
spdFileOut.setBinSize(1)\n \
spdWriter = spdpy.SPDPySeqWriter()\n \
spdWriter.open(spdFileOut, \"test.spd\")\n \
pulses = list()\n \
for row in range(spdFileOut.numBinsY):\n \
\tprint \"Row:\", str(row)\n \
\tfor col in range(spdFileOut.numBinsX):\n \
\t\tspdWriter.writeDataColumn(pulses, col, row)\n \
spdWriter.close(spdFileOut)";

static const string SPDNONSEQIDXWRITER_CLASS_DOC = "SPD non sequecial index file writer.";


#endif
