import spdpy

writeRow = True # Set to false to write one data column at a time instead of an entire row

spdFileIn = spdpy.openSPDFileHeader("/Users/pete/Desktop/apr1dr_1476754e0245792s_20081102212559_aa1m5_subset2_v2.spd")

spdFileOut = spdpy.copySPDFileAttributes(spdFileIn, "/Users/pete/Desktop/apr1dr_1476754e0245792s_20081102212559_aa1m5_subset2_v2_copy.spd")

spdWriter = spdpy.SPDPySeqWriter()
spdWriter.open(spdFileOut, "/Users/pete/Desktop/apr1dr_1476754e0245792s_20081102212559_aa1m5_subset2_v2_copy.spd")

print "Number of Rows: ", str(spdFileIn.numBinsY)
print "Number of Columns: ", str(spdFileIn.numBinsX)

for row in range(spdFileIn.numBinsY):
    print "Row:", str(row)
    rowOfPulses = spdpy.readSPDPulsesRow(spdFileIn, row)

    if writeRow:
        spdWriter.writeDataRow(rowOfPulses, row)
    else:
        col = 0
        #print "list length: ", len(rowOfPulses)
        for pulses in rowOfPulses:
            #print "Column: ", col, " has ", str(len(pulses)), " pulses."
            spdWriter.writeDataColumn(pulses, col, row)
            col = col + 1

print "Completed loop."
spdWriter.close(spdFileOut)
print "Closed writer."
