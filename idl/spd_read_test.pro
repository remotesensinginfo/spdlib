; PURPOSE:
;
;       Read an entire file by row then prints the memory and time required
;       For more information on the SORTED PULSE DATA (SPD) lidar data format, see http://www.spdlib.org/wiki
;
; AUTHOR:
;
;       John Armston
;
; CALLING SEQUENCE:
;
;       status = SPD_Read_Test(inputFile)
;
; ARGUMENTS:
;
;       inputFile = SPD file
;
; KEYWORDS:
;
;       None
;
; RETURN VALUE:
;
;       None
;
; DEPENDENCIES:
;
;       SPD_Read_Header
;       SPD_Read_Data_Row
;
; MODIFICATION HISTORY:
;
;       Written for SPDLib 2.1. John Armston, February 2012.
;-
PRO SPD_READ_TEST, inputFile
  
  ; Get current time and memory
  tstart = systime(1)
  mstart = memory(/current)
  
  ; Read in each row
  header = SPD_Read_Header(inputFile)
  for i=0L, header.NUMBER_BINS_Y-1L, 1L do begin
    status = SPD_Read_Data_Row(inputFile, header, data, row=i, nRow=1, index=index, count=count, pointData=pointData, twfData=twfData, rwfData=rwfData)
    ; print, "Number of pulses in row " + strtrim(i, 2) + ": " + strtrim(ulong64(total(count)), 2)
  endfor
  
  ; print time and memory required
  print, 'Time required: ', systime(1) - tstart, ' seconds'
  print, 'Memory required: ', memory(/highwater) - mstart, ' bytes'
  
END
