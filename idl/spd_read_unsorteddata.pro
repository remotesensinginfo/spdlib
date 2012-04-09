;+
; NAME:
;
;       SPD_Read_UnsortedData
;
; PURPOSE:
;
;       This program reads the pulse data from an unsorted SPD (UPD) file
;       For more information on the SORTED PULSE DATA (SPD) lidar data format, see http://www.spdlib.org/wiki
;
; AUTHOR:
;
;       John Armston
;
; CALLING SEQUENCE:
;
;       status = SPD_Read_UnsortedData(inputFile, header, data, $
;                                  pulse=pulse, nPulses=nPulses, all=all, $
;                                  pointData=pointData, twfData=twfData, rwfData=rwfData)
;
; ARGUMENTS:
;
;       inputFile = SPD file
;       header    = SPD header. If not supplied, it is returned.
;       data      = Pulse data (single record for each laser pulse)
;                   This has values for all pulse properties and indices for the point and waveform data.
;
; KEYWORDS:
;
;       pulse     = Set to the index of the first pulse to read. If not passed then 1 is used.
;       nPulses   = Number of pulses to read from the start pulse. Default is 1.
;       all       = Set to read all the data. If set, this keyword will take precedence over pulse and nPulses.
;       pointData = Set to a named variable to read and return the points (individual return) data.
;       twfData   = Set to a named variable to read and return the transmitted waveforms.
;       rwfData   = Set to a named variable to read and return the received waveforms.
;
; RETURN VALUE:
;
;       Status of 0 is no valid pulses and 1 is valid pulse data.
;       Outputs include a structure containing the header information (header) and a
;       array of structures (data) containing all data fields for the user specified SPD pulses.
;
; DEPENDENCIES:
;
;       SPD_Read_Header
;       SPD_Get_Block
;       SPD_Init_PointData
;       SPD_Init_PulseData
;
; MODIFICATION HISTORY:
;
;       Written by John Armston, July 2011.
;       Added call to H5_CLOSE, February 2011.
;-

FUNCTION SPD_Read_UnsortedData, inputFile, header, data, $
    pulse=pulse, nPulses=nPulses, all=all, $
    pointData=pointData, twfData=twfData, rwfData=rwfData
    
  COMPILE_OPT IDL2
  FORWARD_FUNCTION SPD_Get_Block
  FORWARD_FUNCTION SPD_Read_Header
  
  
  ; Open the SPD file and get header
  IF H5F_IS_HDF5(inputFile) THEN BEGIN
    IF (N_TAGS(header) EQ 0) THEN BEGIN
      header = SPD_Read_Header(inputFile)
    ENDIF ELSE BEGIN
      IF (header.FILE_NAME NE inputFile) THEN BEGIN
        header = SPD_Read_Header(inputFile)
      ENDIF
    ENDELSE
    h5_id = H5F_OPEN(inputFile)
  ENDIF ELSE BEGIN
    MESSAGE, inputFile + ' is not HDF-5', /IOERROR
  ENDELSE
  
  ; Determine subset parameters
  IF KEYWORD_SET(all) THEN BEGIN
    pulse = 1UL
    nPulses = header.NUMBER_OF_PULSES
  ENDIF ELSE BEGIN
    IF NOT KEYWORD_SET(pulse) THEN pulse = 1UL
    IF (pulse GT header.NUMBER_OF_PULSES) THEN MESSAGE, 'pulse > header.NUMBER_OF_PULSES'
    IF NOT KEYWORD_SET(nPulses) THEN nPulses = 1UL
  ENDELSE
  
  ; Open the DATA group
  data_id = H5G_OPEN(h5_id, 'DATA')
  
  ; Read the pulse data
  IF (nPulses GT 0) THEN BEGIN
    pulse_cd_id = H5D_OPEN(data_id, 'PULSES')
    data = SPD_Get_Block(pulse_cd_id, pulse-1UL, 1, nPulses)
    H5D_CLOSE, pulse_cd_id
  ENDIF ELSE BEGIN
    H5G_CLOSE, data_id
    H5F_CLOSE, h5_id
    RETURN, 0
  ENDELSE
  
  ; If requested, read the point data
  IF ARG_PRESENT(pointData) THEN BEGIN
    IF (header.DEFINED_DECOMPOSED_PT EQ 1 OR header.DEFINED_DISCRETE_PT EQ 1) THEN BEGIN
      nPoints = TOTAL(data.NUMBER_OF_RETURNS)
      point_cd_id = H5D_OPEN(data_id, 'POINTS')
      pointData = SPD_Get_Block(point_cd_id, data[0].PTS_START_IDX, 1, nPoints)
      H5D_CLOSE, point_cd_id
    ENDIF ELSE BEGIN
      MESSAGE, 'No point data available (or header is wrong!)', /CONTINUE
    ENDELSE
  ENDIF
  
  ; If requested, read the transmitted waveform data
  IF ARG_PRESENT(twfData) THEN BEGIN
    IF (header.DEFINED_TRANS_WAVEFORM EQ 1) THEN BEGIN
      nBins = TOTAL(data.NUMBER_OF_WAVEFORM_TRANSMITTED_BINS)
      twf_cd_id = H5D_OPEN(data_id, 'TRANSMITTED')
      twfData = SPD_Get_Block(twf_cd_id, data[0].TRANSMITTED_START_IDX, 1, nBins)
      H5D_CLOSE, twf_cd_id
    ENDIF ELSE BEGIN
      MESSAGE, 'No waveform data available (or header is wrong!)', /CONTINUE
    ENDELSE
  ENDIF
  
  ; If requested, read the received waveform data
  IF ARG_PRESENT(rwfData) THEN BEGIN
    IF (header.DEFINED_RECEIVE_WAVEFORM EQ 1) THEN BEGIN
      nBins = TOTAL(data.NUMBER_OF_WAVEFORM_RECEIVED_BINS)
      rwf_cd_id = H5D_OPEN(data_id, 'RECEIVED')
      rwfData = SPD_Get_Block(rwf_cd_id, data[0].RECEIVED_START_IDX, 1, nBins)
      H5D_CLOSE, rwf_cd_id
    ENDIF ELSE BEGIN
      MESSAGE, 'No waveform data available (or header is wrong!)', /CONTINUE
    ENDELSE
  ENDIF
  
  ; Close the DATA group and SPD file and return status
  H5G_CLOSE, data_id
  H5F_CLOSE, h5_id
  H5_CLOSE
  RETURN, 1
  
END
