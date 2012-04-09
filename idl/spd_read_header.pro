;+
; NAME:
;
;       SPD_Read_Header
;
; PURPOSE:
;
;       This program reads the header information from a SPD file
;       Returns a simpler IDL structure compared to IDL's H5_PARSE
;       For more information on the SORTED PULSE DATA (SPD) lidar data format, see http://www.spdlib.org/wiki
;
; AUTHOR:
;
;       John Armston
;
; CALLING SEQUENCE:
;
;       Result = SPD_Read_Header(inputFile)
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
;       The program returns an anonymous structure containing the header information.
;
; DEPENDENCIES:
;
;       None
;
; MODIFICATION HISTORY:
;
;       Written by John Armston, November 2010.
;       Added call to H5_CLOSE, February 2012.
;-

FUNCTION SPD_Read_Header, inputFile

  COMPILE_OPT IDL2
  
  ; Open the SPD file
  h5_id = H5F_OPEN(inputFile)
  
  ; Read the SPD header into an IDL structure
  header = CREATE_STRUCT('FILE_NAME', inputFile)
  header_id = H5G_OPEN(h5_id, 'HEADER')
  nHeaderItems = H5G_GET_NUM_OBJS(header_id)
  FOR i = 0UL, nHeaderItems-1UL, 1UL DO BEGIN
    name = H5G_GET_OBJ_NAME_BY_IDX(header_id, i)
    data_id = H5D_OPEN(header_id, name)
    value = H5D_READ(data_id)
    H5D_CLOSE, data_id
    header = CREATE_STRUCT(name, value[0], header)
  ENDFOR
  H5G_CLOSE, header_id
  
  ; Close the SPD file
  H5F_CLOSE, h5_id
  H5_CLOSE
  
  ; Return the result
  RETURN, header
  
END
