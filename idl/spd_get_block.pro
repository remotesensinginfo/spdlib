;+
; NAME:
;
;       SPD_Get_Block
;
; PURPOSE:
;
;       This function read a block of data from a SPD file.
;       For more information on the SORTED PULSE DATA (SPD) lidar data format, see http://www.spdlib.org/wiki
;
; AUTHOR:
;
;       John Armston
;
; CALLING SEQUENCE:
;
;       Result = SPD_Get_Block(id, index, nBlocks, blockSize, dataType)
;
; ARGUMENTS:
;
;       id        = The Dataspace identifier, returned from H5D_GET_SPACE
;       index     = An m-element vector of integers, where m is the number of dataspace dimensions.
;       nBlocks   = An m-element vector of integers containing the number of blocks to select in each dimension.
;       blockSize = Set to an m-element vector of integers containing the size of a block.
;
; KEYWORDS:
;
;       None
;
; RETURN VALUE:
;
;       Either and array or array of structures depending on the data type being read.
;
; DEPENDENCIES:
;
;       None
;
; MODIFICATION HISTORY:
;
;       Written by John Armston, February 2011.
;-

FUNCTION SPD_Get_Block, id, index, nBlocks, blockSize

  COMPILE_OPT IDL2
  
  ; Get data space ID
  ds_id = H5D_GET_SPACE(id)
  
  ; Create selection for subset
  H5S_SELECT_HYPERSLAB, ds_id, index, nBlocks, BLOCK=blockSize, /RESET
  
  ; Get memory space ID
  ms_id = H5S_CREATE_SIMPLE(blockSize)
  
  ; Read the subset
  data = H5D_READ(id, FILE_SPACE=ds_id, MEMORY_SPACE=ms_id)
  
  ; Close the memory, dataspace and datatype objects
  H5S_CLOSE, ms_id
  H5S_CLOSE, ds_id
  
  ; Return data
  RETURN, data
  
END
