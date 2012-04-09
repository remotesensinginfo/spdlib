;+
; NAME:
;
;       SPD_Init_PointData
;
; PURPOSE:
;
;       This program initialises an IDL structure for SPD point data
;       See http://www.spdlib.org/wiki for more information on the SORTED PULSE DATA (SPD) file format
;
; AUTHOR:
;
;       John Armston
;
; CALLING SEQUENCE:
;
;       Result = SPD_Init_PointData()
;
; ARGUMENTS:
;
;       None
;
; KEYWORDS:
;
;       version  = The point version (default is 2)
;
; RETURN VALUE:
;
;       The program returns a data string for SPD point data
;
; DEPENDENCIES:
;
;       None
;
; MODIFICATION HISTORY:
;
;       Written by John Armston, November 2010.
;       Updated for new Sorted Pulse Data format. John Armston, February 2011.
;       Updated for version 2.1. John Armston, February 2012.
;-

FUNCTION SPD_Init_PointData, version=version

  COMPILE_OPT IDL2
  
  ; Keywords
  if not keyword_set(version) then version = 2
  
  ; Define the data structure
  case version of
    1: begin
      data = {SPDpointformat1,  $
        RETURN_ID                : 0B,   $   ; Return number
        GPS_TIME                 : 0.0D, $   ; GPS time
        X                        : 0.0D, $   ; Easting
        Y                        : 0.0D, $   ; Northing
        Z                        : 0.0,  $   ; Elevation
        HEIGHT                   : 0.0,  $   ; Return height above ground
        RANGE                    : 0.0,  $   ; Return range
        AMPLITUDE_RETURN         : 0.0,  $   ; Return amplitude
        WIDTH_RETURN             : 0.0,  $   ; Return width
        RED                      : 0US,  $   ; Red image channel
        GREEN                    : 0US,  $   ; Green image channel
        BLUE                     : 0US,  $   ; Blue image channel
        CLASSIFICATION           : 0B,   $   ; Return classification
        USER_FIELD               : 0UL,  $   ; User field
        MODEL_KEY_POINT          : 0B,   $   ; Model key point
        LOW_POINT                : 0B,   $   ; Low point
        OVERLAP                  : 0B,   $   ; Overlap
        IGNORE                   : 0B,   $   ; Ignore
        WAVE_PACKET_DESC_IDX     : 0S,   $   ; Waveform packet index
        WAVEFORM_OFFSET          : 0UL   $   ; Waveform offset
        }
    end
    2: begin
      data = {SPDpointformat2,  $
        RETURN_ID                : 0B,   $   ; Return number
        GPS_TIME                 : 0.0D, $   ; GPS time
        X                        : 0.0D, $   ; Easting
        Y                        : 0.0D, $   ; Northing
        Z                        : 0.0,  $   ; Elevation
        HEIGHT                   : 0.0,  $   ; Return height above ground
        RANGE                    : 0.0,  $   ; Return range
        AMPLITUDE_RETURN         : 0.0,  $   ; Return amplitude
        WIDTH_RETURN             : 0.0,  $   ; Return width
        RED                      : 0US,  $   ; Red image channel
        GREEN                    : 0US,  $   ; Green image channel
        BLUE                     : 0US,  $   ; Blue image channel
        CLASSIFICATION           : 0B,   $   ; Return classification
        USER_FIELD               : 0UL,  $   ; User field
        MODEL_KEY_POINT          : 0B,   $   ; Model key point
        LOW_POINT                : 0B,   $   ; Low point
        OVERLAP                  : 0B,   $   ; Overlap
        IGNORE                   : 0B,   $   ; Ignore
        WAVE_PACKET_DESC_IDX     : 0S,   $   ; Waveform packet index
        WAVEFORM_OFFSET          : 0UL   $   ; Waveform offset
        }
    end
  endcase
  
  ; Return the data structure
  RETURN, data
  
END
