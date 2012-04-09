;+
; NAME:
;
;       SPD_Init_PulseData
;
; PURPOSE:
;
;       This program initialises an IDL structure for SPD pulse data
;       See http://www.spdlib.org/wiki for more information on the SORTED PULSE DATA (SPD) file format
;
; AUTHOR:
;
;       John Armston
;
; CALLING SEQUENCE:
;
;       Result = SPD_Init_PulseData()
;
; ARGUMENTS:
;
;       fileName = The name of the SPD file for which the pulse data is being initialised.
;
; KEYWORDS:
;
;       version  = The pulse version (default is 2)
;
; RETURN VALUE:
;
;       The program returns a data structure for SPD pulse data
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

FUNCTION SPD_Init_PulseData, version=version

  COMPILE_OPT IDL2
  
  ; Keywords
  if not keyword_set(version) then version = 2
  
  ; Define the data structure
  case version of
    1: begin
      data = {SPDpulseformat1, $
        GPS_TIME                              : 0.0D,  $   ; GPS time
        PULSE_ID                              : 0ULL,  $   ; Return number
        X_ORIGIN                              : 0.0D,  $   ; Easting sensor
        Y_ORIGIN                              : 0.0D,  $   ; Northing sensor
        Z_ORIGIN                              : 0.0,   $   ; Elevation sensor
        H_ORIGIN                              : 0.0,   $   ; AGH of sensor
        X_IDX                                 : 0D,    $   ; X index
        Y_IDX                                 : 0D,    $   ; Y index
        AZIMUTH                               : 0.0,   $   ; Azimuth
        ZENITH                                : 0.0,   $   ; Zenith
        NUMBER_OF_RETURNS                     : 0B,    $   ; Number of returns
        NUMBER_OF_WAVEFORM_TRANSMITTED_BINS   : 0US,   $   ; Number of transmitted waveforms
        NUMBER_OF_WAVEFORM_RECEIVED_BINS      : 0US,   $   ; Number of received waveforms
        RANGE_TO_WAVEFORM_START               : 0.0,   $   ; Range to waveform start
        AMPLITUDE_PULSE                       : 0.0,   $   ; Intensity
        WIDTH_PULSE                           : 0.0,   $   ; Return width
        USER_FIELD                            : 0UL,   $   ; User field
        SOURCE_ID                             : 0US,   $   ; Source ID
        EDGE_OF_FLIGHT_LINE_FLAG              : 0B,    $   ; Edge of flight line flag
        SCAN_DIRECTION_FLAG                   : 0B,    $   ; Scan direction flag
        SCAN_ANGLE_RANK                       : 0.0,   $   ; Scan angle rank
        WAVE_NOISE_THRES                      : 0.0,   $   ; Waveform noise threshold
        RECEIVE_WAVE_GAIN                     : 0.0,   $   ; Received waveform gain
        RECEIVE_WAVE_OFFSET                   : 0.0,   $   ; Received waveform offset
        TRANS_WAVE_GAIN                       : 0.0,   $   ; Transmitted waveform gain
        TRANS_WAVE_OFFSET                     : 0.0,   $   ; Transmitted waveform offset
        PTS_START_IDX                         : 0ULL,  $   ; Point start index
        TRANSMITTED_START_IDX                 : 0ULL,  $   ; Transmitted start index
        RECEIVED_START_IDX                    : 0ULL   $   ; Received start index
        }
    end
    2: begin
      data = {SPDpulseformat2, $
        GPS_TIME                              : 0.0D,  $   ; GPS time
        PULSE_ID                              : 0ULL,  $   ; Return number
        X_ORIGIN                              : 0.0D,  $   ; Easting sensor
        Y_ORIGIN                              : 0.0D,  $   ; Northing sensor
        Z_ORIGIN                              : 0.0,   $   ; Elevation sensor
        H_ORIGIN                              : 0.0,   $   ; AGH of sensor
        X_IDX                                 : 0D,    $   ; X index
        Y_IDX                                 : 0D,    $   ; Y index
        AZIMUTH                               : 0.0,   $   ; Azimuth
        ZENITH                                : 0.0,   $   ; Zenith
        NUMBER_OF_RETURNS                     : 0B,    $   ; Number of returns
        NUMBER_OF_WAVEFORM_TRANSMITTED_BINS   : 0US,   $   ; Number of transmitted waveforms
        NUMBER_OF_WAVEFORM_RECEIVED_BINS      : 0US,   $   ; Number of received waveforms
        RANGE_TO_WAVEFORM_START               : 0.0,   $   ; Range to waveform start
        AMPLITUDE_PULSE                       : 0.0,   $   ; Intensity
        WIDTH_PULSE                           : 0.0,   $   ; Return width
        USER_FIELD                            : 0UL,   $   ; User field
        SOURCE_ID                             : 0US,   $   ; Source ID
        EDGE_OF_FLIGHT_LINE_FLAG              : 0B,    $   ; Edge of flight line flag
        SCAN_DIRECTION_FLAG                   : 0B,    $   ; Scan direction flag
        SCAN_ANGLE_RANK                       : 0.0,   $   ; Scan angle rank
        WAVE_NOISE_THRES                      : 0.0,   $   ; Waveform noise threshold
        RECEIVE_WAVE_GAIN                     : 0.0,   $   ; Received waveform gain
        RECEIVE_WAVE_OFFSET                   : 0.0,   $   ; Received waveform offset
        TRANS_WAVE_GAIN                       : 0.0,   $   ; Transmitted waveform gain
        TRANS_WAVE_OFFSET                     : 0.0,   $   ; Transmitted waveform offset
        PTS_START_IDX                         : 0ULL,  $   ; Point start index
        TRANSMITTED_START_IDX                 : 0ULL,  $   ; Transmitted start index
        RECEIVED_START_IDX                    : 0ULL   $   ; Received start index
        }
    end
  endcase
  
  ; Return the data structure
  RETURN, data
  
END
