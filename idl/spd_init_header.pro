;+
; NAME:
;
;       SPD_Init_Header
;
; PURPOSE:
;
;       This program initialises an IDL structure for SPD file header
;       See http://www.spdlib.org/wiki for more information on the SORTED PULSE DATA (SPD) file format
;
; AUTHOR:
;
;       John Armston
;
; CALLING SEQUENCE:
;
;       Result = SPD_Init_Header()
;
; ARGUMENTS:
;
;       None
;
; KEYWORDS:
;
;       None
;
; RETURN VALUE:
;
;       The program returns a data structure for SPD file header
;
; DEPENDENCIES:
;
;       None
;
; MODIFICATION HISTORY:
;
;       Written by John Armston, November 2010.
;       Updated for Sorted Pulse Data format 2.1. John Armston, February 2011.
;-

FUNCTION SPD_Init_Header

  COMPILE_OPT IDL2
  
  ; Define the data structure
  header = {SPDheaderformat1,                 $
    AZIMUTH_MAX                  :        0D, $
    AZIMUTH_MIN                  :        0D, $
    BANDWIDTHS                   : fltarr(1), $
    BIN_SIZE                     :       0.0, $
    BLOCK_SIZE_POINT             :     250US, $
    BLOCK_SIZE_PULSE             :     250US, $
    BLOCK_SIZE_RECEIVED          :     250US, $
    BLOCK_SIZE_TRANSMITTED       :     250US, $
    CAPTURE_DAY_OF               :       0US, $
    CAPTURE_HOUR_OF              :       0US, $
    CAPTURE_MINUTE_OF            :       0US, $
    CAPTURE_MONTH_OF             :       0US, $
    CAPTURE_SECOND_OF            :       0US, $
    CAPTURE_YEAR_OF              :       0US, $
    CREATION_DAY_OF              :       0US, $
    CREATION_HOUR_OF             :       0US, $
    CREATION_MINUTE_OF           :       0US, $
    CREATION_MONTH_OF            :       0US, $
    CREATION_SECOND_OF           :       0US, $
    CREATION_YEAR_OF             :       0US, $
    DEFINED_DECOMPOSED_PT        :        0S, $
    DEFINED_DISCRETE_PT          :        0S, $
    DEFINED_HEIGHT               :        0S, $
    DEFINED_ORIGIN               :        0S, $
    DEFINED_RECEIVE_WAVEFORM     :        0S, $
    DEFINED_RGB                  :        0S, $
    DEFINED_TRANS_WAVEFORM       :        0S, $
    FIELD_OF_VIEW                :       0.0, $
    FILE_SIGNATURE               : bytarr(8), $
    FILE_TYPE                    :       0US, $
    GENERATING_SOFTWARE          : bytarr(8), $
    INDEX_TYPE                   :       1US, $
    NUMBER_BINS_X                :       0UL, $
    NUMBER_BINS_Y                :       0UL, $
    NUMBER_OF_POINTS             :      0ULL, $
    NUMBER_OF_PULSES             :      0ULL, $
    NUM_OF_WAVELENGTHS           :       1US, $
    POINT_DENSITY                :       0.0, $
    PULSE_ALONG_TRACK_SPACING    :       0.0, $
    PULSE_ANGULAR_SPACING_AZIMUTH:       0.0, $
    PULSE_ANGULAR_SPACING_ZENITH :       0.0, $
    PULSE_CROSS_TRACK_SPACING    :       0.0, $
    PULSE_DENSITY                :       0.0, $
    PULSE_ENERGY                 :       0.0, $
    PULSE_FOOTPRINT              :       0.0, $
    PULSE_INDEX_METHOD           :        0S, $
    RANGE_MAX                    :       0.0, $
    RANGE_MIN                    :       0.0, $
    RETURN_NUMBERS_SYN_GEN       :        0L, $
    SCANLINE_IDX_MAX             :        0D, $
    SCANLINE_IDX_MIN             :        0D, $
    SCANLINE_MAX                 :        0D, $
    SCANLINE_MIN                 :        0D, $
    SENSOR_APERTURE_SIZE         :       0.0, $
    SENSOR_BEAM_DIVERGENCE       :       0.0, $
    SENSOR_HEIGHT                :        0D, $
    SENSOR_MAX_SCAN_ANGLE        :       0.0, $
    SENSOR_PULSE_REPETITION_FREQ :       0.0, $
    SENSOR_SCAN_RATE             :       0.0, $
    SENSOR_SPEED                 :       0.0, $
    SENSOR_TEMPORAL_BIN_SPACING  :        0D, $
    SPATIAL_REFERENCE            : bytarr(8), $
    SYSTEM_IDENTIFIER            : bytarr(8), $
    USER_META_DATA               : bytarr(8), $
    VERSION_MAJOR_SPD            :       0US, $
    VERSION_MINOR_SPD            :       0US, $
    VERSION_POINT                :       0US, $
    VERSION_PULSE                :       0US, $
    WAVEFORM_BIT_RES             :       0US, $
    WAVELENGTHS                  : fltarr(1), $
    X_MAX                        :        0D, $
    X_MIN                        :        0D, $
    Y_MAX                        :        0D, $
    Y_MIN                        :        0D, $
    ZENITH_MAX                   :        0D, $
    ZENITH_MIN                   :        0D, $
    Z_MAX                        :        0D, $
    Z_MIN                        :        0D  $
    }
    
  ; Return the data structure
  RETURN, header
  
END
