
###############################################################################
# set file locations
set(SPDLIB_SRC_DIR spd)
set(SPDLIB_HEADERS_DIR ../include/spd)
set(CMPFIT_SRC_DIR cmpfit)
set(CMPFIT_HEADER_DIR ../include/spd/cmpfit)
set(TCLAP_HEADERS_DIR ../include/spd/tclap)
set(TPS_SRC_DIR tps)
set(TPS_HEADERS_DIR ../include/spd/tps)
set(ALGLIB_SRC_DIR alglibspd)
set(ALGLIB_HEADERS_DIR ../include/spd/alglibspd)

set(CMPFIT_H
	${CMPFIT_HEADER_DIR}/mpfit.h)

set(TCLAP_H
	${TCLAP_HEADERS_DIR}/Arg.h
	${TCLAP_HEADERS_DIR}/ArgException.h
	${TCLAP_HEADERS_DIR}/ArgTraits.h
	${TCLAP_HEADERS_DIR}/CmdLine.h
	${TCLAP_HEADERS_DIR}/CmdLineInterface.h
	${TCLAP_HEADERS_DIR}/CmdLineOutput.h
	${TCLAP_HEADERS_DIR}/Constraint.h
	${TCLAP_HEADERS_DIR}/DocBookOutput.h
	${TCLAP_HEADERS_DIR}/HelpVisitor.h
	${TCLAP_HEADERS_DIR}/IgnoreRestVisitor.h
	${TCLAP_HEADERS_DIR}/MultiArg.h
	${TCLAP_HEADERS_DIR}/MultiSwitchArg.h
	${TCLAP_HEADERS_DIR}/OptionalUnlabeledTracker.h
	${TCLAP_HEADERS_DIR}/StandardTraits.h
	${TCLAP_HEADERS_DIR}/StdOutput.h
	${TCLAP_HEADERS_DIR}/SwitchArg.h
	${TCLAP_HEADERS_DIR}/UnlabeledMultiArg.h
	${TCLAP_HEADERS_DIR}/UnlabeledValueArg.h
	${TCLAP_HEADERS_DIR}/ValueArg.h
	${TCLAP_HEADERS_DIR}/ValuesConstraint.h
	${TCLAP_HEADERS_DIR}/VersionVisitor.h
	${TCLAP_HEADERS_DIR}/Visitor.h
	${TCLAP_HEADERS_DIR}/XorHandler.h
	${TCLAP_HEADERS_DIR}/ZshCompletionOutput.h)
	
set(TPS_H
	${TPS_HEADERS_DIR}/linalg3d.h 
	${TPS_HEADERS_DIR}/ludecomposition.h
	${TPS_HEADERS_DIR}/spline.h )
	
set(ALGLIB_H
	${ALGLIB_HEADERS_DIR}/alglibinternal.h 
	${ALGLIB_HEADERS_DIR}/alglibmisc.h
	${ALGLIB_HEADERS_DIR}/ap.h
	${ALGLIB_HEADERS_DIR}/integration.h
	${ALGLIB_HEADERS_DIR}/interpolation.h
	${ALGLIB_HEADERS_DIR}/linalg.h
	${ALGLIB_HEADERS_DIR}/optimization.h
	${ALGLIB_HEADERS_DIR}/solvers.h
	${ALGLIB_HEADERS_DIR}/specialfunctions.h
	${ALGLIB_HEADERS_DIR}/stdafx.h )

set(SPDLIB_MIT_H
	${SPDLIB_HEADERS_DIR}/SPDCommon.h
	${SPDLIB_HEADERS_DIR}/SPDDataExporter.h
	${SPDLIB_HEADERS_DIR}/SPDDataImporter.h
	${SPDLIB_HEADERS_DIR}/SPDException.h
	${SPDLIB_HEADERS_DIR}/SPDFile.h
	${SPDLIB_HEADERS_DIR}/SPDFileIncrementalReader.h
	${SPDLIB_HEADERS_DIR}/SPDFileReader.h
	${SPDLIB_HEADERS_DIR}/SPDFileWriter.h
	${SPDLIB_HEADERS_DIR}/SPDIOException.h
	${SPDLIB_HEADERS_DIR}/SPDPoint.h
	${SPDLIB_HEADERS_DIR}/SPDProcessingException.h
	${SPDLIB_HEADERS_DIR}/SPDPulse.h
	${SPDLIB_HEADERS_DIR}/spd-config.h)

set(SPDLIB_H_IO
	${SPDLIB_HEADERS_DIR}/SPDASCIIMultiLineReader.h
	${SPDLIB_HEADERS_DIR}/SPDDecomposedCOOFileImporter.h
	${SPDLIB_HEADERS_DIR}/SPDDecomposedDatFileImporter.h
    ${SPDLIB_HEADERS_DIR}/SPDARADecomposedDatFileImporter.h
	${SPDLIB_HEADERS_DIR}/SPDExportAsReadUnGridded.h
	${SPDLIB_HEADERS_DIR}/SPDExportAsTiles.h
	${SPDLIB_HEADERS_DIR}/SPDFullWaveformDatFileImporter.h
	${SPDLIB_HEADERS_DIR}/SPDGeneralASCIIFileWriter.h
	${SPDLIB_HEADERS_DIR}/SPDIOFactory.h
	${SPDLIB_HEADERS_DIR}/SPDIOUtils.h
	${SPDLIB_HEADERS_DIR}/SPDLASFileExporter.h
	${SPDLIB_HEADERS_DIR}/SPDLASFileImporter.h
	${SPDLIB_HEADERS_DIR}/SPDLineParserASCII.h
	${SPDLIB_HEADERS_DIR}/SPDLineParserASCIIPulsePerRow.h
	${SPDLIB_HEADERS_DIR}/SPDTextFileImporter.h
	${SPDLIB_HEADERS_DIR}/SPDImportSALCAData2SPD.h
	${SPDLIB_HEADERS_DIR}/SPDOptechFullWaveformImport.h)

set(SPDLIB_H_Processing_Interfaces
	${SPDLIB_HEADERS_DIR}/SPDConvertFormats.h
	${SPDLIB_HEADERS_DIR}/SPDMergeFiles.h
	${SPDLIB_HEADERS_DIR}/SPDCalcFileStats.h
	${SPDLIB_HEADERS_DIR}/SPDGenBinaryMask.h
	${SPDLIB_HEADERS_DIR}/SPDSubsetSPDFile.h 
	${SPDLIB_HEADERS_DIR}/SPDExportProcessorSubset.h )
	
set(SPDLIB_H_Processing_Algorithms
	${SPDLIB_HEADERS_DIR}/SPDApplyElevationChange.h
	${SPDLIB_HEADERS_DIR}/SPDCalcMetrics.h
	${SPDLIB_HEADERS_DIR}/SPDClassifyPts.h
	${SPDLIB_HEADERS_DIR}/SPDCopyRemovingClassificationProcessor.h
	${SPDLIB_HEADERS_DIR}/SPDCreateVerticalProfiles.h
	${SPDLIB_HEADERS_DIR}/SPDDecomposeWaveforms.h
	${SPDLIB_HEADERS_DIR}/SPDDefinePulseHeights.h
	${SPDLIB_HEADERS_DIR}/SPDDefineRGBValues.h
	${SPDLIB_HEADERS_DIR}/SPDExtractReturns.h
	${SPDLIB_HEADERS_DIR}/SPDGenerateTiles.h
	${SPDLIB_HEADERS_DIR}/SPDMetrics.h
	${SPDLIB_HEADERS_DIR}/SPDMultiscaleCurvatureGrdClassification.h
	${SPDLIB_HEADERS_DIR}/SPDParameterFreeGroundFilter.h
	${SPDLIB_HEADERS_DIR}/SPDPointGridIndex.h
	${SPDLIB_HEADERS_DIR}/SPDPointInterpolation.h
	${SPDLIB_HEADERS_DIR}/SPDPolyGroundFilter.h
	${SPDLIB_HEADERS_DIR}/SPDProgressiveMophologicalGrdFilter.h
	${SPDLIB_HEADERS_DIR}/SPDRasterInterpolation.h 
	${SPDLIB_HEADERS_DIR}/SPDRemoveVerticalNoise.h
	${SPDLIB_HEADERS_DIR}/SPDSampleInTime.h
	${SPDLIB_HEADERS_DIR}/SPDThinPulses.h
	${SPDLIB_HEADERS_DIR}/SPDWarpData.h
	${SPDLIB_HEADERS_DIR}/SPDTidyGroundReturn.h )

set(SPDLIB_H_Utils
	${SPDLIB_HEADERS_DIR}/SPDFileUtilities.h
	${SPDLIB_HEADERS_DIR}/SPDGridData.h
	${SPDLIB_HEADERS_DIR}/SPDImageException.h
	${SPDLIB_HEADERS_DIR}/SPDImageUtils.h
	${SPDLIB_HEADERS_DIR}/SPDMathsUtils.h
	${SPDLIB_HEADERS_DIR}/SPDMatrixUtils.h
	${SPDLIB_HEADERS_DIR}/SPDTextFileException.h
	${SPDLIB_HEADERS_DIR}/SPDTextFileLineReader.h
	${SPDLIB_HEADERS_DIR}/SPDTextFileUtilities.h
	${SPDLIB_HEADERS_DIR}/SPDVectorUtils.h)
	
set(SPDLIB_H_Frameworks
	${SPDLIB_HEADERS_DIR}/SPDDataBlockProcessor.h
	${SPDLIB_HEADERS_DIR}/SPDPolygonProcessor.h
	${SPDLIB_HEADERS_DIR}/SPDProcessDataBlocks.h
	${SPDLIB_HEADERS_DIR}/SPDProcessPolygons.h
	${SPDLIB_HEADERS_DIR}/SPDProcessPulses.h
	${SPDLIB_HEADERS_DIR}/SPDPulseProcessor.h
	${SPDLIB_HEADERS_DIR}/SPDSetupProcessPolygons.h )

set(CMPFIT_C
	${CMPFIT_SRC_DIR}/mpfit.c )
	
set(TPS_CPP
	${TPS_SRC_DIR}/spline.cpp)
	
set(ALGLIB_CPP
	${ALGLIB_SRC_DIR}/alglibinternal.cpp 
	${ALGLIB_SRC_DIR}/alglibmisc.cpp
	${ALGLIB_SRC_DIR}/ap.cpp
	${ALGLIB_SRC_DIR}/integration.cpp
	${ALGLIB_SRC_DIR}/interpolation.cpp
	${ALGLIB_SRC_DIR}/linalg.cpp
	${ALGLIB_SRC_DIR}/optimization.cpp
	${ALGLIB_SRC_DIR}/solvers.cpp
	${ALGLIB_SRC_DIR}/specialfunctions.cpp )

set(SPDLIB_CPP_MIT_IO
	${SPDLIB_SRC_DIR}/SPDDataExporter.cpp
	${SPDLIB_SRC_DIR}/SPDException.cpp
	${SPDLIB_SRC_DIR}/SPDFile.cpp
	${SPDLIB_SRC_DIR}/SPDFileIncrementalReader.cpp
	${SPDLIB_SRC_DIR}/SPDFileReader.cpp
	${SPDLIB_SRC_DIR}/SPDFileWriter.cpp
	${SPDLIB_SRC_DIR}/SPDIOException.cpp
	${SPDLIB_SRC_DIR}/SPDPoint.cpp
	${SPDLIB_SRC_DIR}/SPDProcessingException.cpp
	${SPDLIB_SRC_DIR}/SPDPulse.cpp )
	
set(SPDLIB_CPP_IO
	${SPDLIB_SRC_DIR}/SPDASCIIMultiLineReader.cpp
	${SPDLIB_SRC_DIR}/SPDDecomposedCOOFileImporter.cpp
	${SPDLIB_SRC_DIR}/SPDDecomposedDatFileImporter.cpp
    ${SPDLIB_SRC_DIR}/SPDARADecomposedDatFileImporter.cpp
	${SPDLIB_SRC_DIR}/SPDExportAsReadUnGridded.cpp
	${SPDLIB_SRC_DIR}/SPDExportAsTiles.cpp
	${SPDLIB_SRC_DIR}/SPDFullWaveformDatFileImporter.cpp
	${SPDLIB_SRC_DIR}/SPDGeneralASCIIFileWriter.cpp
	${SPDLIB_SRC_DIR}/SPDIOFactory.cpp
	${SPDLIB_SRC_DIR}/SPDIOUtils.cpp
	${SPDLIB_SRC_DIR}/SPDLASFileExporter.cpp
	${SPDLIB_SRC_DIR}/SPDLASFileImporter.cpp
	${SPDLIB_SRC_DIR}/SPDLineParserASCII.cpp
	${SPDLIB_SRC_DIR}/SPDLineParserASCIIPulsePerRow.cpp
	${SPDLIB_SRC_DIR}/SPDTextFileImporter.cpp
	${SPDLIB_SRC_DIR}/SPDImportSALCAData2SPD.cpp
	${SPDLIB_SRC_DIR}/SPDOptechFullWaveformImport.cpp )
	
set(SPDLIB_CPP_Processing_Interfaces
	${SPDLIB_SRC_DIR}/SPDConvertFormats.cpp
	${SPDLIB_SRC_DIR}/SPDMergeFiles.cpp
	${SPDLIB_SRC_DIR}/SPDCalcFileStats.cpp
	${SPDLIB_SRC_DIR}/SPDGenBinaryMask.cpp 
	${SPDLIB_SRC_DIR}/SPDSubsetSPDFile.cpp 
	${SPDLIB_SRC_DIR}/SPDExportProcessorSubset.cpp)
	
set(SPDLIB_CPP_Processing_Algorithms
	${SPDLIB_SRC_DIR}/SPDApplyElevationChange.cpp
	${SPDLIB_SRC_DIR}/SPDCalcMetrics.cpp
	${SPDLIB_SRC_DIR}/SPDClassifyPts.cpp
	${SPDLIB_SRC_DIR}/SPDCopyRemovingClassificationProcessor.cpp
	${SPDLIB_SRC_DIR}/SPDCreateVerticalProfiles.cpp
	${SPDLIB_SRC_DIR}/SPDDecomposeWaveforms.cpp
	${SPDLIB_SRC_DIR}/SPDDefinePulseHeights.cpp
	${SPDLIB_SRC_DIR}/SPDDefineRGBValues.cpp
	${SPDLIB_SRC_DIR}/SPDExtractReturns.cpp
	${SPDLIB_SRC_DIR}/SPDGenerateTiles.cpp
	${SPDLIB_SRC_DIR}/SPDMetrics.cpp
	${SPDLIB_SRC_DIR}/SPDMultiscaleCurvatureGrdClassification.cpp
	${SPDLIB_SRC_DIR}/SPDParameterFreeGroundFilter.cpp
	${SPDLIB_SRC_DIR}/SPDPointGridIndex.cpp
	${SPDLIB_SRC_DIR}/SPDPointInterpolation.cpp
	${SPDLIB_SRC_DIR}/SPDPolyGroundFilter.cpp
	${SPDLIB_SRC_DIR}/SPDProgressiveMophologicalGrdFilter.cpp
	${SPDLIB_SRC_DIR}/SPDRasterInterpolation.cpp 
	${SPDLIB_SRC_DIR}/SPDRemoveVerticalNoise.cpp
	${SPDLIB_SRC_DIR}/SPDSampleInTime.cpp
    ${SPDLIB_SRC_DIR}/SPDSplit.cpp
	${SPDLIB_SRC_DIR}/SPDThinPulses.cpp
	${SPDLIB_SRC_DIR}/SPDWarpData.cpp
	${SPDLIB_SRC_DIR}/SPDTidyGroundReturn.cpp )
		
set(SPDLIB_CPP_Utils
	${SPDLIB_SRC_DIR}/SPDFileUtilities.cpp
	${SPDLIB_SRC_DIR}/SPDGridData.cpp
	${SPDLIB_SRC_DIR}/SPDImageException.cpp
	${SPDLIB_SRC_DIR}/SPDImageUtils.cpp
	${SPDLIB_SRC_DIR}/SPDMathsUtils.cpp
	${SPDLIB_SRC_DIR}/SPDMatrixUtils.cpp
	${SPDLIB_SRC_DIR}/SPDTextFileException.cpp
	${SPDLIB_SRC_DIR}/SPDTextFileLineReader.cpp
	${SPDLIB_SRC_DIR}/SPDTextFileUtilities.cpp
	${SPDLIB_SRC_DIR}/SPDVectorUtils.cpp )
	
set(SPDLIB_CPP_Frameworks
	${SPDLIB_SRC_DIR}/SPDProcessDataBlocks.cpp
	${SPDLIB_SRC_DIR}/SPDProcessPolygons.cpp
	${SPDLIB_SRC_DIR}/SPDProcessPulses.cpp
	${SPDLIB_SRC_DIR}/SPDSetupProcessPolygons.cpp )

###############################################################################

###############################################################################
# Group source files for IDE source explorers
source_group("CMake Files" FILES CMakeLists.txt)
source_group("src_spd_MIT" FILES ${SPDLIB_CPP_MIT_IO})
source_group("src_spd_io" FILES ${SPDLIB_CPP_IO})
source_group("src_spd_interface" FILES ${SPDLIB_CPP_Processing_Interfaces})
source_group("src_spd_algorithms" FILES ${SPDLIB_CPP_Processing_Algorithms})
source_group("src_spd_utils" FILES ${SPDLIB_CPP_Utils})
source_group("src_spd_frameworks" FILES ${SPDLIB_CPP_Frameworks})
source_group("src_cmpfit" FILES ${CMPFIT_C})
source_group("src_tps" FILES ${TPS_CPP})
source_group("src_alglib" FILES ${ALGLIB_CPP}) 
source_group("include_spd_MIT" FILES ${SPDLIB_MIT_H})
source_group("include_spd_io" FILES ${SPDLIB_H_IO})
source_group("include_spd_interface" FILES ${SPDLIB_H_Processing_Interfaces})
source_group("include_spd_algorithms" FILES ${SPDLIB_H_Processing_Algorithms})
source_group("include_spd_utils" FILES ${SPDLIB_H_Utils})
source_group("include_spd_frameworks" FILES ${SPDLIB_H_Frameworks})
source_group("include_cmpfit" FILES ${CMPFIT_H})
source_group("include_tclap" FILES ${TCLAP_H})
source_group("include_tps" FILES ${TPS_H}) 
source_group("include_alglib" FILES ${ALGLIB_H}) 
###############################################################################

###############################################################################
# Build and link library

add_library( ${SPDLIB_MITLIB_NAME} ${SPDLIB_CPP_MIT_IO} ${SPDLIB_MIT_H} )
target_link_libraries(${SPDLIB_MITLIB_NAME} ${GDAL_LIBRARIES} ${HDF5_LIBRARIES} )

if (NOT BUILD_SPDIO_ONLY )
    add_library( ${SPDLIB_ALGLIB_NAME} ${ALGLIB_CPP} ${ALGLIB_H} )

    add_library(${SPDLIB_LIB_NAME} ${SPDLIB_CPP_IO} ${SPDLIB_H_IO} ${SPDLIB_CPP_Processing_Interfaces} ${SPDLIB_H_Processing_Interfaces} ${SPDLIB_CPP_Processing_Algorithms} ${SPDLIB_H_Processing_Algorithms} ${SPDLIB_CPP_Utils} ${SPDLIB_H_Utils} ${SPDLIB_CPP_Frameworks} ${SPDLIB_H_Frameworks} ${CMPFIT_C} ${CMPFIT_H} ${TCLAP_H} ${TPS_CPP} ${TPS_H} )
    target_link_libraries(${SPDLIB_LIB_NAME} ${SPDLIB_MITLIB_NAME} ${BOOST_LIBRARIES} ${GDAL_LIBRARIES} ${HDF5_LIBRARIES} ${LIBLAS_LIBRARIES} ${GSL_LIBRARIES} ${CGAL_LIBRARIES} ${XERCESC_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES} ${SPDLIB_ALGLIB_NAME} )
endif(NOT BUILD_SPDIO_ONLY )
###############################################################################

###############################################################################
# Set target properties
SET_TARGET_PROPERTIES(${SPDLIB_MITLIB_NAME}
PROPERTIES
        SOVERSION ${SPDLIB_IO_VERSION}
        VERSION ${SPDLIB_IO_VERSION}
)

if (NOT BUILD_SPDIO_ONLY )
    SET_TARGET_PROPERTIES(${SPDLIB_ALGLIB_NAME}
    PROPERTIES
        SOVERSION ${SPDLIB_ALG_VERSION}
        VERSION ${SPDLIB_ALG_VERSION}
    )
    SET_TARGET_PROPERTIES(${SPDLIB_LIB_NAME}
    PROPERTIES
        SOVERSION ${SPDLIB_VERSION}
        VERSION ${SPDLIB_VERSION}
    )
endif(NOT BUILD_SPDIO_ONLY )
###############################################################################

###############################################################################
# Installation 
install (TARGETS ${SPDLIB_MITLIB_NAME} DESTINATION lib)
if (NOT BUILD_SPDIO_ONLY )
    install (TARGETS ${SPDLIB_LIB_NAME} ${SPDLIB_ALGLIB_NAME} DESTINATION lib)
endif(NOT BUILD_SPDIO_ONLY )
install (FILES ${SPDLIB_MIT_H} DESTINATION include/spd)
install (FILES ${SPDLIB_H_IO} DESTINATION include/spd)
install (FILES ${SPDLIB_H_Processing_Interfaces} DESTINATION include/spd)
install (FILES ${SPDLIB_H_Processing_Algorithms} DESTINATION include/spd)
install (FILES ${SPDLIB_H_Utils} DESTINATION include/spd)
install (FILES ${SPDLIB_H_Frameworks} DESTINATION include/spd)
install (FILES ${CMPFIT_H} DESTINATION include/spd/cmpfit)
install (FILES ${TPS_H} DESTINATION include/spd/tps)
install (FILES ${ALGLIB_H} DESTINATION include/spd/alglibspd)
###############################################################################
