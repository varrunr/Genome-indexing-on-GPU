##############################################################################

# Add source files here
EXECUTABLE	:= MatrixMult
# CUDA source files (compiled with cudacc)
CUFILES		:= MatrixMult.cu
# CUDA dependency files
CU_DEPS		:= MatrixMult.h
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= main.cpp 

################################################################################
# Rules and targets

include /Developer/GPU\ Computing/C/common/common.mk
