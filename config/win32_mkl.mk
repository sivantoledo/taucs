#########################################################
# WIN32 MKL                                             #
# Sivan Toledo Febrary 2025                             #
#########################################################

# CFLAGS    = /nologo /O2 /W3 /D "WIN32" /MT
# CFLAGS    = /Zc:__cplusplus /Zp8 /GR /W3 /EHs /nologo /MD /O2 /Oy- -DWIN32 -DNDEBUG

LIBBLAS   = mkl_intel_lp64_dll.lib mkl_intel_thread_dll.lib mkl_core_dll.lib libiomp5md.lib
LIBLAPACK = 

LIBMETIS  = 
# library extracted from a zip file downloaded from https://github.com/albertony/libf2c
LIBF77    = external\\lib\\win32\\libf2c.lib
LIBC      =









