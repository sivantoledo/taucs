#########################################################
# Linux                                                 #
#########################################################
OBJEXT=.o
LIBEXT=.a
EXEEXT= 
F2CEXT=.f
PATHSEP=/
DEFFLG=-D

FC        = f77
FFLAGS    = -O3 -g -fno-second-underscore -Wall -std=legacy -ffixed-form
FOUTFLG   =-o 

COUTFLG   = -o
CFLAGS    = -O3 -g -D_POSIX_C_SOURCE=199506L -Wall -pedantic -ansi -fPIC -fexceptions -D_GNU_SOURCE 
CFLAGS    = -g -O3 -Wall -Werror -pedantic -ansi 
# for some reason, -std=c99 -pedantic crashes 
# with the error message "imaginary constants are a GCC extension"
# (seems to be a gcc bug, gcc 3.3.1)
CFLAGS    = -O3 -Wall -Werror -std=c89 -pedantic
CFLAGS    = -O3 -Wall -Werror -std=c99
# Sivan Feb 2025
CFLAGS    = -O3 -Wall -std=c99 

LD        = $(CC) 
LDFLAGS   = 
LOUTFLG   = $(COUTFLG)

AR        = ar cr
AOUTFLG   =

RANLIB    = ranlib
RM        = rm -rf

# These are for a Pentium4 version of ATLAS (not bundled with taucs)
#LIBBLAS   = -L /home/stoledo/Public/Linux_P4SSE2/lib -lf77blas -lcblas -latlas \
#            -L /usr/lib/gcc-lib/i386-redhat-linux/2.96 -lg2c
#LIBLAPACK = -L /home/stoledo/Public/Linux_P4SSE2/lib -llapack

LIBBLAS   = -L external/lib/linux -lf77blas -lcblas -latlas
LIBLAPACK = -L external/lib/linux -llapack

LIBMETIS  = -L external/lib/linux -lmetis 

LIBF77 = -lg2c  

# Sivan Feb 2025 trying to get it to compile and link again
LIBBLAS   = -lblas
LIBLAPACK = -llapack
# install on Ubuntu with apt install libmetis-dev
LIBMETIS  = -lmetis 
LIBF77    = -lgfortran

LIBC   = -lm 

#########################################################







