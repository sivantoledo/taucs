#########################################################
# Irix                                                  #
#########################################################
OBJEXT=.o
LIBEXT=.a
EXEEXT= 
F2CEXT=.f
PATHSEP=/
DEFFLG=-D

FC        = f77
FFLAGS    = -64 -mips4 -O3
FOUTFLG   = -o

CC        = cc
CFLAGS    = -64 -mips4 -O3 -diag_suppress 1174,1552
COUTFLG   = -o

LD        = $(CC) 
LDFLAGS   = $(CFLAGS)
LOUTFLG   = $(COUTFLG)

AR        = ar cr
AOUTFLG   =

RANLIB    = echo
RM        = /bin/rm -rf

# Here we use SGIMATH with multiprocessing
# For using the new SCSL see the variant _scsl
LIBBLAS   = -lcomplib.sgimath_mp

LIBLAPACK = 
LIBMETIS  = -L external/lib/irix -lmetis64

LIBF77    = 
LIBC      = -lc -lm

#########################################################
