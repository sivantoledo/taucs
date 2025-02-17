#########################################################
# Linux                                                 #
# Intel Compilers                                       #
# The C compiler defines __INTEL_COMPILER               #
#########################################################
OBJEXT=.o
LIBEXT=.a
EXEEXT= 
F2CEXT=.f
PATHSEP=/
DEFFLG=-D

FC        = ifc
FFLAGS    = -O3 
FOUTFLG   =-o ./

# -Xc: strict ANSI (Xa is extended, -c99 is C99)
# -axW: generate Pentium4 optimized code, as well as generic 386
# -ansi_alias: assume no strange aliases (int to float, etc)
# -fno-fnalias: no array aliasing within functions
CC        = icc
CFLAGS    = -O3 -D_POSIX_C_SOURCE=199506L -c99
CFLAGS    = -O3 -D_POSIX_C_SOURCE=199506L -Xc -axW -ansi_alias -fno-fnalias \
            -I $(HOME)/Tools/MUMPS_4.3/include
COUTFLG   = -o ./

LD        = $(CC) 
# was also -static -Bstatic
LDFLAGS   = -z muldefs -Wl,-rpath /specific/a/home/cc/cs/stoledo/Public
LOUTFLG   = $(COUTFLG)

AR        = ar cr
AOUTFLG   =

RANLIB    = ranlib
RM        = rm -rf

LIBBLAS   = -L$(HOME)/Tools/MUMPS_4.3/lib -ldmumps -lpord \
         -L$(HOME)/Tools/MUMPS_4.3/libseq -lmpiseq \
         -L /home/stoledo/Public -lgoto_p4_512-r0.9

#         -L /home/stoledo/Public -lgoto_p3_256-r0.9

#         -L /specific/a/home/cc/cs/stoledo/Public -lgoto_p4_512-r0.9
#
#         -L /home/stoledo/Public/Linux_P4SSE2/lib  -lf77blas -lcblas -latlas -L /usr/lib/gcc-lib/i386-redhat-linux7/2.96 -lg2c  
#LIBLAPACK = -L /home/stoledo/Public/Linux_P4SSE2/lib -llapack

LIBLAPACK = external/lib/linux/blas_aux.o /specific/a/home/cc/cs/stoledo/Public/lapack_linux.a
LIBMETIS  = -Lexternal/lib/linux -lmetis 

LIBF77 = -lCEPCF90 -lIEPCF90 -lintrins -lF90 -limf -lpthread 
LIBC   = 

#LIBBLAS   = -L /home/stoledo/Public -lgoto_p3_256-r0.9 \
#            external/lib/linux/blas_aux.o
#########################################################







