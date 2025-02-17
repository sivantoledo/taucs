#########################################################
# Irix Cilk
# Sivan's comments, 4 Sep 2003:
# This file is for use with the current branch of Cilk
# on or.iucc.ac.il
# Haim, 26 Dec 2003: Small correction to make it work
#########################################################
FC        = g77
FFLAGS    = -O3 -fno-second-underscore -Wall
FOUTFLG   = -o ./

CC        = gcc
COUTFLG   = -o ./
CFLAGS    = -O3 -Wall -Werror -std=c89

CILKC      = /home/or/tau/sivan/Projects/cilk-devel/current/support/cilkclocal
CILKOUTFLG = -o ./
CILKFLAGS  = -O3 -x cilk

LD        = $(CILKC)
LDFLAGS   = 
LOUTFLG   = -o ./

# We use SCSL without it's multiprocessing ability
# The multiprocessing can not be used with pthreads
LIBBLAS   = -lscs

LIBLAPACK = 
LIBMETIS  = -L external/lib/irix -lmetis32

LIBF77 = -L /usr/local/gcc3/lib/gcc-lib/mips-sgi-irix6.5/3.0.4/ -lg2c -lgcc
LIBC = -L /usr/lib32/mips3 -lc -lm 
