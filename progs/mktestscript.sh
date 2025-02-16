#!/bin/bash
#
# TAUCS: this makes the test script for windows and unix
#

/bin/rm -f testscript.bat
printf "REM This is an automatically-generated script\r\n" > testscript.bat
printf "del testscript.log\r\n" >> testscript.bat
printf "echo TAUCS TEST LOG >  testscript.log\r\n" >> testscript.bat
printf "echo ============== >> testscript.log\r\n" >> testscript.bat
printf "echo Win32          >> testscript.log\r\n" >> testscript.bat
printf "echo ============== >> testscript.log\r\n" >> testscript.bat

/bin/rm -f testscript
printf "#!/bin/bash\n" > testscript
printf "### This is an automatically-generated script\n" >> testscript
printf "/bin/rm testscript.log\n" >> testscript
printf "echo 'TAUCS TEST LOG' > testscript.log\n" >> testscript
printf "hostname >> testscript.log\n" >> testscript
printf "uname    >> testscript.log\n" >> testscript
printf "date     >> testscript.log\n" >> testscript
printf "echo '==============' >> testscript.log\n" >> testscript
printf "echo 'trying to maximize stack size:' >> testscript.log\n" >> testscript
printf "ulimit -s >> testscript.log\n" >> testscript
printf "ulimit -s unlimited >> testscript.log\n" >> testscript
printf "ulimit -s >> testscript.log\n" >> testscript
printf "echo '==============' >> testscript.log\n" >> testscript

chmod 755 testscript

for f in progs/test_*.c ; do
  echo $f
  
  name=`basename $f .c`
  bs='\\'
  fs='/'
  printf "echo =============== >> testscript.log\r\n" >> testscript.bat
  printf "echo = $name = >> testscript.log\r\n"       >> testscript.bat
  printf "call configure in=progs${bs}$name.c %%*\r\n" >> testscript.bat
#  printf "nmake /F build${bs}%%TAUCS_LASTCONF%${bs}makefile clean\r\n" >> testscript.bat
  printf "nmake /F build${bs}%%TAUCS_LASTCONF%%${bs}makefile \r\n"      >> testscript.bat
  printf "bin${bs}%%TAUCS_LASTCONF%%${bs}$name >> testscript.log\r\n" >> testscript.bat
  printf "if errorlevel 1 goto :error_$name\r\n"   >> testscript.bat
  printf "echo = TEST PASSED ($name) >> testscript.log\r\n" >> testscript.bat
  printf "goto :next_$name\r\n"                    >> testscript.bat
  printf ":error_$name\r\n"                        >> testscript.bat
  printf "echo = TEST FAILED ($name) >> testscript.log\r\n" >> testscript.bat
  printf ":next_$name\r\n"                         >> testscript.bat
  printf "echo =============== >> testscript.log\r\n" >> testscript.bat

  printf "echo =============== >> testscript.log\n" >> testscript
  printf "echo = $name = >> testscript.log\n"       >> testscript
  printf ". ./configure in=progs${fs}$name.c \$*\n" >> testscript
  printf "echo last conf is \$TAUCS_LASTCONF >> testscript.log\n" >> testscript
#  printf "make -f build${fs}\${TAUCS_LASTCONF}${fs}makefile clean\n"  >> testscript
  printf "make -f build${fs}\${TAUCS_LASTCONF}${fs}makefile\n"        >> testscript
  printf "if bin${fs}\${TAUCS_LASTCONF}${fs}$name >> testscript.log ; then\n" >> testscript
  printf "echo = TEST PASSED $name >> testscript.log\n" >> testscript
  printf "else\n"                    >> testscript
  printf "echo = TEST FAILED $name >> testscript.log\n" >> testscript
  printf "fi\n"                         >> testscript
  printf "echo =============== >> testscript.log\n" >> testscript
done
