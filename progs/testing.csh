#!/bin/csh -v

# Script to test TAUCS using the example programs
#
# Run from the taucs main directory

bin/$OSTYPE/iter -log stdout -mesh2d 50

bin/$OSTYPE/iter  -ordering genmmd   -log stdout -mesh2d 50             -vaidya -subgraphs 100
bin/$OSTYPE/iter  -ordering genmmd   -log stdout -mesh2d_negative 50    -vaidya -subgraphs 100
bin/$OSTYPE/iter  -ordering genmmd   -log stdout -mesh2d_negative 50    -vaidya -subgraphs 100
bin/$OSTYPE/iter  -ordering genmmd   -log stdout -mesh2d_negative 50    -trick -vaidya -subgraphs 100
bin/$OSTYPE/iter  -ordering genmmd   -log stdout -mesh3d 30 30 30       -vaidya -subgraphs 1000
bin/$OSTYPE/iter  -ordering genmmd   -log stdout -rrn 30 30 30 0.6 1e-8 -vaidya -subgraphs 1000
bin/$OSTYPE/iter  -ordering genmmd   -log stdout -discont 30 30 30 1e8  -vaidya -subgraphs 1000

bin/$OSTYPE/iter  -ordering identity -log stdout -mesh3d 30 30 30        -icc -droptol 0.01 -ordering identity
bin/$OSTYPE/iter  -ordering identity -log stdout -mesh3d 30 30 30        -icc -droptol 0.01 -modified -ordering identity

bin/$OSTYPE/iter  -ordering genmmd   -log stdout -mesh3d 30 30 30        -sg regular:GM:16
bin/$OSTYPE/iter  -ordering genmmd   -log stdout -mesh3d 30 30 30        -sg regular:CT:16
bin/$OSTYPE/iter  -ordering genmmd   -log stdout -mesh3d 30 30 30        -sg regular:CT:1000

bin/$OSTYPE/direct -ordering genmmd   -log stdout -mesh3d 25 25 25        -snmf
bin/$OSTYPE/direct -ordering genmmd   -log stdout -mesh3d 25 25 25        -snll
bin/$OSTYPE/direct -ordering genmmd   -log stdout -mesh3d 25 25 25        -ooc -matrixfile /tmp/taucs 
bin/$OSTYPE/direct -ordering genmmd   -log stdout -mesh3d 25 25 25        -ooc -matrixfile /tmp/taucs -memory 10
bin/$OSTYPE/direct -ordering genmmd   -log stdout -mesh3d 25 25 25







