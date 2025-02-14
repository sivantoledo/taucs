# TAUCS: a Library of Sparse Linear Solvers
### Sivan Toledo, with Doron Chen and Vladimir Rotkin

**September 4, 2003****: Version 2.2!** Multithreading using [Cilk](http://supertech.lcs.mit.edu/cilk/).

**August 21, 2003****: Version 2.1!** Still somewhat preliminary. Main changes are a much better build and configuration system, with (supposedly) perfect out-of-the-box support for Windows and MacOS X, as well as for Linux and Several Unix variants, a unified linear solver interface, preliminary Cilk-parallel support, and some performance improvements.

**July 28, 2003****: Mathematica 5 now uses TAUCS!** More specifically, Mathematica uses TAUCS's direct sparse symmetric-positive-definite solver. You can access the solver using LinearSolve\[A,Method->Cholesky\] when A is a sparse SPD matrix. To the best of my knowledge, the solver is also used inside the interior-point linear-programming solver.

**May 5, 2002****: New version (2.0)!** Complex direct solvers, choice of single or double precision, out-of-core sparse LU with partial pivoting. This version is almost, but now 100%, source-level compatible with earlier versions (we had to change the interface of 3 minor routines to support multiple precisions).  Also, quite a few memory leaks have been fixed.

**December 12, 2001****: New version!** Includes out-of-core sparse Cholesky, support-graph preconditioners, faster factorization routines,  faster construction of Vaidya’s preconditioners, and a few minors additions.

### General

TAUCS is a C library of sparse linear solvers.

Please [let me know](mailto:stoledo@tau.ac.il) if you use the library, especially if you would like to receive email about new versions and bug fixes.

The current version of the library (1.0) includes the following functionality:

*   **Multifrontal Supernodal Cholesky Factorization.** This code is quite fast (several times faster than Matlab 6's sparse Cholesky) but not completely state of the art. It uses the BLAS to factor and compute updates from fundamental supernodes, but it does not use relaxed supernodes.
*   **Left-Looking Supernodal Cholesky Factorization.** Slower than the multifrontal solver but uses less memory.
*   **Drop-Tolerance Incomplete-Cholesky Factorization.** Much slower than the supernodal solvers when it factors a matrix completely, but it can drop small elements from the factorization. It can also modify the diagonal elements to maintain row sums. The code uses a column-based left-looking approach with row lists.
*   **LDL^T Factorization.** Column-based left-looking with row lists. Use the supernodal codes instead.
*   **Out-of-Core, Left-Looking Supernodal Sparse Cholesky Factorization.** Solves huge systems by storing the Cholesky factors in files. Can work with factors whose size is tens of gigabytes on 32-bit machines with 32-bit file systems.
*   **Out-of-Core Sparse LU with Partial Pivoting Factor and Solve.** Can solve huge unsymmetric linear systems.
*   **Ordering Codes and Interfaces to Existing Ordering Codes.** The library includes a unified interface to several ordering codes, mostly existing ones. The ordering codes include Joseph Liu's genmmd (a minimum-degree code in Fortran), Tim Davis's amd codes (approximate minimum degree), Metis (a nested-dissection/minimum-degree code by George Karypis and Vipin Kumar), and a special-purpose minimum-degree code for no-fill ordering of tree-structured matrices. All of these are symmetric orderings.
*   **Matrix Operations.** Matrix-vector multiplication, triangular solvers, matrix reordering.
*   **Matrix Input/Output.** Routines to read and write sparse matrices using a simple file format with one line per nonzero, specifying the row, column, and value. Also routines to read matrices in Harwell-Boeing format.
*   **Matrix Generators.** Routines that generate finite-differences discretizations of 2- and 3-dimensional partial differential equations. Useful for testing the solvers.
*   **Iterative Solvers.** Preconditioned conjugate-gradients and preconditioned minres.
*   **Vaidya's Preconditioners.** Augmented Maximum-weight-basis preconditioners. These preconditioners work by dropping nonzeros from the coefficient matrix and them factoring the preconditioner directly.
*   **Recursive Vaidya's Preconditioners.** These preconditioners also drop nonzeros, but they don't factor the resulting matrix completely. Instead, they eliminate rows and columns which can be eliminated without producing much fill. They then form the Schur complement of the matrix with respect to these rows and columns and drop elements from the Schur complement, and so on. During the preconditioning operation, we solve for the Schur complement elements iteratively.
*   **Multilevel-Support-Graph Preconditioners.** Similar to domain-decomposition preconditioners. Includes the Gremban-Miller preconditioners.
*   **Utility Routines.** Timers (wall-clock and CPU time), physical-memory estimator, and logging.

### Copyright and License

TAUCS Version 2.0, November 29, 2001. Copyright (c) 2001, 2002, 2003 by Sivan Toledo, Tel-Aviv Univesity, stoledo@tau.ac.il. All Rights Reserved.

TAUCS License:

Your use or distribution of TAUCS or any derivative code implies that you agree to this License.

THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY EXPRESSED OR IMPLIED. ANY USE IS AT YOUR OWN RISK.

Permission is hereby granted to use or copy this program, provided that the Copyright, this License, and the Availability of the original version is retained on all copies. User documentation of any code that uses this code or any derivative code must cite the Copyright, this License, the Availability note, and "Used by permission." If this code or any derivative code is accessible from within MATLAB, then typing "help taucs" must cite the Copyright, and "type taucs" must also cite this License and the Availability note. Permission to modify the code and to distribute modified code is granted, provided the Copyright, this License, and the Availability note are retained, and a notice that the code was modified is included. This software is provided to you free of charge.

### Availability

As of version 2.1, we distribute the code in 4 formats: zip and tarred-gzipped (tgz), with or without binaries for external libraries. The bundled external libraries should allow you to build the test programs on Linux, Windows, and MacOS X without installing additional software. We recommend that you download the full distributions, and then perhaps replace the bundled libraries by higher performance ones (e.g., with a BLAS library that is specifically optimized for your machine). If you want to conserve bandwidth and you want to install the required libraries yourself, download the lean distributions. The zip and tgz files are identical, except that on Linux, Unix, and MacOS, unpacking the tgz file ensures that the configure script is marked as executable (unpack with tar zxvpf), otherwise you will have to change its permissions manually.

Click to accept the above license and download:

·       [Version 2.2 of the code, with external libraries, tgz format](2.2/taucs_full.tgz) (8MB)

·       [Version 2.2 of the code, with external libraries, zip format](2.2/taucs_full.zip) (8MB)

·       [Version 2.2 of the code, no external libraries, tgz format](2.2/taucs.tgz) (2MB)

·       [Version 2.2 of the code, no external libraries, zip format](2.2/taucs.zip) (2MB)

·       [Version 2.0](2.0/taucs.tar.gz)

·       [Version 1.0](1.0/taucs.tar.gz) (obsolete)

Last updated on September 4, 2003