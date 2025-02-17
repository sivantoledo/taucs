/*************************************************************************************
 * MultiLU - An unsymmetric multifrontal linear solve library
 *
 * Module: multilu_dense.h
 *
 * Description: Defines the API for a set of basic dense matrix operations that are used
 *              in the code. Using _MULTILU_BLAS we know which BLAS to use. A thing
 *              to notice is whether it is a "STANDARD BLAS" or "API BLAS".
 *              Another thing is whether it is a "CILKED" blas or not.
 *
 *************************************************************************************/
#ifndef _MULTILU_DENSE_H
#define _MULTILU_DENSE_H

#if defined(_MULTILU_CILKED) && defined(_MULTILU_CILKC)
#pragma lang -C
#endif

/*************************************************************************************
 * Calling method for dense subroutines
 *
 * Even if we are building a cilked version we may not be using a cilked blas so 
 * we will be no use for spawnning and syncing. 
 * So we define DENSE_BLAS_CILK, DENSE_BLAS_SYNC and DENSE_BLAS_SPAWN to tell how to
 * call the BLAS.
 * Same for LAPACK.
 *************************************************************************************/

/* Decide if it a standard BLAS or not */
/* If it is then we use a CBLAS interface */
#if defined(_MULTILU_BLAS_ATLAS) || defined(_MULTILU_BLAS_GOTO) || defined(_MULTILU_BLAS_SGIMATH)
#define _MULTILU_STANDARD_BLAS
#endif
#if defined(_MULTILU_BLAS_XDGEMM) || defined(_MULTILU_BLAS_CILKED_XDGEMM) || defined(_MULTILU_BLAS_SUNPERF) 
#define _MULTILU_API_BLAS
#endif

/* If it is cilked or not... */
#ifdef _MULTILU_BLAS_CILKED_XDGEMM
#ifndef _MULTILU_CILKED
#error "Cilked BLAS must be used inside cilked MultiLU!"
#endif
#define _MULTILU_CILKED_BLAS
#endif
#if defined(_MULTILU_BLAS_ATLAS) || defined(_MULTILU_BLAS_GOTO) || defined(_MULTILU_BLAS_SUNPERF) || defined(_MULTILU_BLAS_XDGEMM) || defined(_MULTILU_BLAS_SGIMATH) || defined(_MULTILU_BLAS_SCS)
#define _MULTILU_NONCILKED_BLAS
#endif

/* Unknown type of BLAS... assume standard and non-cilked */
#if !defined(_MULTILU_STANDARD_BLAS) && !defined(_MULTILU_API_BLAS)
#define _MULTILU_STANDARD_BLAS
#endif
#if !defined(_MULTILU_CILKED_BLAS) && !defined(_MULTILU_NONCILKED_BLAS)
#define _MULTILU_NONCILKED_BLAS
#endif

/* TODO: Rid of temprorary */
#ifdef _MULTILU_CILKED
/*#define _MULTILU_NO_FORCE_P */
#define _MULTILU_FORCE_P
#else
#define _MULTILU_NO_FORCE_P
#endif

/* Conventaions for cilk on dense subsystem */
#if defined(_MULTILU_NONCILKED_BLAS) && ! defined(_MULTILU_FORCE_P)

#define DENSE_BLAS_CILK
#define DENSE_BLAS_SPAWN
#define DENSE_BLAS_SYNC

#define DENSE_LAPACK_CILK
#define DENSE_LAPACK_SPAWN
#define DENSE_LAPACK_SYNC

#else

#define DENSE_BLAS_CILK    cilk
#define DENSE_BLAS_SPAWN   spawn 
#define DENSE_BLAS_SYNC    sync

#define DENSE_LAPACK_CILK  cilk
#define DENSE_LAPACK_SPAWN spawn 
#define DENSE_LAPACK_SYNC  sync

#endif

/*************************************************************************************
 * BLAS abstraction
 *************************************************************************************/

/*************************************************************************************
 * Function: CeqMABT
 *
 * Description: Makes C = -A * B'. A is mxk. B is nxk. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
DENSE_BLAS_CILK void CeqMABT(int m, int n, int k, 
			     double *A, int ld_a, 
			     double *B, int ld_b, 
			     double *C, int ld_c);

/*************************************************************************************
 * Function: CaddMABT
 *
 * Description: Makes C -= A * B'. A is mxk. B is nxk. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
DENSE_BLAS_CILK void CaddMABT(int m, int n, int k, 
			      double *A, int ld_a, 
			      double *B, int ld_b, 
			      double *C, int ld_c);

/*************************************************************************************
 * Function: CaddMAB
 *
 * Description: Makes C -= A * B. A is mxk. B is kxn. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
DENSE_BLAS_CILK void CaddMAB(int m, int n, int k, 
			     double *A, int ld_a, 
			     double *B, int ld_b, 
			     double *C, int ld_c);

/*************************************************************************************
 * Function: CaddMATB
 *
 * Description: Makes C -= A' * B. A is kxm. B is kxn. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
DENSE_BLAS_CILK void CaddMATB(int m, int n, int k, 
			      double *A, int ld_a, 
			      double *B, int ld_b, 
			      double *C, int ld_c);

/*************************************************************************************
 * Function: UnitLowerRightTriSolve
 *
 * Description: Solves the unit-lower right triangular system.
 *
 *************************************************************************************/
DENSE_BLAS_CILK void UnitLowerRightTriSolve(int m, int n, 
					    double *L, int ld_l, 
					    double *B, int ld_b);


/*************************************************************************************
 * Function: UnitLowerLeftTriSolve
 *
 * Description: Solves the unit-lower left triangular system.
 *
 *************************************************************************************/
DENSE_BLAS_CILK void UnitLowerLeftTriSolve(int m, int n, 
					   double *L, int ld_l, 
					   double *B, int ld_b);

/*************************************************************************************
 * Function: UnitLowerLeftTriSolve
 *
 * Description: Solves the upper left triangular system.
 *
 *************************************************************************************/
DENSE_BLAS_CILK void UpperLeftTriSolve(int m, int n, 
				       double *L, int ld_l, 
				       double *B, int ld_b);

/*************************************************************************************
 * LAPACK abstraction
 *************************************************************************************/

/*************************************************************************************
 * Function: LU
 *
 * Description: LU factorizes A which is mxn with load lda. rows gives rows number.
 *              On output it is the new row-ordering. This is priority LUing with
 *              priorities given in prty. At each step we choose the row with the 
 *              lowest priority with abs value bigger the the max abs value.
 *              A workspace of m elements is requested.
 *
 *************************************************************************************/
DENSE_LAPACK_CILK void LU(double *A, int m, int n, int lda, 
			  double thresh, int *prty, int *rows, int *workspace);
 
#endif

