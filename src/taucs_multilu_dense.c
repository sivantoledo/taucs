/*************************************************************************************
 * MultiLU - An unsymmetric multifrontal linear solve library
 *
 * Module: multilu_dense.c
 *
 * Description: Code for a set of basic dense matrix operations that are used
 *              in the code. The BLAS and LAPACK code used is according to the 
 *              configuration.
 *
 *************************************************************************************/
#include "taucs.h"

/*#include "taucs_multilu_std.h"*/

#include "taucs_multilu_dense.h"
/*#include "taucs_multilu_profile.h"*/
/*#include "taucs_multilu_param.h"*/

/* MULTILU_XDGEMM_MEM_PAD: Size of memory pad when using XDGEMM */
#define MULTILU_XDGEMM_MEM_PAD          0

/* Forcing parallelism on dense functions parameters: */
/* We have an option to force parallelism using CILK on the dense functions when using any */
/* BLAS. We do by recursively running the function, cilking independent runs and cutting to  */
/* BLAS on small matrices. We also cut to very simple code on very small matrices. The following */
/* parameters determine what is "small" for cutting to BLAS (per function) and what is "very small"  */
/* for cutting to very simple code */

#define MULTILU_THRESHOLD_GEMM_SMALL    0
#define MULTILU_THRESHOLD_GEMM_BLAS    INT_MAX
#define MULTILU_THRESHOLD_TRSM_SMALL    0 
#define MULTILU_THRESHOLD_TRSM_BLAS    INT_MAX
#define MULTILU_THRESHOLD_LASWP_LAPACK INT_MAX
#define MULTILU_THRESHOLD_GETRF_SMALL   0 
#define MULTILU_THRESHOLD_GETRF_LAPACK INT_MAX
#define MULTILU_THRESHOLD_GEMM_FORCEP   0

#define DGER   taucs_ger
#define DGEMM  taucs_gemm
#define DTRSM  taucs_trsm
#define DGETRF taucs_getrf

/* 
   The sun performance library has a "bug": it uses 0-based indices in ipiv
   in GETRF. This is documented in their manuals.
*/
#ifdef _MULTILU_LAPACK_SUNPERF
#define _MULTILU_LAPACK_ZERO_IND 0
#else
#define _MULTILU_LAPACK_ZERO_IND 1
#endif

#define TAUCS_ALWAYS_USES_LAPACK

/*************************************************************************************
 * Conventaions for calling cilk on BLAS (if cilked or not)
 *************************************************************************************/

#ifdef TAUCS_CILK_BLAS

#define BLAS_CILK  cilk
#define BLAS_SPAWN spawn 
#define BLAS_SYNC  sync

#else

#define BLAS_CILK
#define BLAS_SPAWN
#define BLAS_SYNC

#endif

/*************************************************************************************
 * Conventaions for calling cilk on LAPACK (if cilked or not)
 *************************************************************************************/
#ifdef TAUCS_CILK_LAPACK

#define LAPACK_CILK  cilk
#define LAPACK_SPAWN spawn 
#define LAPACK_SYNC  sync

#else

#define LAPACK_CILK
#define LAPACK_SPAWN
#define LAPACK_SYNC

#endif

#if 0
/*************************************************************************************
 * Standard BLAS Functions name
 *************************************************************************************/
#if  defined(WIN32) && ! defined(_MULTILU_BLAS_SUNPERF)
#define DGER   dger
#define DGEMM  dgemm
#define DTRSM  dtrsm
#endif

#if defined(UNIX) && ! defined(_MULTILU_BLAS_SUNPERF)
#define DGER   dger_
#define DGEMM  dgemm_
#define DTRSM  dtrsm_
#endif

#if defined(_MULTILU_BLAS_SUNPERF)
#define DGER   dger
#define DGEMM  dgemm
#define DTRSM  dtrsm
#endif

/*************************************************************************************
 * Standard LAPACK Functions name
 *************************************************************************************/
#if  defined(WIN32) && ! defined(_MULTILU_LAPACK_SUNPERF)
#define DGETRF dgetrf
#endif

#if defined(UNIX) && ! defined(_MULTILU_LAPACK_SUNPERF)
#define DGETRF dgetrf_
#endif

#if defined(_MULTILU_LAPACK_SUNPERF)
#define DGETRF dgetrf
#define DLASWP dlaswp
#endif

#endif /* 0, definition of blas and lapack routines */

/*************************************************************************************
 * No BLAS
 *************************************************************************************/

#ifdef _MULTILU_BLAS_NONE

/*************************************************************************************
 * Function: S_CeqMABT
 *
 * Description: Makes C = - A * B'. A is mxk. B is nxk. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
void S_CeqMABT(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  int i, j, l;
  
  memset(C, 0, n * m * sizeof(double));
  for(i = 0; i < m; i++)
    for(j = 0; j < n; j++)
      for(l = 0; l < k; l++)
	C[i + j * ld_c] -= A[i + l * ld_a] * B[j + l * ld_b];
  
}

/*************************************************************************************
 * Function: S_CaddMABT
 *
 * Description: Makes C -=  A * B'. A is mxk. B is nxk. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
void S_CaddMABT(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  int i, j, l;
  
  for(i = 0; i < m; i++)
    for(j = 0; j < n; j++)
      for(l = 0; l < k; l++)
	C[i + j * ld_c] -= A[i + l * ld_a] * B[j + l * ld_b];
  
}

/*************************************************************************************
 * Function: S_CaddMAB
 *
 * Description: Makes C -=  A * B. A is mxk. B is kxn. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
void S_CaddMAB(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  int i, j, l;
  
  for(i = 0; i < m; i++)
    for(j = 0; j < n; j++)
      for(l = 0; l < k; l++)
	C[i + j * ld_c] -= A[i + l * ld_a] * B[l + j * ld_b];
  
}
#endif


/*************************************************************************************
 * Standard BLAS
 *************************************************************************************/
#ifdef _MULTILU_STANDARD_BLAS

/*************************************************************************************
 * Function: S_UnitLowerRightTriSolve
 *
 * Description: Solves the unit-lower right triangular system.
 *
 *************************************************************************************/
BLAS_CILK void S_UnitLowerRightTriSolve(int m, int n, double *L, int ld_l, double *B, int ld_b)
{
  double dbl_one = 1.0;
  
  BLAS_SPAWN DTRSM("Right", "Lower", "Transpose", "Unit", &m, &n, &dbl_one, 
		   L, &ld_l, B, &ld_b);
  
  BLAS_SYNC;
}

/*************************************************************************************
 * Function: S_UnitLowerLeftTriSolve
 *
 * Description: Solves the unit-lower left triangular system.
 *
 *************************************************************************************/
BLAS_CILK void S_UnitLowerLeftTriSolve(int m, int n, double *L, int ld_l, double *B, int ld_b)
{
  double dbl_one = 1.0;
  
  BLAS_SPAWN DTRSM("Left", "Lower", "No Transpose", "Unit", &m, &n, &dbl_one, 
		   L, &ld_l, B, &ld_b);
  
  BLAS_SYNC;
}

/*************************************************************************************
 * Function: S_UpperLeftTriSolve
 *
 * Description: Solves the upper left triangular system.
 *
 *************************************************************************************/
BLAS_CILK void S_UpperLeftTriSolve(int m, int n, double *L, int ld_l, double *B, int ld_b)
{
  double dbl_one = 1.0;
  
  BLAS_SPAWN DTRSM("Left", "Upper", "No Transpose", "Non Unit", &m, &n, &dbl_one, 
		   L, &ld_l, B, &ld_b);
  
  BLAS_SYNC;
}

/*************************************************************************************
 * Function: S_CeqMABT
 *
 * Description: Makes C = -A * B'. A is mxk. B is nxk. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
BLAS_CILK void S_CeqMABT(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  double dbl_mone = -1.0;
  double dbl_zero = 0.0;
  
  BLAS_SPAWN DGEMM("No Operation", "Transpose", &m, &n, &k, 
		   &dbl_mone, A, &ld_a, B, &ld_b,
		   &dbl_zero, C, &ld_c);
  BLAS_SYNC;
}

/*************************************************************************************
 * Function: S_CaddMABT
 *
 * Description: Makes C -= A * B'. A is mxk. B is nxk. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
BLAS_CILK void S_CaddMABT(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  double dbl_mone = -1.0;
  double dbl_one = 1.0;

  BLAS_SPAWN DGEMM("No Operation", "Transpose", &m, &n, &k, 
		   &dbl_mone, A, &ld_a, B, &ld_b,
		   &dbl_one, C, &ld_c);
  BLAS_SYNC;

}

/*************************************************************************************
 * Function: S_CaddMAB
 *
 * Description: Makes C -= A * B. A is mxk. B is nxk. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
BLAS_CILK void S_CaddMAB(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  double dbl_mone = -1.0;
  double dbl_one = 1.0;
  
  BLAS_SPAWN DGEMM("No Operation", "No Operation", &m, &n, &k, 
		   &dbl_mone, A, &ld_a, B, &ld_b,
		   &dbl_one, C, &ld_c);
  BLAS_SYNC;
}

/*************************************************************************************
 * Function: S_CaddMAB
 *
 * Description: Makes C -= A * B. A is kxm. B is nxk. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
BLAS_CILK void S_CaddMATB(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  double dbl_mone = -1.0;
  double dbl_one = 1.0;
  
  BLAS_SPAWN DGEMM("Transpose", "No Operation", &m, &n, &k, 
		   &dbl_mone, A, &ld_a, B, &ld_b,
		   &dbl_one, C, &ld_c);
  BLAS_SYNC;
}

#endif

/*************************************************************************************
 * Sun Performance BLAS
 *************************************************************************************/
#ifdef _MULTILU_BLAS_SUNPERF

/*************************************************************************************
 * Function: S_UnitLowerRightTriSolve
 *
 * Description: Solves the unit-lower right triangular system.
 *
 *************************************************************************************/
BLAS_CILK void S_UnitLowerRightTriSolve(int m, int n, double *L, int ld_l, double *B, int ld_b)
{
  double dbl_one = 1.0;
  
  BLAS_SPAWN DTRSM('R','L', 'T', 'U', m, n, dbl_one, 
		   L, ld_l, B, ld_b);
  BLAS_SYNC;
}

/*************************************************************************************
 * Function: S_UnitLowerLeftTriSolve
 *
 * Description: Solves the unit-lower Left triangular system.
 *
 *************************************************************************************/
BLAS_CILK void S_UnitLowerLeftTriSolve(int m, int n, double *L, int ld_l, double *B, int ld_b)
{
  double dbl_one = 1.0;
  
  BLAS_SPAWN DTRSM('L','L', 'T', 'N', m, n, dbl_one, 
		   L, ld_l, B, ld_b);
  BLAS_SYNC;
}

/*************************************************************************************
 * Function: CeqMABT
 *
 * Description: Makes C = -A * B'. A is mxk. B is nxk. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
BLAS_CILK void S_CeqMABT(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  BLAS_SPAWN DGEMM('N', 'T', m, n, k, 
		     -1.0, A, ld_a, B, ld_b,
		     1.0, C, ld_c); 
  BLAS_SYNC;
}

/*************************************************************************************
 * Function: CaddMABT
 *
 * Description: Makes C -= A * B'. A is mxk. B is nxk. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
BLAS_CILK void S_CaddMABT(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  BLAS_SPAWN S_DGEMM('N', 'T', m, n, k, 
		     -1.0, A, ld_a, B, ld_b,
		     1.0, C, ld_c); 
  BLAS_SYNC;
}

/*************************************************************************************
 * Function: CaddMAB
 *
 * Description: Makes C -= A * B. A is mxk. B is nxk. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
BLAS_CILK void S_CaddMAB(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  BLAS_SPAWN DGEMM('N', 'N', m, n, k, 
		   -1.0, A, ld_a, B, ld_b,
		   1.0, C, ld_c); 
  BLAS_SYNC;
}

#endif

/*************************************************************************************
 * XDGEMM, cilked or not
 *************************************************************************************/
#if defined(_MULTILU_BLAS_XDGEMM) || defined(_MULTILU_BLAS_CILKED_XDGEMM)

#ifdef _MULTILU_BLAS_XDGEMM
#include "xdgemm.h"
#else
#include "xdgemmcilk.h"
#endif

/*************************************************************************************
 * Function: S_UnitLowerRightTriSolve
 *
 * Description: Solves the unit-lower right triangular system.
 *
 *************************************************************************************/
BLAS_CILK void S_UnitLowerRightTriSolve(int m, int n, double *L, int ld_l, double *B, int ld_b)
{
  Tmatrix _L, _B;
  int i;

  TAUCS_PROFILE_START(taucs_profile_multilu_xdgm_copy);

  /* Prepare matrices */
  prepare_RBR_Tmatrix(n, n, MULTILU_XDGEMM_MEM_PAD, &_L, TRUE);
  prepare_RBR_Tmatrix(m, n, MULTILU_XDGEMM_MEM_PAD, &_B, TRUE);

  /* Copy them */
  BLAS_SPAWN recursive_copy_src_NT(n, n, L, ld_l, _L);
  BLAS_SPAWN recursive_copy_src_NT(m, n, B, ld_b, _B);
  BLAS_SYNC; 

  TAUCS_PROFILE_STOP(taucs_profile_multilu_xdgm_copy);
  
  /* Set L diagonal to one */
  /* TODO: Check if we can patch the XDGEMM library to handle unit-lower */
  for(i = 0; i < n; i++)
    putAij((&_L), i, i, 1.0);

  /* Calculate */
  BLAS_SPAWN recursive_dtrsm_RLTN_O_bt(_L, _B);
  BLAS_SYNC;

  TAUCS_PROFILE_START(taucs_profile_xdgm_copy);

  /* Copy out */
  BLAS_SPAWN recursive_copy_src_NT_out(m, n, B, ld_b, _B);
  BLAS_SYNC;
  
  /* Free memory */
  free_Tmatrix(&_B);

  TAUCS_PROFILE_STOP(taucs_profile_xdgm_copy);
}

/*************************************************************************************
 * Function: S_CeqMABT
 *
 * Description: Makes C = -A * B'. A is mxk. B is nxk. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
BLAS_CILK void S_CeqMABT(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  Tmatrix _A, _B, _C;
 
  /* Zero the C matrix */
  memset(C, 0, m * n * sizeof(double));

  TAUCS_PROFILE_START(taucs_profile_xdgm_copy);

  /* Prepare matrices */
  prepare_RBR_Tmatrix(m, k, MULTILU_XDGEMM_MEM_PAD, &_A, TRUE);
  prepare_RBR_Tmatrix(n, k, MULTILU_XDGEMM_MEM_PAD, &_B, TRUE);
  prepare_RBR_Tmatrix(m, n, MULTILU_XDGEMM_MEM_PAD, &_C, TRUE);

  /* Copy them */
  BLAS_SPAWN recursive_copy_src_NT(m, k, A, ld_a, _A);
  BLAS_SPAWN recursive_copy_src_NT(n, k, B, ld_b, _B);
  BLAS_SPAWN recursive_copy_src_NT(m, n, C, ld_c, _C);
  BLAS_SYNC;

  TAUCS_PROFILE_STOP(taucs_profile_xdgm_copy);

  /* Calculate */
  BLAS_SPAWN recursive_dgemm_minus(_C, _A, _B);
  BLAS_SYNC;

  TAUCS_PROFILE_START(taucs_profile_xdgm_copy);

  /* Copy out */
  BLAS_SPAWN recursive_copy_src_NT_out(m, n, C, ld_c, _C);
  BLAS_SYNC;

  /* Free memory */
  free_Tmatrix(&_A);
  free_Tmatrix(&_B);
  free_Tmatrix(&_C);

  TAUCS_PROFILE_STOP(taucs_profile_xdgm_copy);
}

/*************************************************************************************
 * Function: S_CaddMABT
 *
 * Description: Makes C -= A * B'. A is mxk. B is nxk. Their load is given too.
 *              A load for C is passed too.
 *
 *************************************************************************************/
BLAS_CILK void S_CaddMABT(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  Tmatrix _A, _B, _C;
 
  /* MULTILU_PROFILE_START; ??? */
  

  TAUCS_PROFILE_START(taucs_profile_xdgm_copy);

  /* Prepare matrices */
  prepare_RBR_Tmatrix(m, k, MULTILU_XDGEMM_MEM_PAD, &_A, TRUE);
  prepare_RBR_Tmatrix(n, k, MULTILU_XDGEMM_MEM_PAD, &_B, TRUE);
  prepare_RBR_Tmatrix(m, n, MULTILU_XDGEMM_MEM_PAD, &_C, TRUE);

  /* Copy them */
  BLAS_SPAWN recursive_copy_src_NT(m, k, A, ld_a, _A);
  BLAS_SPAWN recursive_copy_src_NT(n, k, B, ld_b, _B);
  BLAS_SPAWN recursive_copy_src_NT(m, n, C, ld_c, _C);
  BLAS_SYNC;

  TAUCS_PROFILE_STOP(taucs_profile_xdgm_copy);

  /* Calculate */
  BLAS_SPAWN recursive_dgemm_minus(_C, _A, _B);
  BLAS_SYNC;

  TAUCS_PROFILE_START(taucs_profile_xdgm_copy);

  /* Copy out */
  BLAS_SPAWN recursive_copy_src_NT_out(m, n, C, ld_c, _C);
  BLAS_SYNC;

  /* Free memory */
  free_Tmatrix(&_A);
  free_Tmatrix(&_B);
  free_Tmatrix(&_C);

  TAUCS_PROFILE_STOP(taucs_profile_xdgm_copy);
}

#endif

/*************************************************************************************
 * NO LAPACK
 *************************************************************************************/
#ifdef _MULTILU_LAPACK_NONE

#define _MULTILU_LAPACK_ZERO_IND 1

/*************************************************************************************
 * Function: S_LU
 *
 * Description: LU factorizes A which is mxn with load lda. rows gives rows number.
 *              On output it is the new row-ordering. This is priority LUing with
 *              priorities given in prty. At each step we choose the row with the 
 *              lowest priority with abs value bigger the the max abs value.
 *              A workspace of m elements is requested.
 *
 *************************************************************************************/
int S_LU_ind(double *A, int m, int n, int lda, double thresh, int *prty, int *ipiv)
{
  /* TODO: Make this work with thresh != 1 */
  int c, r, k, ind, t;
  double val;

  for (c = 0; c < n; c++)
  {
    /* Find pivot */
    val = fabs(A[c * lda + c]); ind = c;
    for(r = c + 1; r < m; r++)
      if (fabs(A[c * lda + r]) > val)
      {
	ind = r;
	val = fabs(A[c * lda + r]);
      }
    
    /* Mark pivot and make switch */
    ipiv[c] = ind;
    for(k = 0; k < n; k++)
    {
      val = A[k * lda + c];
      A[k * lda + c] =  A[k * lda + ind];
      A[k * lda + ind] = val;
    }


    /* Scale */
    for(r = c + 1; r < m; r++)
      /*      A[c * lda + r] /= A[c * lda + c]; */
      A[r * lda + c] /= A[c * lda + c];

    /* Schur update */
    for(r = c + 1; r < m; r++)
      for (k = c + 1; k < n; k++)
	A[k * lda + r] -= (A[c * lda + r] * A[k * lda + c]);
  }

  return 0;
}

#endif

/*************************************************************************************
 * LAPACK or ATLAS for LAPACK
 *************************************************************************************/
#if defined(_MULTILU_LAPACK_LAPACK) || defined(_MULTILU_LAPACK_ATLAS) || defined(_MULTILU_LAPACK_SGIMATH) || defined(_MULTILU_LAPACK_SCS) || defined(TAUCS_ALWAYS_USES_LAPACK)

#define _MULTILU_LAPACK_ZERO_IND 1

/*************************************************************************************
 * Function: S_LU_ind
 *
 * Description: LU factorizes A which is mxn with load lda. Pivoting indexs are 
 *              returned in ipiv. This is priority LUing with priorities given in prty. 
 *              At each step we choose the row with the lowest priority with abs value 
 *              bigger the the max abs value.
 *
 *************************************************************************************/
int S_LU_ind(double *A, int m, int n, int lda, double thresh, int *prty, int *ipiv)
{
  int info;

  DGETRF(&m, &n, A, &lda, ipiv, &info);

  return info;
}

#endif

/*************************************************************************************
 * Sun Performance instead of LAPACK
 *************************************************************************************/
#ifdef _MULTILU_LAPACK_SUNPERF

#define _MULTILU_LAPACK_ZERO_IND 0

/*************************************************************************************
 * Function: S_LU_ind
 *
 * Description: LU factorizes A which is mxn with load lda. Pivoting indexs are 
 *              returned in ipiv. This is priority LUing with priorities given in prty. 
 *              At each step we choose the row with the lowest priority with abs value 
 *              bigger the the max abs value.
 *
 *************************************************************************************/
int S_LU_ind(double *A, int m, int n, int lda, double thresh, int *prty, int *ipiv)
{
  /* TODO: Make this work with thresh != 1 */
  int info;
  
  DGETRF(m, n, A, lda, ipiv, &info);

  return info;
}

#endif

/*************************************************************************************
 * S_LU based on S_LU_ind for all LAPACK
 *************************************************************************************/

/*************************************************************************************
 * Function: S_LU
 *
 * Description: LU factorizes A which is mxn with load lda. rows gives rows number.
 *              On output it is the new row-ordering. This is priority LUing with
 *              priorities given in prty. At each step we choose the row with the 
 *              lowest priority with abs value bigger the the max abs value.
 *              A workspace of m elements is requested.
 *
 *************************************************************************************/
void S_LU(double *A, int m, int n, int lda, double thresh, int *prty, int *rows, int *workspace)
{
  int i;

  S_LU_ind(A, m, n, lda, thresh, prty, workspace);
  
  /* Reform the list of rows */
  for(i = 0; i < n; i++)
  {
    int temp = rows[i];
    rows[i] = rows[workspace[i] - _MULTILU_LAPACK_ZERO_IND];
    rows[workspace[i] - _MULTILU_LAPACK_ZERO_IND] = temp;
  }
}

/*************************************************************************************
 * No parallelizing of BLAS 
 *************************************************************************************/
#ifdef _MULTILU_NO_FORCE_P

BLAS_CILK void LU(double *A, int m, int n, int lda, double thresh, int *prty, int *rows, int *workspace)
{
  TAUCS_PROFILE_START(taucs_profile_getrf);

  BLAS_SPAWN S_LU(A, m, n, lda, thresh, prty, rows, workspace);
  BLAS_SYNC;

  TAUCS_PROFILE_STOP(taucs_profile_getrf);
}

BLAS_CILK void CeqMABT(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  TAUCS_PROFILE_START(taucs_profile_gemm);

  BLAS_SPAWN S_CeqMABT(m, n, k, A, ld_a, B, ld_b, C, ld_c);
  BLAS_SYNC;

  TAUCS_PROFILE_STOP(taucs_profile_gemm);
}

BLAS_CILK void CaddMABT(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  TAUCS_PROFILE_START(taucs_profile_gemm);

  BLAS_SPAWN S_CaddMABT(m, n, k, A, ld_a, B, ld_b, C, ld_c);
  BLAS_SYNC;

  TAUCS_PROFILE_STOP(taucs_profile_gemm);
}

BLAS_CILK void CaddMAB(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  TAUCS_PROFILE_START(taucs_profile_gemm);

  BLAS_SPAWN S_CaddMAB(m, n, k, A, ld_a, B, ld_b, C, ld_c);
  BLAS_SYNC;

  TAUCS_PROFILE_STOP(taucs_profile_gemm);
}

BLAS_CILK void CaddMATB(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  TAUCS_PROFILE_START(taucs_profile_gemm);

  BLAS_SPAWN S_CaddMATB(m, n, k, A, ld_a, B, ld_b, C, ld_c);
  BLAS_SYNC;

  TAUCS_PROFILE_STOP(taucs_profile_gemm);
}


BLAS_CILK void UnitLowerRightTriSolve(int m, int n, double *L, int ld_l, double *B, int ld_b)
{
  TAUCS_PROFILE_START(taucs_profile_trsm);

  BLAS_SPAWN S_UnitLowerRightTriSolve(m, n, L, ld_l, B, ld_b);
  BLAS_SYNC;

  TAUCS_PROFILE_STOP(taucs_profile_trsm);
}

BLAS_CILK void UnitLowerLeftTriSolve(int m, int n, double *L, int ld_l, double *B, int ld_b)
{
  TAUCS_PROFILE_START(taucs_profile_trsm);

  BLAS_SPAWN S_UnitLowerLeftTriSolve(m, n, L, ld_l, B, ld_b);
  BLAS_SYNC;

  TAUCS_PROFILE_STOP(taucs_profile_trsm);
}

BLAS_CILK void UpperLeftTriSolve(int m, int n, double *L, int ld_l, double *B, int ld_b)
{
  TAUCS_PROFILE_START(taucs_profile_trsm);

  BLAS_SPAWN S_UpperLeftTriSolve(m, n, L, ld_l, B, ld_b);
  BLAS_SYNC;

  TAUCS_PROFILE_STOP(taucs_profile_trsm);
}

#endif

/*************************************************************************************
 * No parallelizing of BLAS 
 *************************************************************************************/
#ifdef _MULTILU_FORCE_P


/* CaddMABT and CaddMAB */


void S_CaddMABT_small(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  int j, i, l;
  double *Cj, *Ail, *Bjl;
  double Cij;

  Cj = C;
  for (j = 0; j < n; j++) 
  {
    for (i = 0; i < m; i++) 
    {
      Cij = *Cj;
      Ail = A + i;
      Bjl = B + j;
      for (l = 0; l < k; l++) 
      {
	Cij -= ((*Ail) * (*Bjl));
	Ail += ld_a;
	Bjl += ld_b;
      }
      *Cj = Cij;
      Cj++;
    }
    Cj = Cj + ld_c - m;     /* now Cj is at the top of column j + 1 */
  }
}

void S_CaddMAB_small(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  int j, i, l;
  double *Cj, *Ail, *Blj;
  double Cij;

  Cj = C;
  for (j = 0; j < n; j++) 
  {
    for (i = 0; i < m; i++) 
    {
      Cij = *Cj;
      Ail = A + i;
      Blj = B + j * ld_b;
      for (l = 0; l < k; l++) 
      {
	Cij -= ((*Ail) * (*Blj));
	Ail += ld_a;
	Blj++;
      }
      *Cj = Cij;
      Cj++;
    }
    Cj = Cj + ld_c - m;     /* now Cj is at the top of column j + 1 */
  }
}

cilk void CaddMABT(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  /* Very small matrices - use simple code */
  if (n <= MULTILU_THRESHOLD_GEMM_SMALL && k <= MULTILU_THRESHOLD_GEMM_SMALL) 
  {
    S_CaddMABT_small(m, n, k, A, ld_a, B, ld_b, C, ld_c);
    return;
  }

  /* Small matrices - use BLAS */
  if (n <= MULTILU_THRESHOLD_GEMM_BLAS && k <= MULTILU_THRESHOLD_GEMM_BLAS) 
  {
    BLAS_SPAWN S_CaddMABT(m, n, k, A, ld_a, B, ld_b, C, ld_c);
    BLAS_SYNC;
    return;
  }

  if (k >= n && k >= m) 
  {
    int khalf1 = k / 2;
    int khalf2 = k - khalf1;

    spawn CaddMABT(m, n, khalf1, A, ld_a, B, ld_b, C, ld_c);
    sync;

    spawn CaddMABT(m, n, khalf2, A + khalf1 * ld_a, ld_a, B + khalf1 * ld_b, ld_b, C, ld_c);
    sync;

    return;
  } 

  if (n >= k && n >= m) 
  {
    int nhalf1 = n / 2;
    int nhalf2 = n - nhalf1;
    
    spawn CaddMABT(m, nhalf1, k, A, ld_a, B, ld_b, C, ld_c);
    spawn CaddMABT(m, nhalf2, k, A, ld_a, B + nhalf1, ld_b, C + nhalf1 * ld_c, ld_c);
    sync;

    return;
  }

  /* The following condition must be true so we put 1 */
  if (1 /* m >= k && m >= n*/) 
  {
    int mhalf1 = m / 2;
    int mhalf2 = m - mhalf1;

    spawn CaddMABT(mhalf1, n, k, A, ld_a, B, ld_b, C, ld_c);
    spawn CaddMABT(mhalf2, n, k, A + mhalf1, ld_a, B, ld_b, C + mhalf1, ld_c);
    sync;

    return;
  }

  assert(0);
}


cilk void CaddMAB(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  /* Very small matrices - use simple code */
  if (n <= MULTILU_THRESHOLD_GEMM_SMALL && k <= MULTILU_THRESHOLD_GEMM_SMALL) 
  {
    S_CaddMAB_small(m,n,k, A, ld_a, B, ld_b, C, ld_c);
    return;
  }

  /* Small matrices - use BLAS */
  if (n <= MULTILU_THRESHOLD_GEMM_BLAS && k <= MULTILU_THRESHOLD_GEMM_BLAS) 
  {
    BLAS_SPAWN S_CaddMAB(m, n, k, A, ld_a, B, ld_b, C, ld_c);
    BLAS_SYNC;
    return;
  }

  if (k >= n && k >= m) 
  {
    int khalf1 = k / 2;
    int khalf2 = k - khalf1;

    spawn CaddMAB(m, n, khalf1, A, ld_a, B, ld_b, C, ld_c);
    sync;

    spawn CaddMAB(m, n, khalf2, A + khalf1 * ld_a, ld_a, B + khalf1, ld_b, C, ld_c);
    sync;

    return;
  } 

  if (n >= k && n >= m) 
  {
    int nhalf1 = n / 2;
    int nhalf2 = n - nhalf1;
    
    spawn CaddMAB(m, nhalf1, k, A, ld_a, B, ld_b, C, ld_c);
    spawn CaddMAB(m, nhalf2, k, A, ld_a, B + nhalf1 * ld_b, ld_b, C + nhalf1 * ld_c, ld_c);
    sync;

    return;
  }

  /* The following condition must be true so we put 1 */
  if (1 /* m >= k && m >= n*/) 
  {
    int mhalf1 = m / 2;
    int mhalf2 = m - mhalf1;

    spawn CaddMAB(mhalf1, n, k, A, ld_a, B, ld_b, C, ld_c);
    spawn CaddMAB(mhalf2, n, k, A + mhalf1, ld_a, B, ld_b, C + mhalf1, ld_c);
    sync;

    return;
  }

  assert(0);
}

cilk void CeqMABT(int m, int n, int k, double *A, int ld_a, double *B, int ld_b, double *C, int ld_c)
{
  TAUCS_PROFILE_START(taucs_profile_gemm);

  memset(C, 0, ld_c * n * sizeof(double));
  if (m > MULTILU_THRESHOLD_GEMM_FORCEP && n > MULTILU_THRESHOLD_GEMM_FORCEP && k > MULTILU_THRESHOLD_GEMM_FORCEP)
  {
    spawn CaddMABT(m, n, k, A, ld_a, B, ld_b, C, ld_c);
    sync;
  }
  else
    S_CaddMABT(m, n, k, A, ld_a, B, ld_b, C, ld_c);

  TAUCS_PROFILE_STOP(taucs_profile_gemm);
}

/* UnitLowerRightTriSolve and UnitLowerLeftTriSolve */


void UnitLowerRightTriSolve_small(int m, int n, double *L, int ld_l, double *B, int ld_b)
{
  int j, i, k;
  double *Bi, *Bik, *Ljk;
  double Bij;

  Bi = B;
  for (i = 0; i < m; i++) 
  {
    for (j = 0; j< n; j++) 
    {
      Bij = *(Bi + j * ld_b);
      Bik = Bi;
      Ljk = L + j;
      for (k = 0; k < j; k++) 
      {
	Bij -= ((*Bik) * (*Ljk)); 
	Bik += ld_b;
	Ljk += ld_l;
      }
      *(Bi + j * ld_b) = Bij;
    }
    Bi = Bi + 1;
  }
}
cilk void UnitLowerLeftTriSolve(int m, int n, double *L, int ld_l, double *B, int ld_b)
{
  /* Very small matrices - use simple code */
  /*if (m <= MULTILU_THRESHOLD_TRSM_SMALL && n <= MULTILU_THRESHOLD_TRSM_SMALL) 
  {
    UnitLowerLeftTriSolve_small(m, n, L, ld_l, B, ld_b);
    return;
    }*/

  /* Small matrices - use BLAS */
  if (m <= MULTILU_THRESHOLD_TRSM_BLAS && n <= MULTILU_THRESHOLD_TRSM_BLAS) 
  {
    BLAS_SPAWN S_UnitLowerLeftTriSolve(m, n, L, ld_l, B, ld_b);
    BLAS_SYNC;
    return;
  }

  /* Case A: n >= m */
  if (n >= m) 
  {
    int nhalf1 = n / 2;
    int nhalf2 = n - nhalf1;

    spawn UnitLowerLeftTriSolve(m, nhalf1, L, ld_l, B, ld_b);
    spawn UnitLowerLeftTriSolve(m, nhalf2, L, ld_l, B + nhalf1 * ld_b, ld_b);
    sync;
    return;
  } 
  else
    /* Case B - m > n */
  {
    int mhalf1 = m / 2;
    int mhalf2 = m - mhalf1;
    
    spawn UnitLowerLeftTriSolve(mhalf1, n, L, ld_l, B, ld_b);
    sync;

    spawn CaddMAB(mhalf2, n, mhalf1, L + mhalf1, ld_l, B, ld_b, B + mhalf1, ld_b);
    sync;

    spawn UnitLowerLeftTriSolve(mhalf2, n, L + mhalf1 * (ld_l + 1), ld_l, B + mhalf1, ld_b);
    sync;

    return;
  }
}

cilk void NP_UnitLowerRightTriSolve(int m, int n, double *L, int ld_l, double *B, int ld_b)
{
  /* Very small matrices - use simple code */
  if (m <= MULTILU_THRESHOLD_TRSM_SMALL && n <= MULTILU_THRESHOLD_TRSM_SMALL) 
  {
    UnitLowerRightTriSolve_small(m, n, L, ld_l, B, ld_b);
    return;
  }

  /* Small matrices - use BLAS */
  if (m <= MULTILU_THRESHOLD_TRSM_BLAS && n <= MULTILU_THRESHOLD_TRSM_BLAS) 
  {
    BLAS_SPAWN S_UnitLowerRightTriSolve(m, n, L, ld_l, B, ld_b);
    BLAS_SYNC;
    return;
  }

  /* Case A: m >= n */
  if (m >= n) 
  {
    int mhalf1 = m / 2;
    int mhalf2 = m - mhalf1;

    spawn NP_UnitLowerRightTriSolve(mhalf1, n, L, ld_l, B, ld_b);
    spawn NP_UnitLowerRightTriSolve(mhalf2, n, L, ld_l, B + mhalf1, ld_b);
    sync;
    return;
  } 
  else
    /* Case B - m < n */
  {
    int nhalf1 = n / 2;
    int nhalf2 = n - nhalf1;
    
    spawn NP_UnitLowerRightTriSolve(m, nhalf1, L, ld_l, B, ld_b);
    sync;


    spawn CaddMABT(m, nhalf2, nhalf1, B, ld_b, L + nhalf1, ld_l, B + nhalf1 * ld_b, ld_b);
    sync;

    spawn NP_UnitLowerRightTriSolve(m, nhalf2, L + nhalf1 * (ld_l + 1), ld_l, B + nhalf1 * ld_b, ld_b);
    sync;

    return;
  }
}

cilk void UnitLowerRightTriSolve(int m, int n, double *L, int ld_l, double *B, int ld_b)
{
  TAUCS_PROFILE_START(taucs_profile_trsm);

  spawn NP_UnitLowerRightTriSolve(m, n, L, ld_l, B, ld_b);
  sync;

  TAUCS_PROFILE_STOP(taucs_profile_trsm);
}

/* Swap Lines */

/*************************************************************************************
 * Function: S_SwapLines
 *
 * Description: Swap lines for A when spaws are given in ipiv's indexes k1:k2-1.
 *              k1 and k2 in are C indexes (zero based)
 *
 *************************************************************************************/
void S_SwapLines(double *A, int n, int lda, int *ipiv, int k1, int k2)
{
  int i, j, t;
  double v;

  for (j = 0; j < n; j++)
    for (i = k1; i < k2; i++)
    {
      t = ipiv[i] - _MULTILU_LAPACK_ZERO_IND;
      v = A[j * lda + i];
      A[j * lda + i] = A[j * lda + t];
      A[j * lda + t] = v;
    }
}

cilk void SwapLines(double *A, int n, int lda, int *ipiv, int k1, int k2)
{
  int nhalf1, nhalf2;

  /* Small matrices - use LAPACK */
  if (n <= MULTILU_THRESHOLD_LASWP_LAPACK)
  {
    S_SwapLines(A, n, lda, ipiv, k1, k2);
    return;
  }

  /* Break matrix into two and run in parallel */
  nhalf1 = n / 2;
  nhalf2 = n - nhalf1;

  spawn SwapLines(A, nhalf1, lda, ipiv, k1, k2);
  spawn SwapLines(A + nhalf1 * lda, nhalf2, lda, ipiv, k1, k2);
  sync;
}

/* LU */
cilk int LU_ind(double *A, int m, int n, int lda, double thresh, int *prty, int *ipiv)
{
  int i, r, nhalf1, nhalf2;

  /* Very small matrix - use simple code */

  /* Small matrix - use BLAS */
  if (n <= MULTILU_THRESHOLD_GETRF_LAPACK)
  {
    r = BLAS_SPAWN S_LU_ind(A, m, n, lda, thresh, prty, ipiv);
    BLAS_SYNC;
    return r;
  }

  nhalf1 = n / 2;
  nhalf2 = n - nhalf1;

  /* Call on first half */
  r = spawn LU_ind(A, m, nhalf1, lda, thresh, prty, ipiv);
  sync;

  if (r != 0)
    return r;
    
  if (nhalf2 > 0)
  {
    /* Make swaps for rest of matrix */
    spawn SwapLines(A + (nhalf1 * lda), nhalf2, lda, ipiv, 0, nhalf1);
    sync;

    /* Find U part */
    spawn UnitLowerLeftTriSolve(nhalf1, nhalf2, A, lda, A + (nhalf1 * lda), lda);
    sync;

    /* Schur complement the rest and update it */
    if (m > nhalf1)
    {
      int mrest = m - nhalf1;
      int mn = min(m, n);
      
      spawn CaddMAB(mrest, nhalf2, nhalf1, A + nhalf1, lda, A + nhalf1 * lda, lda, A + nhalf1 * (lda + 1), lda);
      sync;

      r = spawn LU_ind(A + nhalf1 * (lda + 1), mrest, nhalf2, lda, thresh, prty + nhalf1, ipiv + nhalf1);
      sync;
      
      /* Increase ipivs numbers */
      for(i = nhalf1; i < mn; i++)
	ipiv[i] += nhalf1;
      
      /* Make swaps for this part of the matrix */
      spawn SwapLines(A, nhalf1, lda, ipiv, nhalf1, mn);
      sync;
    }
  }

  return r;
}

cilk void LU(double *A, int m, int n, int lda, double thresh, int *prty, int *rows, int *workspace)
{
  int i, t;

  TAUCS_PROFILE_START(taucs_profile_getrf);

  t = spawn LU_ind(A, m, n, lda, thresh, prty, workspace);
  sync;

  /* Reform the list of rows */
  for(i = 0; i < n; i++)
  {
    int temp = rows[i];
    rows[i] = rows[workspace[i] - _MULTILU_LAPACK_ZERO_IND];
    rows[workspace[i] - _MULTILU_LAPACK_ZERO_IND] = temp;
  }

  TAUCS_PROFILE_STOP(taucs_profile_getrf);
}

#endif


