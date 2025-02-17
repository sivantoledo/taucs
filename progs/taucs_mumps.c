/*********************************************************/
/* TAUCS                                                 */
/* Author: Sivan Toledo                                  */
/*********************************************************/
/* 

TAUCS_CONFIG_BEGIN
TAUCS_CONFIG_DEFAULT OFF
TAUCS_CONFIG TIMING
TAUCS_CONFIG BASE
TAUCS_CONFIG DREAL
TAUCS_CONFIG METIS
TAUCS_CONFIG MALLOC_STUBS
TAUCS_CONFIG MATRIX_GENERATORS
TAUCS_CONFIG MATRIX_IO
TAUCS_CONFIG AD_HOC_TEST
TAUCS_CONFIG_END

*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "taucs.h"

/* this only works for generic complex */
#define taucs_im(x)    ((x).i)
#define taucs_re(x)    ((x).r)

#define TRUE  1
#define FALSE 0

/*********************************************************/
/* argument parsing                                      */
/*********************************************************/

int taucs_getopt_boolean(char* cmd, void* args[], char* name, int* x) {
  int lc = strlen(cmd);
  int ln = strlen(name);
  if (!strncmp(cmd,name,ln)) {
    if (lc > ln && cmd[ln] == '.') return 0;
    if (lc > ln && cmd[ln] == '=') {
      if (cmd[ln+1] == '#') {
	unsigned int p;
	if (sscanf(cmd+ln+2,"%u",&p) == 1) {
	  unsigned int i;
	  for (i=0; args[i]; i++) {
	    if (i==p) { *x = *( (int*) args[i] ); return 1; }
	  }
	  taucs_printf("taucs: WARNING, pointer argument out of range in [%s]\n",cmd);
	}
	taucs_printf("taucs: WARNING, illegal pointer argument in [%s]\n",cmd);
	return 0;
      }
      if (!strcmp(cmd+ln+1,"true")) {
	*x = TRUE;
	return 1;
      }
      if (!strcmp(cmd+ln+1,"false")) {
	*x = FALSE;
	return 1;
      }
    }
    taucs_printf("taucs: WARNING, illegal argument in [%s]\n",cmd);
  }

  return 0;
}

int taucs_getopt_double(char* cmd, void* args[], char* name, double* x) {
  int lc = strlen(cmd);
  int ln = strlen(name);
  if (!strncmp(cmd,name,ln)) {
    if (lc > ln && cmd[ln] == '.') return 0;
    if (lc > ln && cmd[ln] == '=') {
      if (cmd[ln+1] == '#') {
	unsigned int p;
	if (sscanf(cmd+ln+2,"%u",&p) == 1) {
	  unsigned int i;
	  for (i=0; args[i]; i++) {
	    if (i==p) { *x = *( (double*) args[i] ); return 1; }
	  }
	  taucs_printf("taucs: WARNING, pointer argument out of range in [%s]\n",cmd);
	}
	taucs_printf("taucs: WARNING, illegal pointer argument in [%s]\n",cmd);
	return 0;
      }
      if (sscanf(cmd+ln+1,"%le",x) == 1) {
	return 1;
      }
    }
    taucs_printf("taucs: WARNING, illegal argument in [%s]\n",cmd);
  }

  return 0;
}


int taucs_getopt_pointer(char* cmd, void* args[], char* name, void** x) {
  int lc = strlen(cmd);
  int ln = strlen(name);
  if (!strncmp(cmd,name,ln)) {
    if (lc > ln && cmd[ln] == '.') return 0;
    if (lc > ln && cmd[ln] == '=') {
      if (cmd[ln+1] == '#') {
	unsigned int p;
	if (sscanf(cmd+ln+2,"%u",&p) == 1) {
	  unsigned int i;
	  for (i=0; args[i]; i++)
	    if (i==p) { *x = *( (void**) args[i] ); return 1; }
	  taucs_printf("taucs: WARNING, pointer argument out of range in [%s]\n",cmd);
	}
	taucs_printf("taucs: WARNING, illegal pointer argument in [%s]\n",cmd);
	return 0;
      }
    }
    taucs_printf("taucs: WARNING, illegal argument in [%s]\n",cmd);
  }

  return 0;
}

int taucs_getopt_string(char* cmd, void* args[], char* name, char** x) {
  int lc = strlen(cmd);
  int ln = strlen(name);
  if (!strncmp(cmd,name,ln)) {
    if (lc > ln && cmd[ln] == '.') return 0;
    if (lc > ln && cmd[ln] == '=') {
      if (cmd[ln+1] == '#') {
	unsigned int p;
	if (sscanf(cmd+ln+2,"%u",&p) == 1) {
	  unsigned int i;
	  for (i=0; args[i]; i++)
	    if (i==p) { *x = *( (char**) args[i] ); return 1; }
	  taucs_printf("taucs: WARNING, pointer argument out of range in [%s]\n",cmd);
	}
	taucs_printf("taucs: WARNING, illegal pointer argument in [%s]\n",cmd);
	return 0;
      }
      *x = cmd+ln+1;
      return 1;
    }
    taucs_printf("taucs: WARNING, illegal argument in [%s]\n",cmd);
  }

  return 0;
}
void __ctype_b() {} /* just a hack to get it to link */
void s_stop() {} /* just a hack to get it to link */

void rnorm(taucs_ccs_matrix* A, void* x, void* b, void* aux)
{
  int i;
  int one = 1;
  double relerr;
  
  taucs_ccs_times_vec(A,x,aux);

  taucs_vec_axpby(A->n,A->flags,1.0,aux,-1.0,b,aux);

  relerr = taucs_vec_norm2(A->n,A->flags,aux) 
           / taucs_vec_norm2(A->n,A->flags,b);


  taucs_printf("relative 2-norm of the residual %.2e \n",relerr);
}

#include "dmumps_c.h"
#define ICNTL(I) icntl[(I)-1]
#define RINFO(I) rinfo[(I)-1]
#define INFO(I)  info[(I)-1] 

void call_mumps(taucs_ccs_matrix* A, double* X, double* B,
		int spd, int ind)
{
  DMUMPS_STRUC_C id;
  int i,j,nnz,nz,n,ip;
  double tw,tc;

  if (spd && !ind) id.sym=1; /* symmetric positive definite */
  else if (!spd && ind) id.sym=2; /* symmetric indefinite */
  else {
    taucs_printf("taucs_mumps: you must specify whether A is spd or ind (use taucs_run.mumps[spd/ind]\n");
    exit(1);
  }

  id.job = -1; /* init library */
  id.par = 1;  /* this processor participates */
  /*id.sym = 1;*/  /* symmetric positive definite */
  id.comm_fortran = -987654; /* use comm_world */
  dmumps_c(&id);

  id.job = 6; /* analyze=factor+solve */
  id.job = 1; /* analyze */

  n = A->n;
  nnz = (A->colptr)[ n ];

  id.n = A->n; 
  id.nz = nnz; 
  id.irn= (int*) malloc(nnz*sizeof(int)); 
  id.jcn= (int*) malloc(nnz*sizeof(int)); 
  assert(id.irn && id.jcn);
  nz = 0;
  for (j=0; j<n; j++) {
    for (ip=(A->colptr)[j]; ip<(A->colptr)[j+1]; ip++) {
      i = (A->rowind)[ip];
      (id.irn)[nz] = i+1;
      (id.jcn)[nz] = j+1;
      nz++;
    }
  }
  id.a = A->values.d; 
  for (i=0; i<n; i++) X[i] = B[i]; /* mumps overwrites the RHS */
  id.rhs = X;

  /* No outputs */
  /*id.ICNTL(1)=-1; id.ICNTL(2)=-1; id.ICNTL(3)=-1; id.ICNTL(4)=0;*/
  id.ICNTL(5) = 0; /* default, assembled */
  id.ICNTL(6) = 0; /* default for symmetric, no column permutation */
  id.ICNTL(7) = 5; /* metis metis_nodend */
  id.ICNTL(8) = 0; /* default for symmetric ,no scaling */

  tw = taucs_wtime();
  tc = taucs_ctime();
  dmumps_c(&id);
  taucs_printf("mumps analyze time: %.02e seconds (%.02e seconds CPU time)\n",taucs_wtime()-tw,taucs_ctime()-tc);
  

  taucs_printf("mumps analyze info: analyze outcome = %s\n",id.INFO(1)==0?"success":"failure");
  taucs_printf("mumps analyze info: estimated     flops %.02e\n",id.RINFO(1));

  id.job = 2; /* factor */

  tw = taucs_wtime();
  tc = taucs_ctime();
  dmumps_c(&id);
  taucs_printf("mumps factor time: %.02e seconds (%.02e seconds CPU time)\n",taucs_wtime()-tw,taucs_ctime()-tc);

  taucs_printf("mumps factor info: factor outcome = %s\n",id.INFO(1)==0?"success":"failure");
  taucs_printf("mumps factor info: assembly      flops %.02e\n",id.RINFO(2));
  taucs_printf("mumps factor info: factorization flops %.02e\n",id.RINFO(3));

  tw = taucs_wtime();
  tc = taucs_ctime();
  id.job = 3; /* solve */
  dmumps_c(&id);
  taucs_printf("mumps solve time: %.02e seconds (%.02e seconds CPU time)\n",taucs_wtime()-tw,taucs_ctime()-tc);

  taucs_printf("mumps solve info: solve outcome = %s\n",id.INFO(1)==0?"success":"failure");

  id.job = -2; /* release library */
  dmumps_c(&id);
}

int main(int argc, char* argv[])
{
  int rc;
  int i;

  taucs_ccs_matrix*  A = NULL;
  void*      X;
  void*      B;
  void*      Y;
  void*      Z;
  void* opt_arg[] = { NULL };

  char* opt_ijv = NULL;
  char* opt_hb  = NULL;
  char* opt_log = "stdout";
  double opt_3d = -1.0;
  double opt_2d = -1.0;
  double opt_dense = -1.0;
  char*  opt_2d_type = "dirichlet";
  int opt_3d_rand = 0;/*not random*/
  double opt_3d_small = -1.0;/*regular*/

  int opt_mumps_spd = 0;
  int opt_mumps_ind = 0;

  int opt_sreal    = 0;
  int opt_dreal    = 0;
  int opt_scomplex = 0;
  int opt_dcomplex = 0;
  int datatype     = TAUCS_DOUBLE;

  /*
  {
    void ATL_buildinfo(void);
    ATL_buildinfo();
  }
  */ 

  for (i=0; argv[i]; i++) {
    int understood = FALSE;
    
    understood |= taucs_getopt_boolean(argv[i],opt_arg,"taucs_run.mumps.spd",&opt_mumps_spd);
    understood |= taucs_getopt_boolean(argv[i],opt_arg,"taucs_run.mumps.ind",&opt_mumps_ind);
    understood |= taucs_getopt_boolean(argv[i],opt_arg,"taucs_run.sreal",&opt_sreal);
    understood |= taucs_getopt_boolean(argv[i],opt_arg,"taucs_run.dreal",&opt_dreal);
    understood |= taucs_getopt_boolean(argv[i],opt_arg,"taucs_run.scomplex",&opt_scomplex);
    understood |= taucs_getopt_boolean(argv[i],opt_arg,"taucs_run.dcomplex",&opt_dcomplex);

    understood |= taucs_getopt_string(argv[i],opt_arg,"taucs_run.ijv",&opt_ijv);
    understood |= taucs_getopt_string(argv[i],opt_arg,"taucs_run.hb", &opt_hb );
    understood |= taucs_getopt_string(argv[i],opt_arg,"taucs_run.log",&opt_log);
    understood |= taucs_getopt_double(argv[i],opt_arg,"taucs_run.dense",&opt_dense);
    understood |= taucs_getopt_double(argv[i],opt_arg,"taucs_run.mesh3d",&opt_3d);
    understood |= taucs_getopt_boolean(argv[i],opt_arg,"taucs_run.mesh3d.rand",&opt_3d_rand);
    understood |= taucs_getopt_double(argv[i],opt_arg,"taucs_run.mesh3d.small",&opt_3d_small);
    understood |= taucs_getopt_double(argv[i],opt_arg,"taucs_run.mesh2d",&opt_2d);
    understood |= taucs_getopt_string(argv[i],opt_arg,"taucs_run.mesh2d.type",&opt_2d_type);
    
    if (!understood) taucs_printf("taucs_run: illegal option [%s]\n",
				  argv[i]);
  }

  if (opt_sreal   ) datatype = TAUCS_SINGLE;
  if (opt_dreal   ) datatype = TAUCS_DOUBLE;
  if (opt_scomplex) datatype = TAUCS_SCOMPLEX;
  if (opt_dcomplex) datatype = TAUCS_DCOMPLEX;

  taucs_logfile(opt_log);

  if (opt_dense > 0) {
    A = taucs_ccs_generate_dense((int)opt_dense,(int)opt_dense,TAUCS_SYMMETRIC);
    if (!A) {
      taucs_printf("Matrix generation failed\n");
      return 1;
    }
    datatype = TAUCS_DOUBLE;
  }

  if (opt_3d > 0) {
    if ( opt_3d_small > 0 ) {
        A = taucs_ccs_generate_mesh3d_random((int)opt_3d_small,(int)opt_3d,(int)opt_3d,opt_3d_rand);
    }
    A = taucs_ccs_generate_mesh3d_random((int)opt_3d,(int)opt_3d,(int)opt_3d,opt_3d_rand);
    if (!A) {
      taucs_printf("Matrix generation failed\n");
      return 1;
    }
    datatype = TAUCS_DOUBLE;
  }

  if (opt_2d > 0) {
    A = taucs_ccs_generate_mesh2d((int)opt_2d,opt_2d_type);
    if (!A) {
      taucs_printf("Matrix generation failed\n");
      return 1;
    }
    datatype = TAUCS_DOUBLE;
  }


  if (opt_ijv) {
    switch (datatype) {
    case TAUCS_SINGLE:
      A = taucs_ccs_read_ijv (opt_ijv,TAUCS_SYMMETRIC | TAUCS_SINGLE); break;
      break;
    case TAUCS_DOUBLE:
      A = taucs_ccs_read_ijv (opt_ijv,TAUCS_SYMMETRIC | TAUCS_DOUBLE); break;
      break;
    case TAUCS_SCOMPLEX:
      A = taucs_ccs_read_ijv (opt_ijv,TAUCS_HERMITIAN | TAUCS_SCOMPLEX); break;
      break;
    case TAUCS_DCOMPLEX:
      A = taucs_ccs_read_ijv (opt_ijv,TAUCS_HERMITIAN | TAUCS_DCOMPLEX); break;
      break;
    default:
      taucs_printf("taucs_run: incorrect datatype\n");
      return 1;
      break;
    }      
  }

  if (opt_hb) {
    switch (datatype) {
    case TAUCS_SINGLE:
      A = taucs_ccs_read_hb (opt_hb, TAUCS_SINGLE); break;
      break;
    case TAUCS_DOUBLE:
      A = taucs_ccs_read_hb (opt_hb, TAUCS_DOUBLE); break;
      break;
    case TAUCS_SCOMPLEX:
      A = taucs_ccs_read_hb (opt_hb, TAUCS_SCOMPLEX); break;
      break;
    case TAUCS_DCOMPLEX:
      A = taucs_ccs_read_hb (opt_hb, TAUCS_DCOMPLEX); break;
      break;
    default:
      taucs_printf("taucs_run: incorrect datatype\n");
      return 1;
      break;
    }
    datatype = A->flags;
  }

  if (!A) {
    taucs_printf("taucs_run: there is no matrix!\n");
    return 1;
  }

  taucs_printf("taucs_run: matrix dimentions %d by %d with %d nonzeros\n",
	       A->m, A->n, (A->colptr)[ A->n ]);

  X = taucs_vec_create(A->n,A->flags);
  B = taucs_vec_create(A->n,A->flags);
  Y = taucs_vec_create(A->n,A->flags);
  Z = taucs_vec_create(A->n,A->flags);
  if (!X || !B || !Y || !Z) {
    taucs_printf("taucs_run: vector allocation failed\n");
    return 1;
  }

  for(i=0; i<A->n; i++) {
    if (datatype & TAUCS_SINGLE) 
      ((taucs_single*)X)[i]=(taucs_single) ((double)random()/(double)RAND_MAX);
    if (datatype & TAUCS_DOUBLE) 
      ((taucs_double*)X)[i]=(taucs_double) ((double)random()/(double)RAND_MAX);
#if 0
    if (datatype & TAUCS_SCOMPLEX) {
      taucs_re(((taucs_scomplex*)X)[i])=(taucs_single) ((double)random()/(double)RAND_MAX);
      taucs_im(((taucs_scomplex*)X)[i])=(taucs_single) ((double)random()/(double)RAND_MAX);
    }
    if (datatype & TAUCS_DCOMPLEX) {
      taucs_re(((taucs_dcomplex*)X)[i])=(taucs_double) ((double)random()/(double)RAND_MAX);
      taucs_im(((taucs_dcomplex*)X)[i])=(taucs_double) ((double)random()/(double)RAND_MAX);
    }
#endif
  }

  taucs_ccs_times_vec(A,X,B);
  
  /*rc = taucs_linsolve(A,NULL,1,Y,B,argv,opt_arg);*/

  call_mumps(A,Y,B,opt_mumps_spd,opt_mumps_ind);

  rnorm(A,Y,B,Z);

  return 0;
}
