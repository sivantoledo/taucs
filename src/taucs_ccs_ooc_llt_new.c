
/*
 * taucs_ccs_ooc_ldlt.c 
 *
 * Out-of-core sparse Indefinite factorization
 *
 * authors: Omer Meshar & Sivan Toledo 
 *
 * Copyright, 2003.
 */

/*************************************************************/
/*                                                           */
/*************************************************************/

#include <stdio.h>
#include <stdlib.h>


/*#include <unistd.h>*/
/*#include <sys/uio.h>*/

#include <assert.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "taucs.h"

#define FALSE 0
#define TRUE  1

/* #define BLAS_FLOPS_CUTOFF  1000.0 */

#define BLAS_FLOPS_CUTOFF  -1.0
#define SOLVE_DENSE_CUTOFF 5

/* number of matrices in the header of the file */
/*#define IO_BASE    6*/
#define IO_BASE    7
/* multiple files of at most 1 GB or single file */
#define MULTIFILE 0

/*************************************************************/
/* structures                                                */
/*************************************************************/

typedef struct {
  char    uplo;     /* 'u' for upper, 'l' for lower, ' ' don't know; prefer lower. */
  int     n;        /* size of matrix */
  int     n_sn;     /* number of supernodes */

  int* parent;      /* supernodal elimination tree */
  int* first_child; 
  int* next_child;
  int* ipostorder;
  int* col_to_sn_map;     

  int* sn_size;     /* size of supernodes (diagonal block) */
  int* sn_up_size;  /* size of subdiagonal update blocks   */
  int** sn_struct;  /* row structure of supernodes         */

  taucs_datatype** sn_blocks; /* supernode blocks        */
  taucs_datatype** up_blocks; /* update blocks           */
} supernodal_factor_matrix_ooc_ldlt;

typedef struct {
	int n_pn;					/* number of the panel */
	int n_sn;					/* number of supernodes */
	int n_postorder;	/* current number in postorder */
	int used;					/* was there a panel that was used */
	
	int max_size;	/* max size of supernode in panel */
	
	int* sn_to_panel_map; 
										/* maps each supernode to its panel */
	int* parent;			/* parent tree */
	int* sn_in_core;	/* sn in core */
	int* postorder;		/* post order of supernodes */
	
	double avail_mem; /* available memory in panel */
	double max_mem;		/* max memory allowed in panel */
} supernodal_panel;

/*************************************************************/
/* for qsort                                                 */
/*************************************************************/

static int compare_ints(void* vx, void* vy)
{
  int* ix = (int*)vx;
  int* iy = (int*)vy;
  if (*ix < *iy) return -1;
  if (*ix > *iy) return  1;
  return 0;
}

static int* compare_indirect_map;
static int compare_indirect_ints(const void* vx, const void* vy)/*(void* vx, void* vy) omer*/
{
  int* ix = (int*)vx;
  int* iy = (int*)vy;
  if (compare_indirect_map[*ix] < compare_indirect_map[*iy]) return -1;
  if (compare_indirect_map[*ix] > compare_indirect_map[*iy]) return  1;
  return 0;
}

#ifndef TAUCS_CORE_GENERAL

/*************************************************************/
/* create and free the panel object                          */
/*************************************************************/
static supernodal_panel* 
taucs_panel_create( double memory, int n_sn, int* sn_parent )
{
	int i;
	supernodal_panel* P;

	P = (supernodal_panel*) taucs_malloc(sizeof(supernodal_panel));
	if (!P) return NULL;

	P->max_mem = memory;
	P->avail_mem = memory;
	P->max_size = 0;
	P->n_sn = 0;
	P->n_pn = 0;
	P->n_postorder = 0;
	P->used = 0;
	P->sn_to_panel_map = taucs_malloc(sizeof(int)*(n_sn+1));
	if (!P->sn_to_panel_map) {
		taucs_free(P);
		return NULL;
	}
	P->parent = taucs_malloc(sizeof(int)*(n_sn+1));
	if (!P->parent) {
		taucs_free(P);
		return NULL;
	}
	P->sn_in_core = taucs_malloc(sizeof(int)*(n_sn+1));
	if (!P->sn_in_core) {
		taucs_free(P);
		return NULL;
	}
	P->postorder = taucs_malloc(sizeof(int)*(n_sn+1));
	if (!P->postorder) {
		taucs_free(P);
		return NULL;
	}
	for (i=0; i<n_sn+1; i++) {
		P->sn_in_core[i] = 0;
		P->sn_to_panel_map[i] = -1;
		P->postorder[i] = -1;
		if ( sn_parent[i] )
			P->parent[i] = sn_parent[i];
		else
			P->parent[i] = -1;
	}

	return P;
}

static void
taucs_panel_free( supernodal_panel* P )
{
	taucs_free( P->sn_to_panel_map );
	taucs_free( P->parent );
	taucs_free( P->sn_in_core );
	taucs_free( P->postorder );
	taucs_free( P );
}


/*************************************************************/
/* create and free the factor object                         */
/*************************************************************/

static supernodal_factor_matrix_ooc_ldlt*
multifrontal_supernodal_create()
{
  supernodal_factor_matrix_ooc_ldlt* L;
  
  L = (supernodal_factor_matrix_ooc_ldlt*) taucs_malloc(sizeof(supernodal_factor_matrix_ooc_ldlt));
  if (!L) return NULL;
  L->uplo      = 'l';
  L->n         = -1; /* unused */

  L->sn_struct   = NULL;
  L->sn_size     = NULL;
  L->sn_up_size  = NULL;
  L->parent      = NULL;
  L->col_to_sn_map = NULL;
  L->first_child = NULL;
  L->next_child  = NULL;
  L->ipostorder  = NULL;
  L->sn_blocks     = NULL;
  L->up_blocks     = NULL;

  return L;
}

static void
ooc_supernodal_free(supernodal_factor_matrix_ooc_ldlt* L,
										int sn)
{
	taucs_free(L->sn_struct[sn]);
  taucs_free(L->sn_blocks[sn]);
  taucs_free(L->up_blocks[sn]);
	L->sn_struct[sn] = NULL;
	L->sn_blocks[sn] = NULL;
	L->up_blocks[sn] = NULL;
}

static void 
ooc_supernodal_factor_free(void* vL)
{
  supernodal_factor_matrix_ooc_ldlt* L = (supernodal_factor_matrix_ooc_ldlt*) vL;
  int sn;
  
  taucs_free(L->parent);
  taucs_free(L->first_child);
  taucs_free(L->next_child);
  taucs_free(L->col_to_sn_map);

  taucs_free(L->sn_size);
  taucs_free(L->sn_up_size);
  for (sn=0; sn<L->n_sn; sn++) {
    ooc_supernodal_free(L,sn);
  }

  taucs_free(L->sn_struct);
  taucs_free(L->sn_blocks);
  taucs_free(L->up_blocks);

  taucs_free(L);
}

static void
recursive_symbolic_elimination(int            j,
			       taucs_ccs_matrix* A,
			       int            first_child[],
			       int            next_child[],
			       int*           n_sn,
			       int            sn_size[],
			       int            sn_up_size[],
			       int*           sn_rowind[],
			       int            sn_first_child[], 
			       int            sn_next_child[], 
						 int						sn_parent[],
			       int            rowind[],
			       int            column_to_sn_map[],
			       int            map[],
			       int            do_order,
			       int            ipostorder[],
			       double         given_mem,
			       void           (*sn_struct_handler)(),
			       void*          sn_struct_handler_arg
			       )
{
  int  i,ip,c,c_sn;
  int  in_previous_sn;
  int  nnz = 0;

  for (c=first_child[j]; c != -1; c = next_child[c]) {
    recursive_symbolic_elimination(c,A,
				   first_child,next_child,
				   n_sn,
				   sn_size,sn_up_size,sn_rowind,
				   sn_first_child,sn_next_child,sn_parent,
				   rowind, /* temporary */
				   column_to_sn_map,
				   map,
				   do_order,ipostorder,given_mem,
				   sn_struct_handler,sn_struct_handler_arg
				   );
  }

  
  in_previous_sn = 1;
  if (j == A->n) 
    in_previous_sn = 0; /* this is not a real column */
  else if (first_child[j] == -1) 
    in_previous_sn = 0; /* this is a leaf */
  else if (next_child[first_child[j]] != -1) 
    in_previous_sn = 0; /* more than 1 child */
  else if ((double)sn_up_size[column_to_sn_map[first_child[j]]]
	   *(double)(sn_size[column_to_sn_map[first_child[j]]]+1)
	   *sizeof(taucs_datatype) > given_mem)
    in_previous_sn = 0; /* size of supernode great than given memory */
  else { 
    /* check that the structure is nested */
    /* map contains child markers         */

    c=first_child[j];
    for (ip=(A->colptr)[j]; ip<(A->colptr)[j+1]; ip++) {
      i = (A->rowind)[ip];
      in_previous_sn = in_previous_sn && (map[i] == c);
    }
  }

  if (in_previous_sn) {
    c = first_child[j];
    c_sn = column_to_sn_map[c];
    column_to_sn_map[j] = c_sn;

    /* swap row indices so j is at the end of the */
    /* supernode, not in the update indices       */
    for (ip=sn_size[c_sn]; ip<sn_up_size[c_sn]; ip++) 
      if (sn_rowind[c_sn][ip] == j) break;
    assert(ip<sn_up_size[c_sn]);
    sn_rowind[c_sn][ip] = sn_rowind[c_sn][sn_size[c_sn]];
    sn_rowind[c_sn][sn_size[c_sn]] = j;

    /* mark the nonzeros in the map */
    for (ip=sn_size[c_sn]; ip<sn_up_size[c_sn]; ip++) 
      map[ sn_rowind[c_sn][ip] ] = j;

    sn_size   [c_sn]++;
    if((double)sn_size[c_sn]*(double)sn_up_size[c_sn]>=(1024.0*1024.0*1024.0))
      taucs_printf("debug!!!: sn_size[%d] = %d sn_up_size[%d] = %d\n ",c_sn,sn_size[c_sn],c_sn,sn_up_size[c_sn]);
    /* return c_sn; */
    return;
  }

  /* we are in a new supernode */
  if (j < A->n) {
    nnz = 1;
    rowind[0] = j;
    map[j]    = j;
    
    for (c=first_child[j]; c != -1; c = next_child[c]) {
      c_sn = column_to_sn_map[c];
      for (ip=sn_size[c_sn]; ip<sn_up_size[c_sn]; ip++) {
				i = sn_rowind[c_sn][ip];
				if (i > j && map[i] != j) { /* new row index */
					map[i] = j;
					rowind[nnz] = i;
					nnz++;
				}
      }
      if (sn_struct_handler)
				(*sn_struct_handler)(sn_struct_handler_arg,
			     c_sn,sn_up_size[c_sn],&(sn_rowind[c_sn]));
    }

    for (ip=(A->colptr)[j]; ip<(A->colptr)[j+1]; ip++) {
      i = (A->rowind)[ip];
      if (map[i] != j) { /* new row index */
				map[i] = j;
				rowind[nnz] = i;
				nnz++;
      }
    }
  }

  /* append childs from root*/
  if (j == A->n) {
    for (c=first_child[j]; c != -1; c = next_child[c]) {
      c_sn = column_to_sn_map[c];
      if (sn_struct_handler)
				(*sn_struct_handler)(sn_struct_handler_arg,
			     c_sn,sn_up_size[c_sn],&(sn_rowind[c_sn]));
    }
  }

  /*printf("children of sn %d: ",*n_sn);*/
  for (c=first_child[j]; c != -1; c = next_child[c]) {
    c_sn = column_to_sn_map[c];

		/* omer - added parent info */
		sn_parent[c_sn] = *n_sn;

    /*printf("%d ",c_sn);*/
		if (c==first_child[j]) 
      sn_first_child[*n_sn] = c_sn;
		else {
      sn_next_child[ c_sn ] = sn_first_child[*n_sn];
      sn_first_child[*n_sn] = c_sn;
    }
  }
  /*printf("\n");*/

  if (j < A->n) {
    column_to_sn_map[j] = *n_sn;
    sn_size   [*n_sn] = 1;
    sn_up_size[*n_sn] = nnz;
    sn_rowind [*n_sn] = (int*) taucs_malloc(nnz * sizeof(int));
    for (ip=0; ip<nnz; ip++) sn_rowind[*n_sn][ip] = rowind[ip];
    if (do_order) {
      /* Sivan and Vladimir: we think that we can sort in */
      /* column order, not only in etree postorder.       */
      /*
				radix_sort(sn_rowind [*n_sn],nnz);
				qsort(sn_rowind [*n_sn],nnz,sizeof(int),compare_ints);
      */
      compare_indirect_map = ipostorder;
      qsort(sn_rowind [*n_sn],nnz,sizeof(int),compare_indirect_ints);
    }
    assert(sn_rowind [*n_sn][0] == j);
    (*n_sn)++;
  }
}

static
void recursive_postorder(int  j,
			 int  first_child[],
			 int  next_child[],
			 int  postorder[],
			 int  ipostorder[],
			 int* next
			 )
{
  int c;


  for (c=first_child[j]; c != -1; c = next_child[c]) {
    /*printf("*** %d is child of %d\n",c,j);*/
    recursive_postorder(c,first_child,next_child,
			postorder,ipostorder,next
			);
  }
  /*  printf(">>> j=%d next=%d\n",j,*next);*/
  if (postorder)  postorder [*next] = j;
  if (ipostorder) ipostorder[j] = *next;
  (*next)++;
}

static void
taucs_ccs_ooc_symbolic_elimination(taucs_ccs_matrix* A,
				   void* vL,
					 int* sn_parent,
				   int do_order,
				   int do_column_to_sn_map,
				   double given_mem,
				   void           (*sn_struct_handler)(),
				   void*          sn_struct_handler_arg
				   )
{
  supernodal_factor_matrix_ooc_ldlt* L = (supernodal_factor_matrix_ooc_ldlt*) vL;
	int* first_child;
  int* next_child;
  int j;
  int* column_to_sn_map;
  int* map;
  int* rowind;
  int* parent;
  int* ipostorder;

  L->n         = A->n;
  L->sn_struct = (int**)taucs_malloc((A->n  )*sizeof(int*));
  L->sn_size   = (int*) taucs_malloc((A->n+1)*sizeof(int));
  L->sn_up_size   = (int*) taucs_malloc((A->n+1)*sizeof(int));
  L->first_child = (int*) taucs_malloc((A->n+1)*sizeof(int));
  L->next_child  = (int*) taucs_malloc((A->n+1)*sizeof(int));

  column_to_sn_map = (int*)taucs_malloc((A->n+1)*sizeof(int));
  map              = (int*) taucs_malloc((A->n+1)*sizeof(int));

  first_child = (int*) taucs_malloc(((A->n)+1)*sizeof(int));
  next_child  = (int*) taucs_malloc(((A->n)+1)*sizeof(int));
    
  rowind      = (int*) taucs_malloc((A->n)*sizeof(int));

  taucs_printf("STARTING SYMB 1\n");

  /* compute the vertex elimination tree */
  parent      = (int*)taucs_malloc((A->n+1)*sizeof(int));
	
  taucs_ccs_etree(A,parent,NULL,NULL,NULL);
  for (j=0; j <= (A->n); j++) first_child[j] = -1;
  for (j = (A->n)-1; j >= 0; j--) {
    int p = parent[j];
    next_child[j] = first_child[p];
    first_child[p] = j;
  }
  taucs_free(parent);

  taucs_printf("STARTING SYMB 2\n");

  ipostorder = (int*)taucs_malloc((A->n+1)*sizeof(int));
  { 
    int next = 0;
    /*int* postorder = (int*)taucs_malloc((A->n+1)*sizeof(int));*/
    recursive_postorder(A->n,first_child,next_child,
			NULL,
			ipostorder,&next);
    /*
    printf("ipostorder ");
    for (j=0; j <= (A->n); j++) printf("%d ",ipostorder[j]);
    printf("\n");
    printf(" postorder ");
    for (j=0; j <= (A->n); j++) printf("%d ",postorder[j]);
    printf("\n");
    */
  }

  taucs_printf("STARTING SYMB 3\n");

  L->n_sn = 0;
  for (j=0; j < (A->n); j++) map[j] = -1;
  for (j=0; j <= (A->n); j++) (L->first_child)[j] = (L->next_child)[j] = -1;

  taucs_printf("STARTING SYMB\n");

  recursive_symbolic_elimination(A->n,
				 A,
				 first_child,next_child,
				 &(L->n_sn),
				 L->sn_size,L->sn_up_size,L->sn_struct,
				 L->first_child,L->next_child,sn_parent,
				 rowind,
				 column_to_sn_map,
				 map,
				 do_order,ipostorder,given_mem,
				 sn_struct_handler,sn_struct_handler_arg
				 );

  taucs_printf("AFTER SYMB\n");

  {
    double nnz   = 0.0;
    double flops = 0.0;
    int sn,i,colnnz;
    for (sn=0; sn<(L->n_sn); sn++) {
      for (i=0, colnnz = (L->sn_up_size)[sn]; 
	   i<(L->sn_size)[sn]; 
	   i++, colnnz--) {
	flops += 1.0 + ((double)(colnnz)) * ((double)(colnnz));
	nnz   += (double) (colnnz);
      }
    }
    taucs_printf("\t\tSymbolic Analysis of LL^T: %.2e nonzeros, %.2e flops\n",
		 nnz, flops);
  }

  /*
  {
    int i;
    printf("c2sn: ");
    for (i=0; i<A->n; i++) printf("%d ",column_to_sn_map[i]);
    printf("\n");
  }
  */
  
  L->sn_struct = (int**)taucs_realloc( L->sn_struct,(L->n_sn  )*sizeof(int*));
  L->sn_size   = (int*) taucs_realloc( L->sn_size,(L->n_sn+1)*sizeof(int));
  L->sn_up_size   = (int*) taucs_realloc(L->sn_up_size,(L->n_sn+1)*sizeof(int));
  L->first_child = (int*) taucs_realloc(L->first_child,(L->n_sn+1)*sizeof(int));
  L->next_child  = (int*) taucs_realloc(L->next_child,(L->n_sn+1)*sizeof(int));
	
  L->sn_blocks     = taucs_calloc((L->n_sn), sizeof(taucs_datatype*)); /* so we can free before allocation */
  L->up_blocks     = taucs_calloc((L->n_sn), sizeof(taucs_datatype*));

  taucs_free(rowind);
  taucs_free(map);

  if(do_column_to_sn_map)
    L->col_to_sn_map = column_to_sn_map;
  else
    taucs_free(column_to_sn_map);

  taucs_free(next_child);
  taucs_free(first_child);
  taucs_free(ipostorder);
}

/*************************************************************/
/* left-looking factor routines                              */
/*************************************************************/

static void
recursive_leftlooking_supernodal_update(int J,int K,
					int bitmap[],
					taucs_datatype* dense_update_matrix,
					taucs_ccs_matrix* A,
					supernodal_factor_matrix_ooc_ldlt* L)
{
  int i,j,ir;
  int  child;
  int* first_child   = L->first_child;
  int* next_child    = L->next_child;
  int sn_size_father = (L->sn_size)[J];
  int sn_up_size_father = (L->sn_up_size)[J];
  int sn_size_child = (L->sn_size)[K];
  int sn_up_size_child = (L->sn_up_size)[K];
  int exist_upd=0;
  int first_row=-1;
  int row_count=0;
  int PK,M,N,LDA,LDB,LDC;

  for(i=0;i<sn_size_father;i++) {
    bitmap[L->sn_struct[J][i]]=i+1;
  }

  for(i=sn_size_father;i<sn_up_size_father;i++)
    bitmap[L->sn_struct[J][i]] = i - sn_size_father + 1;

  for(i=sn_size_child;i<sn_up_size_child;i++)
    /* is this row index included in the columns of sn J? */
    if(bitmap[L->sn_struct[K][i]]
       && L->sn_struct[K][i] <= L->sn_struct[J][sn_size_father-1]) {
      if(!exist_upd) first_row = i;
      row_count++;
      exist_upd = 1;
      /*taucs_printf("update from K = %d to J = %d \n",K,J);*/
      /* loop over columns of sn K */
            
      /* for(j=0;j<sn_size_child;j++)
	for(ir=i;ir<sn_up_size_child;ir++)
	  if( L->sn_struct[K][ir] <= L->sn_struct[J][sn_size_father-1]){
	    L->sn_blocks[J][ (bitmap[L->sn_struct[K][i]]-1)*(L->sn_blocks_ld[J])+(bitmap[L->sn_struct[K][ir]]-1)] -= L->up_blocks[K][j*(L->up_blocks_ld[K])+ir-sn_size_child]* L->up_blocks[K][j*L->up_blocks_ld[K]+i-sn_size_child];
	    taucs_printf("sn_block: L[%d,%d] = %lf\n",(bitmap[L->sn_struct[K][ir]]-1),(bitmap[L->sn_struct[K][i]]-1),L->sn_blocks[J][ (bitmap[L->sn_struct[K][i]]-1)*(L->sn_blocks_ld[J])+(bitmap[L->sn_struct[K][ir]]-1)]);}
	  else{
	    L->up_blocks[J][ (bitmap[L->sn_struct[K][i]]-1)*(L->up_blocks_ld[J])+(bitmap[L->sn_struct[K][ir]]-1)] -=  L->up_blocks[K][j*L->up_blocks_ld[K]+ir-sn_size_child]* L->up_blocks[K][j*L->up_blocks_ld[K]+i-sn_size_child];
	   taucs_printf("up_block: L[%d,%d] = %lf\n",(bitmap[L->sn_struct[K][ir]]-1),(bitmap[L->sn_struct[K][i]]-1),L->up_blocks[J][ (bitmap[L->sn_struct[K][i]]-1)*(L->up_blocks_ld[J])+(bitmap[L->sn_struct[K][ir]]-1)]);
	   }*/
        }

  if(exist_upd){
    LDA = LDB = (L->sn_up_size)[K]-(L->sn_size)[K];
    M  = sn_up_size_child - first_row ; /* +-1 ? */    
    LDC =  sn_up_size_father;
    N  = row_count; 
    PK = L->sn_size[K];    
   
    taucs_gemm ("No Conjugate",
		"Conjugate",
		&M,&N,&PK,
		&taucs_one_const,
		&(L->up_blocks[K][first_row-sn_size_child]),&LDA,
		&(L->up_blocks[K][first_row-sn_size_child]),&LDB,
		&taucs_zero_const,
		dense_update_matrix,&LDC);

    /*for(j=0;j<row_count;j++)
       for(ir=0;ir<sn_up_size_father;ir++)
	 taucs_printf("dense[%d,%d] = %lf\n",ir,j,dense_update_matrix[j*LDC+ir]);
    */
    for(j=0;j<row_count;j++)
      for(ir=j;ir<row_count;ir++){
	/*
	L->sn_blocks[J][(bitmap[L->sn_struct[K][first_row+j]]-1)*sn_size_father+(bitmap[L->sn_struct[K][first_row+ir]]-1)] -= dense_update_matrix[j*LDC+ir];
	*/
	L->sn_blocks[J][(bitmap[L->sn_struct[K][first_row+j]]-1)*sn_size_father+(bitmap[L->sn_struct[K][first_row+ir]]-1)] =
	  taucs_sub(L->sn_blocks[J][(bitmap[L->sn_struct[K][first_row+j]]-1)*sn_size_father+(bitmap[L->sn_struct[K][first_row+ir]]-1)] , dense_update_matrix[j*LDC+ir]);

      }

    for(j=0;j<row_count;j++)
      for(ir=row_count;ir<M;ir++){
	/*
	L->up_blocks[J][(bitmap[L->sn_struct[K][first_row+j]]-1)*(L->sn_up_size[J]-L->sn_size[J])+(bitmap[L->sn_struct[K][ir+first_row]]-1)] -= dense_update_matrix[j*LDC+ir];
	*/
	L->up_blocks[J][(bitmap[L->sn_struct[K][first_row+j]]-1)*(L->sn_up_size[J]-L->sn_size[J])+(bitmap[L->sn_struct[K][ir+first_row]]-1)] =
	  taucs_sub(L->up_blocks[J][(bitmap[L->sn_struct[K][first_row+j]]-1)*(L->sn_up_size[J]-L->sn_size[J])+(bitmap[L->sn_struct[K][ir+first_row]]-1)] , dense_update_matrix[j*LDC+ir]);
	}
    for(i=0;i<sn_up_size_father;i++)
      bitmap[L->sn_struct[J][i]]=0;
    
    for (child = first_child[K]; child != -1; child = next_child[child]) {
      recursive_leftlooking_supernodal_update(J,child,
					      bitmap,dense_update_matrix,
					      A,L);
    }
  }
  else
    for(i=0;i<sn_up_size_father;i++)
      bitmap[L->sn_struct[J][i]]=0;

}


static int
leftlooking_supernodal_front_factor(int sn,
				    int* indmap,
				    taucs_ccs_matrix* A,
				    supernodal_factor_matrix_ooc_ldlt* L)
{
  int ip,jp;
  int*    ind;
  taucs_datatype* re;
  int INFO;
  int sn_size = (L->sn_size)[sn];
  int up_size = (L->sn_up_size)[sn] - (L->sn_size)[sn];

  /* creating transform for real indices */
  for(ip=0;ip<(L->sn_up_size)[sn];ip++) indmap[(L->sn_struct)[sn][ip]] = ip;

  for(jp=0;jp<sn_size;jp++) {
    ind = &(A->rowind[A->colptr[ (L->sn_struct)[sn][jp] ]]);
    re  = &(A->taucs_values[A->colptr[ (L->sn_struct)[sn][jp] ]]); 
    for(ip=0;
	ip < A->colptr[ (L->sn_struct)[sn][jp] + 1 ] 
           - A->colptr[ (L->sn_struct)[sn][jp] ];
	ip++) {
      if (indmap[ind[ip]] < sn_size)
	(L->sn_blocks)[sn][sn_size*jp + indmap[ind[ip]]] =
	  taucs_add( (L->sn_blocks)[sn][sn_size*jp + indmap[ind[ip]]] , re[ip]);
      else
	(L->up_blocks)[sn][up_size*jp + indmap[ind[ip]] - sn_size] =
	  taucs_add((L->up_blocks)[sn][up_size*jp + indmap[ind[ip]] - sn_size] , re[ip]);
    }
  }

  /* we use the BLAS through the Fortran interface */

  /* solving of lower triangular system for L */
  if (sn_size)
    taucs_potrf ("LOWER",
		 &sn_size,
		 (L->sn_blocks)[sn],&sn_size,
		 &INFO);

  if (INFO) {
    taucs_printf("\t\tLL^T Factorization: Matrix is not positive definite.\n");
    taucs_printf("\t\t in sn = %d   nonpositive pivot in column %d\n",
		 sn,(L->sn_struct)[sn][INFO-1]);
    return -1;
  }

  /* getting completion for found columns of L */
  if (up_size && sn_size)
    taucs_trsm ("Right",
		"Lower",
		"Conjugate",
		"No unit diagonal",
		&up_size,&sn_size,
		&taucs_one_const,
		(L->sn_blocks)[sn],&sn_size,
		(L->up_blocks)[sn],&up_size);
  
  /* zeroes map */
  for(ip=0;ip<(L->sn_up_size)[sn];ip++) indmap[(L->sn_struct)[sn][ip]] = 0;

  return 0;
}

static int
recursive_leftlooking_supernodal_factor_ldlt(int sn,       /* this supernode */
																						int is_root,  /* is v the root? */
																						int* map,
																						taucs_ccs_matrix* A,
																						supernodal_factor_matrix_ooc_ldlt* L)
{
  int  child;
	int sn_size = L->sn_size[sn];
	int sn_up_size = L->sn_up_size[sn];
  taucs_datatype* dense_update_matrix = NULL;

  for (child = L->first_child[sn]; child != -1; child = L->next_child[child]) {
    if (recursive_leftlooking_supernodal_factor_ldlt(child,
						    FALSE,
						    map,
						    A,L)) {
      /* failure */
      return -1;
    }
  }    

  if (!is_root) {
    (L->sn_blocks)[sn] = (taucs_datatype*)
			taucs_calloc((sn_size*sn_size),sizeof(taucs_datatype));
    (L->up_blocks)[sn] = (taucs_datatype*)
			taucs_calloc(((sn_up_size-sn_size)*sn_size),sizeof(taucs_datatype));
    if (!dense_update_matrix && (L->first_child[sn] != -1)) 
      dense_update_matrix = (taucs_datatype*) 
				taucs_calloc(sn_up_size*sn_size,sizeof(taucs_datatype));

    for (child = L->first_child[sn]; child != -1; child = L->next_child[child])
      recursive_leftlooking_supernodal_update(sn,child,
								map,dense_update_matrix,
					      A,L);

    taucs_free(dense_update_matrix);
    if (leftlooking_supernodal_front_factor(sn,map,A,L)) {
      /* nonpositive pivot */
      return -1;
    }
  }

  return 0;
}

#if 0
void* taucs_ccs_factor_ldlt_ll(taucs_ccs_matrix* A)
{
  supernodal_factor_matrix_ooc_ldlt* L;
  int i,j,ip,jp;
  int sn,p;
  int* map;
  double wtime, ctime;
 
  wtime = taucs_wtime();
  ctime = taucs_ctime();

  L = multifrontal_supernodal_create();

  taucs_ccs_ooc_symbolic_elimination(A,L,
				     TRUE /* sort row indices */,
				     FALSE /* don't return col_tosn_map */,
				     1.0/0.0,
				     NULL,NULL /* sn_struct handler*/ );

  wtime = taucs_wtime()-wtime;
  ctime = taucs_ctime()-ctime;
  taucs_printf("\t\tSymbolic Analysis            = % 10.3f seconds (%.3f cpu)\n",
	       wtime,ctime);


  map  = (int*)taucs_malloc((A->n+1)*sizeof(int));

  wtime = taucs_wtime();
  ctime = taucs_ctime();

  if (recursive_leftlooking_supernodal_factor_ldlt((L->n_sn),  
						  TRUE, 
						  map,
						  A,L)) {
    ooc_supernodal_factor_free(L);
    taucs_free(map);

    return NULL;
  }

  wtime = taucs_wtime()-wtime;
  ctime = taucs_ctime()-ctime;
  taucs_printf("\t\tSupernodal Left-Looking LL^T = % 10.3f seconds (%.3f cpu)\n",
	       wtime,ctime);

  taucs_free(map);

  return (void*) L;
}
#endif

/*************************************************************/
/* left-looking ooc factor routines                          */
/*************************************************************/

static
void
ooc_sn_struct_handler(void* argument, 
		      int sn, int sn_up_size, int* sn_struct_ptr[])
{
  taucs_io_handle* handle = (taucs_io_handle*) argument;
  taucs_io_append(handle,
		  IO_BASE+sn, /* matrix written in postorder */
		  1,
		  sn_up_size,
		  TAUCS_INT,
		  *sn_struct_ptr);
  taucs_free(*sn_struct_ptr);
  *sn_struct_ptr = NULL;
}


static int
taucs_io_append_supernode(int sn,
													 taucs_io_handle* handle,
													 supernodal_factor_matrix_ooc_ldlt* L)
{
	int ret;
	int  sn_size = L->sn_size[sn];
	int	 sn_up_size = L->sn_up_size[sn];

	ret = taucs_io_append(handle,IO_BASE+L->n_sn+2*sn,
												sn_size,sn_size,
												TAUCS_CORE_DATATYPE,L->sn_blocks[sn]);

	if (ret) return ret; /*failure*/

	ret = taucs_io_append(handle,IO_BASE+L->n_sn+2*sn+1,
												sn_up_size - sn_size,sn_size,
												TAUCS_CORE_DATATYPE,L->up_blocks[sn]); 

	return ret;
}
														

static int
recursive_append_L(	int sn,   /* this supernode */
										int is_root,/* is v the root?*/
										taucs_io_handle* handle,
										supernodal_factor_matrix_ooc_ldlt* L)
{
  int  child;
  
  for (child = L->first_child[sn]; child != -1; child = L->next_child[child]) {
    if (recursive_append_L(child,FALSE,handle,L)) {
      /* failure */
      return -1;
    }
  }
    
  if (!is_root) { 
    taucs_io_append_supernode(sn,handle,L);
		ooc_supernodal_free(L,sn);
 }

  return 0;
}

static int
recursive_read_L(int sn,   /* this supernode */
		 int is_root,/* is v the root?*/
		 taucs_io_handle* handle,
		 supernodal_factor_matrix_ooc_ldlt* L)
{
  int  child;
  /*int  sn_size;*/
  int* first_child   = L->first_child;
  int* next_child    = L->next_child;
  
  for (child = first_child[sn]; child != -1; child = next_child[child]) {
    if (recursive_read_L(child,FALSE,handle,L)) {
      /* failure */
      return -1;
    }
  }
    
  if (!is_root) { 
    (L->sn_blocks)[sn] = (taucs_datatype*)taucs_calloc(((L->sn_size)[sn])
						 *((L->sn_size)[sn]),
						 sizeof(taucs_datatype));
    (L->up_blocks   )[sn] = (taucs_datatype*)taucs_calloc(((L->sn_up_size)[sn]-(L->sn_size)[sn])
						    *((L->sn_size)[sn]),
						    sizeof(taucs_datatype));

    taucs_io_read(handle,IO_BASE+L->n_sn+2*sn,
		  L->sn_size[sn],L->sn_size[sn],
		  TAUCS_CORE_DATATYPE,L->sn_blocks[sn]);

    taucs_io_read(handle,IO_BASE+L->n_sn+2*sn+1,
		  L->sn_up_size[sn] - L->sn_size[sn],L->sn_size[sn],
		  TAUCS_CORE_DATATYPE,L->up_blocks[sn]); 
 }

  return 0;
}

static int
recursive_read_L_cols(int sn,   /* this supernode */
											int is_root,/* is v the root?*/
											taucs_io_handle* handle,
											supernodal_factor_matrix_ooc_ldlt* L)
{
  int  child;
  int  sn_up_size = L->sn_up_size[sn];
  
  for (child = L->first_child[sn]; child != -1; child = L->next_child[child]) {
    if (recursive_read_L_cols(child,FALSE,handle,L)) {
      /* failure */
      return -1;
    }
  }
    
  if (!is_root) { 
    L->sn_struct[sn] = (int*)taucs_malloc(sn_up_size*sizeof(int));
    taucs_io_read(handle,IO_BASE+sn,1,sn_up_size,TAUCS_INT,L->sn_struct[sn]);
  }

  return 0;
}


static double
recursive_compute_supernodes_ll_in_core(int sn,       /* this supernode */
																				int is_root,  /* is v the root? */
																				double avail_mem,
																				int* sn_in_core,
																				supernodal_factor_matrix_ooc_ldlt* L)
{
  int  child;
	int  sn_size = L->sn_size[sn];
	int  sn_up_size = L->sn_up_size[sn];
  /*double curr_avail_mem = avail_mem;*/
  double child_mem = 0.0;
  double children_mem = 0.0;
  double total_mem = 0.0;
  
  /*new_panel = 0;*/
  for (child = L->first_child[sn]; child != -1; child = L->next_child[child]) {
    child_mem = recursive_compute_supernodes_ll_in_core(child,FALSE,avail_mem,sn_in_core,L);
    /*if (child_mem == 0) new_panel = 1; */
    children_mem += child_mem;
  }

  /*if (first_child[sn] == -1) 
    total_mem = (double)(L->sn_size)[sn]*(double)(L->sn_up_size)[sn]*sizeof(taucs_datatype)+(double)(L->sn_up_size)[sn]*sizeof(int);
  else
    total_mem = children_mem + 2*(double)(L->sn_size)[sn]*(double)(L->sn_up_size)[sn]*sizeof(taucs_datatype)+(double)(L->sn_up_size)[sn]*sizeof(int);
  
  if (total_mem <= avail_mem || first_child[sn] == -1) 
    sn_in_core[sn] = 1;
   else 
     sn_in_core[sn] = 0;
  return total_mem;
  */
	if (!is_root) {
		total_mem = children_mem + (double)sn_size*(double)sn_up_size*sizeof(taucs_datatype) +
															 (double)sn_up_size*sizeof(int);
	} else {
		total_mem = children_mem;
	}

  if ((total_mem + (double)sn_size*(double)sn_up_size*sizeof(taucs_datatype)) <= avail_mem || 
			L->first_child[sn] == -1) {
    sn_in_core[sn] = 1;
    return total_mem;
  } else {
    sn_in_core[sn] = 0;
    return (total_mem + (double)sn_size*(double)sn_up_size*sizeof(taucs_datatype));
  }
  
}


static void
recursive_compute_supernodes_ipostorder(int sn,       /* this supernode */
					int is_root,  /* is v the root? */
					int* current_index_ptr,
					supernodal_factor_matrix_ooc_ldlt* L)
{
  int  child;
  int* first_child   = L->first_child;
  int* next_child    = L->next_child;
 
  for (child = first_child[sn]; child != -1; child = next_child[child]) {
    recursive_compute_supernodes_ipostorder(child,
					     FALSE,
					     current_index_ptr,
					     L);
  
  }
  L->ipostorder[sn] = *current_index_ptr;
  (*current_index_ptr)++;
 }


/******************** OOC SOLVE **************************/

static void 
recursive_supernodal_solve_l_ooc(int sn,       /* this supernode */
				 int is_root,  /* is v the root? */
				 taucs_io_handle* handle,
				 int n_sn,
				 int* first_child, int* next_child,
				 int** sn_struct, int* sn_sizes, int* sn_up_sizes,
				 taucs_datatype x[], taucs_datatype b[],
				 taucs_datatype t[])
{
  int child;
  int  sn_size; /* number of rows/columns in the supernode    */
  int  up_size; /* number of rows that this supernode updates */
  int    ione = 1;
  /*
  double done = 1.0;
  double dzero = 0.0;
  */
  taucs_datatype* xdense = NULL;
  taucs_datatype* bdense = NULL;
  double  flops;
  int i,j,ip,jp;
  taucs_datatype* sn_block = NULL;
  taucs_datatype* up_block = NULL;

  for (child = first_child[sn]; child != -1; child = next_child[child]) {
    recursive_supernodal_solve_l_ooc(child,
				     FALSE,
				     handle,
				     n_sn,
				     first_child,next_child,
				     sn_struct,sn_sizes,sn_up_sizes,
				     x,b,t);
  }

  if(!is_root) {
    sn_size = sn_sizes[sn];
    up_size = sn_up_sizes[sn] - sn_sizes[sn];

    sn_struct[sn] = (int*)taucs_malloc((sn_size+up_size)*sizeof(int));
    taucs_io_read(handle,IO_BASE+sn,1,sn_size+up_size,TAUCS_INT,sn_struct[sn]);
    
    sn_block = (taucs_datatype*)taucs_calloc(sn_size*sn_size,sizeof(taucs_datatype));
    taucs_io_read(handle,IO_BASE+n_sn+2*sn,
		  sn_size,
		  sn_size ,
		  TAUCS_CORE_DATATYPE,sn_block);
   
    if (up_size > 0 && sn_size > 0) {
      up_block = (taucs_datatype*)taucs_calloc(up_size*sn_size,sizeof(taucs_datatype));
      taucs_io_read(handle,IO_BASE+n_sn+2*sn+1,
		    up_size,
		    sn_size ,
		    TAUCS_CORE_DATATYPE,up_block);
    }

    flops = ((double)sn_size)*((double)sn_size) 
      + 2.0*((double)sn_size)*((double)up_size);

    if (flops > BLAS_FLOPS_CUTOFF) {
      xdense = t;
      bdense = t + sn_size;
      
      for (i=0; i<sn_size; i++)
	xdense[i] = b[ sn_struct[ sn ][ i ] ];
      for (i=0; i<up_size; i++)
	bdense[i] = taucs_zero;
      
      taucs_trsm ("Left",
	     "Lower",
	     "No Conjugate",
	     "No unit diagonal",
	     &sn_size,&ione,
	     &taucs_one_const,
	     sn_block,&sn_size,
	     xdense , &sn_size);
      
      if (up_size > 0 && sn_size > 0) 
	taucs_gemm ("No Conjugate","No Conjugate",
		    &up_size, &ione, &sn_size,
		    &taucs_one_const,
		    up_block,&up_size,
		    xdense       ,&sn_size,
		    &taucs_zero_const,
		    bdense       ,&up_size);
      
      for (i=0; i<sn_size; i++)
	x[ sn_struct[ sn][ i ] ]  = xdense[i];
      for (i=0; i<up_size; i++)
	/*b[ sn_struct[ sn ][ sn_size + i ] ] -= bdense[i];*/
	b[ sn_struct[ sn ][ sn_size + i ] ] =
	  taucs_sub(b[ sn_struct[ sn ][ sn_size + i ] ] , bdense[i]);

    } else if (sn_size > SOLVE_DENSE_CUTOFF) {

      xdense = t;
      bdense = t + sn_size;
      
      for (i=0; i<sn_size; i++)
	xdense[i] = b[ sn_struct[ sn ][ i ] ];
      for (i=0; i<up_size; i++)
	bdense[i] = taucs_zero;
      
      for (jp=0; jp<sn_size; jp++) {
	/*xdense[jp] = xdense[jp] / sn_block[ sn_size*jp + jp];*/
	xdense[jp] = taucs_div(xdense[jp] , sn_block[ sn_size*jp + jp]);

	for (ip=jp+1; ip<sn_size; ip++) {
	  /*xdense[ip] -= xdense[jp] * sn_block[ sn_size*jp + ip];*/
	  xdense[ip] = taucs_sub(xdense[i],
				 taucs_mul(xdense[jp] , sn_block[ sn_size*jp + ip]));
	}
      }

      for (jp=0; jp<sn_size; jp++) {
	for (ip=0; ip<up_size; ip++) {
	  /*bdense[ip] += xdense[jp] * up_block[ up_size*jp + ip];*/
	  bdense[ip] = taucs_add(bdense[ip],
				 taucs_mul(xdense[jp] , up_block[ up_size*jp + ip]));
	}
      }

      for (i=0; i<sn_size; i++)
	x[ sn_struct[ sn][ i ] ]  = xdense[i];
      for (i=0; i<up_size; i++)
	/*b[ sn_struct[ sn ][ sn_size + i ] ] -= bdense[i];*/
	b[ sn_struct[ sn ][ sn_size + i ] ] =
	  taucs_sub(b[ sn_struct[ sn ][ sn_size + i ] ] , bdense[i]);
      
    } else {
      for (jp=0; jp<sn_size; jp++) {
	j = sn_struct[sn][jp];
	/*x[j] = b[j] / sn_block[ sn_size*jp + jp];*/
	x[j] = taucs_div(b[j] , sn_block[ sn_size*jp + jp]);
	for (ip=jp+1; ip<sn_size; ip++) {
	  i = sn_struct[sn][ip];
	  /*b[i] -= x[j] * sn_block[ sn_size*jp + ip];*/
	  b[i] = taucs_sub(b[i],
			   taucs_mul(x[j] , sn_block[ sn_size*jp + ip]));
	}

	for (ip=0; ip<up_size; ip++) {
	  i = sn_struct[sn][sn_size + ip];
	  /*b[i] -= x[j] * up_block[ up_size*jp + ip];*/
	  b[i] = taucs_sub(b[i],
			   taucs_mul(x[j] , up_block[ up_size*jp + ip]));
	}
      }
    }
    taucs_free(sn_struct[sn]);
    taucs_free(sn_block);
    if (up_size > 0 && sn_size > 0) taucs_free(up_block);
    sn_struct[sn] = NULL;
    sn_block = NULL;
    up_block = NULL;
  }
}

static void 
recursive_supernodal_solve_lt_ooc(int sn,       /* this supernode */
				  int is_root,  /* is v the root? */
				  taucs_io_handle* handle,
				  int n_sn,
				  int* first_child, int* next_child,
				  int** sn_struct, int* sn_sizes, int* sn_up_sizes,
				  taucs_datatype x[], taucs_datatype b[],
				  taucs_datatype t[])
{
  int child;
  int  sn_size; /* number of rows/columns in the supernode    */
  int  up_size; /* number of rows that this supernode updates */
  int    ione = 1;
  taucs_datatype* xdense = NULL;
  taucs_datatype* bdense = NULL;
  double  flops;
  int i,j,ip,jp;
  taucs_datatype* sn_block = NULL;
  taucs_datatype* up_block = NULL;

  if(!is_root) {

    sn_size = sn_sizes[sn];
    up_size = sn_up_sizes[sn]-sn_sizes[sn];

    sn_struct[sn] = (int*)taucs_malloc((sn_size+up_size)*sizeof(int));
    taucs_io_read(handle,IO_BASE+sn,1,sn_size+up_size,TAUCS_INT,sn_struct[sn]);
    
    sn_block = (taucs_datatype*)taucs_calloc(sn_size*sn_size,sizeof(taucs_datatype));
    taucs_io_read(handle,IO_BASE+n_sn+2*sn,
		  sn_size,
		  sn_size ,
		  TAUCS_CORE_DATATYPE,sn_block);
    if (up_size > 0 && sn_size > 0){
      up_block = (taucs_datatype*)taucs_calloc(up_size*sn_size,sizeof(taucs_datatype));
      taucs_io_read(handle,IO_BASE+n_sn+2*sn+1,
		    up_size,
		    sn_size ,
		    TAUCS_CORE_DATATYPE,up_block);
    }

    flops = ((double)sn_size)*((double)sn_size) 
      + 2.0*((double)sn_size)*((double)up_size);

    if (flops > BLAS_FLOPS_CUTOFF) {
      bdense = t;
      xdense = t + sn_size;
      
      for (i=0; i<sn_size; i++)
	bdense[i] = b[ sn_struct[ sn][ i ] ];
      for (i=0; i<up_size; i++)
	xdense[i] = x[ sn_struct[sn][sn_size+i] ];
      
      if (up_size > 0 && sn_size > 0)
	taucs_gemm ("Conjugate","No Conjugate",
	       &sn_size, &ione, &up_size,
	       &taucs_minusone_const,
	       up_block,&up_size,
	       xdense       ,&up_size,
	       &taucs_one_const,
	       bdense       ,&sn_size);
            
      taucs_trsm ("Left",
	     "Lower",
	     "Conjugate",
	     "No unit diagonal",
	     &sn_size,&ione,
	     &taucs_one_const,
	     sn_block,&sn_size,
	     bdense       ,&sn_size);
      
      for (i=0; i<sn_size; i++)
	x[ sn_struct[ sn][ i ] ]  = bdense[i];
    
    } else if (sn_size > SOLVE_DENSE_CUTOFF) {
      bdense = t;
      xdense = t + sn_size;
      
      for (i=0; i<sn_size; i++)
	bdense[i] = b[ sn_struct[ sn][ i ] ];
      
      for (i=0; i<up_size; i++)
	xdense[i] = x[ sn_struct[sn][sn_size+i] ];
     
      for (ip=sn_size-1; ip>=0; ip--) {
	for (jp=0; jp<up_size; jp++) {
	  /*bdense[ip] -= xdense[jp] * up_block[ up_size*ip + jp];*/
	  bdense[ip] = taucs_sub(bdense[ip],
				 taucs_mul(xdense[jp] , up_block[ up_size*ip + jp]));
	}
      }
     
      for (ip=sn_size-1; ip>=0; ip--) {
	for (jp=sn_size-1; jp>ip; jp--) {
	  /*bdense[ip] -= bdense[jp] * sn_block[ sn_size*ip + jp];*/
	  bdense[ip] = taucs_sub(bdense[ip],
				 taucs_mul(bdense[jp] , sn_block[ sn_size*ip + jp]));
	}
	/*bdense[ip] = bdense[ip] / sn_block[ sn_size*ip + ip];*/
	bdense[ip] = taucs_div(bdense[ip] , sn_block[ sn_size*ip + ip]);
      }

      for (i=0; i<sn_size; i++)
	x[ sn_struct[ sn][ i ] ]  = bdense[i];
    
    } else {
      for (ip=sn_size-1; ip>=0; ip--) {
	i = sn_struct[sn][ip];

	for (jp=0; jp<up_size; jp++) {
	  j = sn_struct[sn][sn_size + jp];
	  /*b[i] -= x[j] * up_block[ up_size*ip + jp];*/
	  b[i] = taucs_sub(b[i],
			   taucs_mul(x[j] , up_block[ up_size*ip + jp]));
	}

	for (jp=sn_size-1; jp>ip; jp--) {
	  j = sn_struct[sn][jp];
	  /*b[i] -= x[j] * sn_block[ sn_size*ip + jp];*/
	  b[i] = taucs_sub(b[i],
			   taucs_mul(x[j] , sn_block[ sn_size*ip + jp]));
	}
	/*x[i] = b[i] / sn_block[ sn_size*ip + ip];*/
	x[i] = taucs_div(b[i] , sn_block[ sn_size*ip + ip]);
      }

    }
    taucs_free(sn_struct[sn]);
    taucs_free(sn_block);
    if (up_size > 0 && sn_size > 0) taucs_free(up_block);
    sn_struct[sn] = NULL;
    sn_block = NULL;
    up_block = NULL;
  }

  for (child = first_child[sn]; child != -1; child = next_child[child]) {
    recursive_supernodal_solve_lt_ooc(child,
				      FALSE,
				      handle,
				      n_sn,
				      first_child,next_child,
				      sn_struct,sn_sizes,sn_up_sizes,
				      x,b,t);
  }
}

int taucs_dtl(ooc_solve_ldlt)(void* vL,
			     void* vx, void* vb)
{
  taucs_io_handle* handle = (taucs_io_handle*) vL;
  taucs_datatype* x = (taucs_datatype*) vx;
  taucs_datatype* b = (taucs_datatype*) vb;
  /*  char* filename = (char*) vL; */
  supernodal_factor_matrix_ooc_ldlt* L;

  taucs_datatype* y;
  taucs_datatype* t; /* temporary vector */
  int     i;

  L = multifrontal_supernodal_create();
  /* READ n, n_sn, first_child, next_child*/
  /*
  if(!MULTIFILE)
    handle = taucs_io_open_singlefile(filename);
  else
    handle = taucs_io_open_multifile(filename);
  */
  taucs_io_read(handle,5,1,1,TAUCS_INT,&(L->n));
  taucs_io_read(handle,0,1,1,TAUCS_INT,&(L->n_sn));
  L->sn_struct = (int**)taucs_malloc((L->n_sn  )*sizeof(int*));
  L->sn_blocks = (taucs_datatype**)taucs_malloc((L->n_sn  )*sizeof(taucs_datatype*));
  L->up_blocks = (taucs_datatype**)taucs_malloc((L->n_sn  )*sizeof(taucs_datatype*));
  L->sn_size   = (int*) taucs_malloc((L->n_sn+1)*sizeof(int));
  L->sn_up_size   = (int*) taucs_malloc((L->n_sn+1)*sizeof(int));
  L->first_child = (int*) taucs_malloc((L->n_sn+1)*sizeof(int));
  L->next_child  = (int*) taucs_malloc((L->n_sn+1)*sizeof(int));
  taucs_io_read(handle,1,1,L->n_sn+1,TAUCS_INT,L->first_child);
  taucs_io_read(handle,2,1,L->n_sn+1,TAUCS_INT,L->next_child);
  taucs_io_read(handle,3,1,L->n_sn,TAUCS_INT,L->sn_size);
  taucs_io_read(handle,4,1,L->n_sn,TAUCS_INT,L->sn_up_size);
  /*  for (i=0; i<L->n_sn; i++) {
    L->sn_struct[i] = (int*)taucs_malloc(L->sn_up_size[i]*sizeof(int));
      taucs_io_read(handle,IO_BASE+i,1,L->sn_up_size[i],TAUCS_INT,L->sn_struct[i]);
      }*/
  
  for(i=0;i<L->n_sn;i++){
    L->sn_struct[i] = NULL;
    L->sn_blocks[i] = NULL;
    L->up_blocks[i] = NULL;
  }
  
  y = (taucs_datatype*) taucs_malloc((L->n) * sizeof(taucs_datatype));
  t = (taucs_datatype*) taucs_malloc((L->n) * sizeof(taucs_datatype));
  if (!y || !t) {
    taucs_free(y);
    taucs_free(t);
    taucs_printf("leftlooking_supernodal_solve_ldlt: out of memory\n");
    return -1;
  }

  for (i=0; i<L->n; i++) x[i] = b[i];

  recursive_supernodal_solve_l_ooc (L->n_sn,
				    TRUE,  /* this is the root */
				    handle,
				    L->n_sn,
				    L->first_child, L->next_child,
				    L->sn_struct,L->sn_size,L->sn_up_size,
				    y, x, t);

  recursive_supernodal_solve_lt_ooc(L->n_sn,
				    TRUE,  /* this is the root */
				    handle,
				    L->n_sn,
				    L->first_child, L->next_child,
				    L->sn_struct,L->sn_size,L->sn_up_size,
				    x, y, t);

  taucs_free(y);
  taucs_free(t);
  ooc_supernodal_factor_free(L);
  return 0;
}
/*******************************************************************/
/**                     OOC Panelize Factor                       **/
/*******************************************************************/

static double
recursive_smart_panelize_ooc_supernodes(int sn,       /* this supernode */
					int is_root,  /* is v the root? */
					double global_mem,
					int* curr_panel,
					int* sn_in_core,
					int* sn_to_panel_map,
					supernodal_factor_matrix_ooc_ldlt* L)
{
  int  child;
  double this_sn_mem = 0.0;
  double max_child_mem = 0.0;
  double curr_child_mem = 0.0;
  int* first_child   = L->first_child;
  int* next_child    = L->next_child;


  for (child = first_child[sn]; child != -1; child = next_child[child]) {
    if(!sn_in_core[child]){
      curr_child_mem = recursive_smart_panelize_ooc_supernodes(child,
							       FALSE,
							       global_mem,
							       curr_panel,
							       sn_in_core,
							       sn_to_panel_map,
							       L);
      if(curr_child_mem > max_child_mem) max_child_mem = curr_child_mem;
    }
  }

  if (!is_root){
    this_sn_mem = 
      (double) (L->sn_size)[sn] * (double) (L->sn_up_size)[sn] * (double) sizeof(taucs_datatype) 
      + (double) (L->sn_up_size)[sn] * (double) sizeof(int);
    if(max_child_mem+this_sn_mem < global_mem){
      sn_to_panel_map[sn] = *curr_panel;
      return (max_child_mem+this_sn_mem);
    } else {
      (*curr_panel)++;
      sn_to_panel_map[sn] = *curr_panel;
      return this_sn_mem;
    }
  } 

  /* reached only at the root */
  return 0.0;
}

static double
recursive_dumb_panelize_ooc_supernodes(int sn,       /* this supernode */
				       int is_root,  /* is v the root? */
				       int* curr_panel,
				       int* sn_in_core,
				       int* sn_to_panel_map,
				       supernodal_factor_matrix_ooc_ldlt* L)
{
  int  child;
  int* first_child   = L->first_child;
  int* next_child    = L->next_child;

  for (child = first_child[sn]; child != -1; child = next_child[child]) {
    if(!sn_in_core[child]) 
      (void) recursive_dumb_panelize_ooc_supernodes(child,
						    FALSE,
						    curr_panel,
						    sn_in_core,
						    sn_to_panel_map,
						    L);
  }

  if (!is_root){
    (*curr_panel)++;
    sn_to_panel_map[sn]=*curr_panel;
  }

  return 0.0;
}

static double
recursive_panelize_ooc_supernodes(int sn,       /* this supernode */
				  int is_root,  /* is v the root? */
				  double global_mem,
				  double avail_mem,
				  int* curr_panel,
				  int* sn_in_core,
				  int* sn_to_panel_map,
				  supernodal_factor_matrix_ooc_ldlt* L)
{
  int  child;
  double curr_avail_mem = avail_mem;
  double this_sn_mem = 0.0;
  
  for (child = L->first_child[sn]; child != -1; child = L->next_child[child]) {
    if(!sn_in_core[child]) 
      curr_avail_mem=recursive_panelize_ooc_supernodes(child,
						       FALSE,
						       global_mem,
						       curr_avail_mem,
						       curr_panel,
						       sn_in_core,
						       sn_to_panel_map,
						       L);
  }

  if (!is_root){
    this_sn_mem = (double)(L->sn_size)[sn] * 
									(double)(L->sn_up_size)[sn] * 
									sizeof(taucs_datatype) + 
									(double)(L->sn_up_size)[sn] * sizeof(int);
    if(curr_avail_mem-this_sn_mem>0.0){
      sn_to_panel_map[sn]=*curr_panel;
      curr_avail_mem -= this_sn_mem;
    } else {
      (*curr_panel)++;
      sn_to_panel_map[sn]=*curr_panel;
      curr_avail_mem = global_mem-this_sn_mem;
    }
  }
  return curr_avail_mem;
}

static void
recursive_find_parent(int sn,
											supernodal_panel* P,
											supernodal_factor_matrix_ooc_ldlt* L,
											int* first_leaf)
{
	int child;

	if ( L->first_child[sn] == -1 && *first_leaf == -1 )
		*first_leaf = sn;

	for ( child = L->first_child[sn];child != -1;child = L->next_child[child] )
	{
		recursive_find_parent(child,P,L,first_leaf);
		P->parent[child] = sn;
	}
}

static int
leftlooking_supernodal_update_ooc(int J,
																	int K,
																	int bitmap[],
																	taucs_datatype* dense_update_matrix,
																	taucs_io_handle* handle,
																	taucs_ccs_matrix* A,
																	supernodal_factor_matrix_ooc_ldlt* L)
{
	int i,j,ir;
	int sn_size_child = (L->sn_size)[K];
	int sn_up_size_child = (L->sn_up_size)[K];
	int exist_upd=0;
	int first_row=0;
	int row_count=0;
	int PK,M,N,LDA,LDB,LDC;

	for(i=sn_size_child;i<sn_up_size_child;i++) {
		/* We want to update only supernodes greater than J with using of
			 sort of indices */
		if(L->col_to_sn_map[L->sn_struct[K][i]]<J)
			continue;

		/* is this row index included in the columns of J? */
		if(L->col_to_sn_map[L->sn_struct[K][i]]==J) {
			if(!exist_upd) first_row = i;
			row_count++;
			exist_upd = 1;
		}
	}

	if(exist_upd) { 
		if(!(L->up_blocks)[K]) {
			(L->up_blocks)[K] = (taucs_datatype*)
				taucs_calloc((sn_up_size_child-sn_size_child)*sn_size_child,
										 sizeof(taucs_datatype));
			taucs_io_read(handle,IO_BASE+L->n_sn+2*K+1,
										sn_up_size_child-sn_size_child,sn_size_child ,
										TAUCS_CORE_DATATYPE,(L->up_blocks)[K]);
			/*taucs_printf("reading update of %d\n",K);*/
		}
		LDA = LDB = sn_up_size_child-sn_size_child;
		M  = sn_up_size_child - first_row ;
		LDC =  M;
		N  = row_count; 
		PK = sn_size_child; 	 
		
		taucs_gemm ("No Conjugate",
						"Conjugate",
						&M,&N,&PK,
						&taucs_one_const,
						&(L->up_blocks[K][first_row-sn_size_child]),&LDA,
						&(L->up_blocks[K][first_row-sn_size_child]),&LDB,
						&taucs_zero_const,
						dense_update_matrix,&LDC);

		
		/* NOTE: we allocate only for the time when we are
			dealing with the last panel*/
		/*
		if((L->sn_size)[up_sn] && !(L->sn_blocks)[up_sn])
			(L->sn_blocks)[up_sn] = (taucs_datatype*)
				taucs_calloc(((L->sn_size)[up_sn])*((L->sn_size)[up_sn]),
					sizeof(taucs_datatype));
		
		if(L->sn_up_size[up_sn]-L->sn_size[up_sn]>0 && !(L->up_blocks)[up_sn])
			(L->up_blocks)[up_sn] = (taucs_datatype*)
				taucs_calloc(((L->sn_up_size)[up_sn]-(L->sn_size)[up_sn]) *((L->sn_size)[up_sn]),
					sizeof(taucs_datatype));

		if(!L->sn_struct[up_sn]){
			L->sn_struct[up_sn] = (int*)taucs_malloc((L->sn_up_size)[up_sn]*sizeof(int));
				taucs_io_read(handle,IO_BASE+up_sn,
											1,(L->sn_up_size)[up_sn],
											TAUCS_INT,L->sn_struct[up_sn]);
		}*/

		for(i=0;i<L->sn_size[J];i++) {
			bitmap[L->sn_struct[J][i]]=i+1;
		}

		for(i=L->sn_size[J];i<L->sn_up_size[J];i++){
			bitmap[L->sn_struct[J][i]] = i-L->sn_size[J]+1;
		}
		 
		assert((double)row_count*(double)LDC < 2048.0*1024.0*1024.0);

		for(j=0;j<row_count;j++) {
			for(ir=j;ir<row_count;ir++){
				L->sn_blocks[J][(bitmap[L->sn_struct[K][first_row+j]]-1)*L->sn_size[J] +
												(bitmap[L->sn_struct[K][first_row+ir]]-1)] =
					taucs_sub(L->sn_blocks[J][(bitmap[L->sn_struct[K][first_row+j]]-1)*
										L->sn_size[J]+(bitmap[L->sn_struct[K][first_row+ir]]-1)] , 
										dense_update_matrix[j*LDC+ir]);

				/* to find overflows */
				assert((double)(bitmap[L->sn_struct[K][first_row+j]]-1)*(double)L->sn_size[J] < 2048.0*1024.0*1024.0);
			}
		}
		for(j=0;j<row_count;j++) {
			for(ir=row_count;ir<M;ir++){
				L->up_blocks[J][(bitmap[L->sn_struct[K][first_row+j]]-1)*
												(L->sn_up_size[J]-L->sn_size[J]) +
												(bitmap[L->sn_struct[K][ir+first_row]]-1)] =
					taucs_sub(L->up_blocks[J][(bitmap[L->sn_struct[K][first_row+j]]-1)*
										(L->sn_up_size[J]-L->sn_size[J])+
										(bitmap[L->sn_struct[K][ir+first_row]]-1)] ,
						dense_update_matrix[j*LDC+ir]);

				/* to find overflow */
				assert((double)(bitmap[L->sn_struct[K][first_row+j]]-1)*(double)(L->sn_up_size[J]-L->sn_size[J]) < 2048.0*1024.0*1024.0);
			} 
		}
		for(i=0;i<L->sn_up_size[J];i++) {
			bitmap[L->sn_struct[J][i]]=0;
		}
	}

	return exist_upd;
}

static int
leftlooking_supernodal_update_panel_ooc(int J,
																				int K,
																				int bitmap[],
																				taucs_datatype* dense_update_matrix,
																				taucs_io_handle* handle,
																				taucs_ccs_matrix* A,
																				supernodal_factor_matrix_ooc_ldlt* L,
																				supernodal_panel* P)
{
	int parent;
	int exist_upd = 0;

	/* update the first supernode in the panel to be updated */
	exist_upd = leftlooking_supernodal_update_ooc(J,K,bitmap,dense_update_matrix,
																								handle,A,L);

	/* update the supernode's parents in the panel */
	for (parent=P->parent[J];parent!=-1;parent=P->parent[parent]) {
		if ( P->sn_to_panel_map[J] == P->sn_to_panel_map[parent] ) {
			if (leftlooking_supernodal_update_ooc(parent,K,bitmap,dense_update_matrix,
																						handle,A,L) )
				exist_upd = 1;
		}
	}
	return exist_upd;
}



static void
recursive_leftlooking_supernodal_update_panel_ooc(int J,int K,
																									int bitmap[],
																									taucs_datatype* dense_update_matrix,
																									taucs_io_handle* handle,
																									taucs_ccs_matrix* A,
																									supernodal_factor_matrix_ooc_ldlt* L,
																									supernodal_panel* P)
{
	int  child;
	int sn_size_child = (L->sn_size)[K];
	int sn_up_size_child = (L->sn_up_size)[K];
	int exist_upd = 0;
	
	if(sn_up_size_child-sn_size_child>0) {
		
		if(!(L->sn_struct)[K]) {
			L->sn_struct[K] = (int*)taucs_malloc(sn_up_size_child*sizeof(int));
			taucs_io_read(handle,IO_BASE+K,1,sn_up_size_child,TAUCS_INT,L->sn_struct[K]);
			/*taucs_printf("reading struct of %d\n",K);*/
		}

		/* if end of update to up_sn or edge condition */
		/*	if( L->col_to_sn_map[L->sn_struct[K][i]]!=up_sn ||
					i==sn_up_size_child-1) {*/

		/* is this row index included in the columns of sn in the same panel? */
		/*	if(L->col_to_sn_map[L->sn_struct[K][i]]!=up_sn) {
					if(sn_to_panel_map[L->col_to_sn_map[L->sn_struct[K][i]]]==updated_panel) {
						up_sn = L->col_to_sn_map[L->sn_struct[K][i]];
						if(!exist_upd) first_row = i;
						row_count++;
						exist_upd = 1;
						if( i==sn_up_size_child-1) { */

		exist_upd = leftlooking_supernodal_update_panel_ooc(J,K,bitmap,dense_update_matrix,
																												handle,A,L,P);
							
				
	}
	
	/* free update sn from memory */
	taucs_free((L->up_blocks	 )[K]);
	(L->up_blocks 	)[K] = NULL;
	taucs_free( L->sn_struct[K]);
	L->sn_struct[K] = NULL;

	if(exist_upd &&
		 P->sn_to_panel_map[J]!=P->sn_to_panel_map[K]) {	  
	
		for (child = L->first_child[K]; child != -1; child = L->next_child[child]) {
				recursive_leftlooking_supernodal_update_panel_ooc(J,child,
						bitmap,
						dense_update_matrix,
						handle,A,L,P);
		}
	} 
}

static double
get_memory_overhead( int n )
{
	double memory_overhead;
	memory_overhead = 
    4.0*(double)(n*sizeof(int)) + /* integer vectors in L */
    3.0*(double)(n*sizeof(int)) + /* pointer arrays in L  */
    2.0*(double)(n*sizeof(int)) + /* integer vectors in program  */
    4.0*3.0*(double)(n*sizeof(int));  /* singlefile matrix arrays */
	return memory_overhead;
}

/* this method is called when a panel is full and allocated,
it factors each supernode in the panel, in postorder, and updates
and factors it. the update is done so that each supernode in the panel
is updated from each supernode child braught from the disk. */
static int
leftlooking_supernodal_factor_panel_ldlt_ooc(int postorder,
																						 int max_postorder,
																						 int* map,
																						 taucs_io_handle* handle,
																						 taucs_ccs_matrix* A,
																						 supernodal_factor_matrix_ooc_ldlt* L,
																						 supernodal_panel* P)
{ 
	int i;
	taucs_datatype* dense_update_matrix = NULL;

	if ( postorder == max_postorder ) /* no supernodes to factor */
		return 0;

	P->used = TRUE;

	if (!dense_update_matrix) 
		dense_update_matrix = (taucs_datatype*) 
			taucs_calloc(P->max_size,sizeof(taucs_datatype));

	/* we go through each of the supernodes in the panel,
		 in postorder, and factor them */
	for (i=postorder; i<max_postorder; i++) {
		
		int sn = P->postorder[i];
		int child;
		
    for (child = L->first_child[sn]; child != -1; child = L->next_child[child]) {
      if(P->sn_to_panel_map[sn]!=P->sn_to_panel_map[child])
				/* this method should update all supernodes in panel!! */
				recursive_leftlooking_supernodal_update_panel_ooc(sn,child,
							  map,
							  dense_update_matrix,
							  handle,A,L,P);
    }

    if (leftlooking_supernodal_front_factor(sn,
					    map,
					    A,
					    L)) {
      /* nonpositive pivot */
      return -1;
    }
    
		taucs_io_append_supernode(sn,handle,L);
    
		if ( P->sn_to_panel_map[sn] == P->sn_to_panel_map[P->parent[sn]] )
			leftlooking_supernodal_update_panel_ooc(P->parent[sn],sn,map,dense_update_matrix,
																							handle,A,L,P);

    ooc_supernodal_free(L,sn);
	}

	taucs_free(dense_update_matrix);

  return 0;
}


static int
recursive_leftlooking_supernodal_factor_panel_ldlt_ooc(int sn,    /* this supernode */
																											int father_sn,
																											int is_root,/* is sn the root?*/
																											int* map,
																											taucs_io_handle* handle,
																											taucs_ccs_matrix* A,
																											supernodal_factor_matrix_ooc_ldlt* L,
																											supernodal_panel* P,
																											int* first_sn_in_panel)
{
  int  child;
	int failed = 0;
	int sn_size = L->sn_size[sn];
	int sn_up_size = L->sn_up_size[sn];
	double this_sn_mem = 0.0;
  
  for (child = L->first_child[sn]; child != -1; child = L->next_child[child]) {
    if(P->sn_in_core[child]){
      if (recursive_read_L_cols(child,FALSE,handle,L)) {
				return -1;
      }
      if (recursive_leftlooking_supernodal_factor_ldlt(child,FALSE,map,A,L)) {
				/* failure */
				return -1;
      }
      if (recursive_append_L(child,FALSE,handle,L)) {
				return -1;
      }
		} else {
      if (recursive_leftlooking_supernodal_factor_panel_ldlt_ooc(child,
								sn,
								FALSE,
								map,
								handle,
								A,L,P,first_sn_in_panel)) {
				/* failure */
				return -1;
      }
		}
  }

	if (!is_root) { 

		/* check if this supernode can be inside current panel */
		this_sn_mem = (double)(sn_size*sn_up_size*sizeof(taucs_datatype)) + 
									(double)(sn_up_size*sizeof(int));
    if( P->avail_mem - this_sn_mem < 0.0 ) {
      /* this supernode cannot fit into the current panel,
				 factor the panel and start a new one with this sn */

			failed = leftlooking_supernodal_factor_panel_ldlt_ooc(*first_sn_in_panel,
																									P->n_postorder,
																									map,
																									handle,
																									A,L,P);

			if ( failed ) return -1;

			P->n_pn++;
			P->avail_mem = P->max_mem;
      P->max_size = 0;
			*first_sn_in_panel = P->n_postorder;

    } 

		/* this supernode is part of the currect panel.
			 add him to the panel and continue */
		P->sn_to_panel_map[sn] = P->n_pn;
    P->avail_mem -= this_sn_mem;
		P->max_size = max(sn_size*sn_up_size,P->max_size);

		/* put this supernode in its place in the postorder */
		P->postorder[P->n_postorder] = sn;
		P->n_postorder++;

		/* allocate the supernode */
		if(!(L->sn_blocks)[sn])
			(L->sn_blocks)[sn] = (taucs_datatype*) 
				taucs_calloc(sn_size*sn_size,sizeof(taucs_datatype));

		if(sn_up_size-sn_size>0 && !(L->up_blocks)[sn])
			(L->up_blocks)[sn] = (taucs_datatype*)
				taucs_calloc((sn_up_size-sn_size) * sn_size,sizeof(taucs_datatype));

		if(!L->sn_struct[sn]){
			L->sn_struct[sn] = (int*)taucs_malloc(sn_up_size*sizeof(int));
			taucs_io_read(handle,IO_BASE+sn,1,sn_up_size,TAUCS_INT,L->sn_struct[sn]);
			/*taucs_printf("reading struct of %d\n",sn);*/
		}
	} else {
		/* we need to factor the last panel */
		leftlooking_supernodal_factor_panel_ldlt_ooc(*first_sn_in_panel,
																								 P->n_postorder,
																								 map,
																								 handle,
																								 A,L,P);
	}

	return 0;
}	

  


int taucs_dtl(ooc_factor_ldlt)(taucs_ccs_matrix* A, 
			      taucs_io_handle* handle,
			      double memory)
{
  supernodal_factor_matrix_ooc_ldlt* L;
	supernodal_panel* P;
  int i,first_sn_in_panel = 0;
  int* map,* sn_parent;
  double wtime, ctime;
	double memory_overhead;
  
  /* compute fixed memory overhead */
  
  memory_overhead = get_memory_overhead(A->n);

  taucs_printf("\t\tOOC memory overhead bound %.0lf MB (out of %.0lf MB available)\n",
	       memory_overhead/1048576.0,memory/1048576.0);

  taucs_printf(">>> 1\n");

  if ( memory - memory_overhead < 
       2.0*(double)((A->n)*sizeof(taucs_datatype)) + 
       2.0*(double)((A->n)*sizeof(int)) ) {
    taucs_printf("\t\ttaucs_ccs_factor_ldlt_ll_ooc: not enough memory\n");
    return -1;
  }

  wtime = taucs_wtime();
  ctime = taucs_ctime();

	sn_parent		= (int*)taucs_calloc((A->n+1),sizeof(int));

  L = multifrontal_supernodal_create();
  
	taucs_io_append(handle,5,1,1,TAUCS_INT,&(A->n));

  taucs_ccs_ooc_symbolic_elimination(A,L,sn_parent,
				     TRUE /* sort row indices */,
				     TRUE /* return col_to_sn_map */,
				     (memory - memory_overhead)/3.0,
				     ooc_sn_struct_handler,handle);
  
  wtime = taucs_wtime()-wtime;
  ctime = taucs_ctime()-ctime;
  taucs_printf("\t\tSymbolic Analysis            = % 10.3f seconds (%.3f cpu)\n",
	       wtime,ctime);

  /* we now compute an exact memory overhead bound using n_sn */
  memory_overhead = get_memory_overhead(L->n_sn);
    
  taucs_printf("\t\tOOC actual memory overhead %.0lf MB (out of %.0lf MB available)\n",
	       memory_overhead/1048576.0,memory/1048576.0);
 
  wtime = taucs_wtime();
  ctime = taucs_ctime();

  taucs_io_append(handle,0,1,1,TAUCS_INT,&(L->n_sn));
  taucs_io_append(handle,1,1,L->n_sn+1,TAUCS_INT,L->first_child);
  taucs_io_append(handle,2,1,L->n_sn+1,TAUCS_INT,L->next_child);
  taucs_io_append(handle,3,1,L->n_sn,TAUCS_INT,L->sn_size);
  taucs_io_append(handle,4,1,L->n_sn,TAUCS_INT,L->sn_up_size);
  /*taucs_io_append(handle,5,1,1,TAUCS_INT,&(L->n));*/
  taucs_io_append(handle,6,1,1,TAUCS_INT,&(A->flags));

  wtime = taucs_wtime()-wtime;
  ctime = taucs_ctime()-ctime;
  taucs_printf("\t\tOOC Supernodal Left-Looking Prepare L = % 10.3f seconds (%.3f cpu)\n",
	       wtime,ctime);

  wtime = taucs_wtime();
  ctime = taucs_ctime();

	P = taucs_panel_create( (memory - memory_overhead)/3, L->n_sn, sn_parent );
	taucs_free(sn_parent);

  map  = (int*)taucs_malloc((A->n+1)*sizeof(int));

  for(i=0;i<L->n_sn;i++){
    (L->sn_blocks)[i] = NULL;
    (L->up_blocks)[i] = NULL;
    (L->sn_struct)[i] = NULL;
  }

  wtime = taucs_wtime();
  ctime = taucs_ctime();
  if(recursive_compute_supernodes_ll_in_core(L->n_sn,
					     TRUE,
					     P->max_mem,
					     P->sn_in_core,
					     L)<0.0) {
    ooc_supernodal_factor_free(L);
    taucs_free(map);
		taucs_panel_free(P);
    return -1;
  }
  
	
  wtime = taucs_wtime()-wtime;
  ctime = taucs_ctime()-ctime;
  taucs_printf("\t\tOOC Supernodal Left-Looking Scheduling = % 10.3f seconds (%.3f cpu)\n",
	       wtime,ctime);
  
  wtime = taucs_wtime();
  ctime = taucs_ctime();

	if (recursive_leftlooking_supernodal_factor_panel_ldlt_ooc(L->n_sn,
							    -1,  
							    TRUE, 
							    map,
							    handle,
							    A,L,P,&first_sn_in_panel)) {
    ooc_supernodal_factor_free(L);
    taucs_free(map);
    return -1;
  }
 

	taucs_printf("\t\tOOC Supernodal Left-Looking:\n");
	taucs_printf("\t\t\tread count           = %.0f \n",handle->nreads);
	taucs_printf("\t\t\tread volume (bytes)  = %.2e \n",handle->bytes_read);
	taucs_printf("\t\t\tread time (seconds)  = %.0f \n",handle->read_time);
	taucs_printf("\t\t\twrite count          = %.0f \n",handle->nwrites);
	taucs_printf("\t\t\twrite volume (bytes) = %.2e \n",handle->bytes_written);
	taucs_printf("\t\t\twrite time (seconds) = %.0f \n",handle->write_time);
	if (P->used)
		taucs_printf("\t\t\tnumber of panels %d\n",P->n_pn + 1);
	else
		taucs_printf("\t\t\tnumber of panels %d\n",P->n_pn);

  wtime = taucs_wtime()-wtime;
  ctime = taucs_ctime()-ctime;
  taucs_printf("\t\tOOC Supernodal Left-Looking LL^T = % 10.3f seconds (%.3f cpu)\n",
	       wtime,ctime);

  wtime = taucs_wtime();
  ctime = taucs_ctime();

	for(i=0;i<L->n_sn;i++)
		if ( P->sn_to_panel_map[i] != -1 )
			taucs_printf("sn\t%d\t,panel\t%d\n",i,P->sn_to_panel_map[i]);

  taucs_free(map);
	taucs_panel_free(P);
  ooc_supernodal_factor_free(L);
  
  wtime = taucs_wtime()-wtime;
  ctime = taucs_ctime()-ctime;
  
	taucs_printf("\t\tOOC Supernodal Left-Looking Cleanup = % 10.3f seconds (%.3f cpu)\n",
	       wtime,ctime);
  
  return 0;
}

#endif

/*************************************************************/
/* generic interfaces to user-callable routines              */
/*************************************************************/

#ifdef TAUCS_CORE_GENERAL

int taucs_ooc_factor_ldlt(taucs_ccs_matrix* A,
			 taucs_io_handle*  L,
			 double memory)
{
#ifdef TAUCS_DOUBLE_IN_BUILD
  if (A->flags & TAUCS_DOUBLE)
    return taucs_dooc_factor_ldlt(A,L,memory);
#endif

#ifdef TAUCS_SINGLE_IN_BUILD
  if (A->flags & TAUCS_SINGLE)
    return taucs_sooc_factor_ldlt(A,L,memory);
#endif

#ifdef TAUCS_DCOMPLEX_IN_BUILD
  if (A->flags & TAUCS_DCOMPLEX)
    return taucs_zooc_factor_ldlt(A,L,memory);
#endif

#ifdef TAUCS_SCOMPLEX_IN_BUILD
  if (A->flags & TAUCS_SCOMPLEX)
    return taucs_cooc_factor_ldlt(A,L,memory);
#endif


  assert(0);
  return -1;
}


/* 
   this generic function retrieves the data type
   from the file and uses it to call a specialized 
   function.
*/

int taucs_ooc_solve_ldlt (void* L /* actual type: taucs_io_handle* */,
			 void* x, void* b)
{
  int flags;

  taucs_io_read((taucs_io_handle*)L,
		6,1,1,TAUCS_INT,
		&flags);

  if (flags & TAUCS_DOUBLE)
    return taucs_dooc_solve_ldlt(L,x,b);


  assert(0);
  return -1;
}

#endif

/*************************************************************/
/* end of file                                               */
/*************************************************************/




