#include "include/problemGeneration.hpp"

/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian27pt( HYPRE_ParCSRMatrix  *A_ptr,
                       ProblemOptionsList &options     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = ceil(cbrt(options.n))*ceil(cbrt(num_procs));
   ny = ceil(cbrt(options.n))*ceil(cbrt(num_procs));
   nz = ceil(cbrt(options.n))*ceil(cbrt(num_procs));

   P  = round(cbrt(num_procs));
   Q  = round(cbrt(num_procs));
   R  = num_procs / (P*Q);

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("  Laplacian_27pt:\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(HYPRE_Real,  2, HYPRE_MEMORY_HOST);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
      values[0] = 8.0;
   if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
      values[0] = 2.0;
   values[1] = -1.;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}


/*----------------------------------------------------------------------
 * Build 7-point in 2D 
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParRotate7pt( HYPRE_ParCSRMatrix  *A_ptr,
                   ProblemOptionsList &options     )
{
   HYPRE_Int                 nx, ny;
   HYPRE_Int                 P, Q;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q;
   HYPRE_Real          eps, alpha;

   alpha =  33.75;
   eps = 0.001;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = ceil(sqrt(options.n))*ceil(sqrt(num_procs));
   ny = ceil(sqrt(options.n))*ceil(sqrt(num_procs));

   P  = round(sqrt(num_procs));
   Q  = num_procs / P;

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Rotate 7pt:\n");
      hypre_printf("    alpha = %f, eps = %f\n", alpha,eps);
      hypre_printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      hypre_printf("    (Px, Py) = (%d, %d)\n", P,  Q);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q from P,Q and myid */
   p = myid % P;
   q = ( myid - p)/P;

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/

   A = (HYPRE_ParCSRMatrix) GenerateRotate7pt(hypre_MPI_COMM_WORLD,
                                              nx, ny, P, Q, p, q, alpha, eps);

   *A_ptr = A;

   return (0);
}

static inline HYPRE_Int sign_double(HYPRE_Real a)
{
   return ( (0.0 < a) - (0.0 > a) );
}

/*----------------------------------------------------------------------
 * Build standard 5-point convection-diffusion operator in 2D
 * Parameters given in command line.
 * Operator:
 *
 *  -epsilon (Dxx + Dyy) + a Dx + b Dy = f
 *
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParDifConv( HYPRE_ParCSRMatrix  *A_ptr,
                 ProblemOptionsList &options,
                 ProblemInfo &probInfo)
{
   HYPRE_BigInt        nx, ny, nz;
   HYPRE_Int           P, Q, R;
   HYPRE_Real          cx, cy, cz;
   HYPRE_Real          ax, ay, az, atype;
   HYPRE_Real          hinx,hiny,hinz;
   HYPRE_Int           sign_prod;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int           num_procs, myid;
   HYPRE_Int           p, q, r;
   HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/


   switch (options.problem)
   {
      case -3:
         nx = ceil(sqrt(options.n))*ceil(sqrt(num_procs));
         ny = ceil(sqrt(options.n))*ceil(sqrt(num_procs));
         nz = 1;

         P  = round(sqrt(num_procs));
         Q  = num_procs / P;
         R  = 1;

         cx = options.epsilon;
         cy = options.epsilon;
         cz = 0.;

         ax = options.a;
         ay = options.b;
         az = 0.;

         break;

      // case -4:
      //    nx = ceil(cbrt(options.n))*ceil(cbrt(num_procs));
      //    ny = ceil(cbrt(options.n))*ceil(cbrt(num_procs));
      //    nz = ceil(cbrt(options.n))*ceil(cbrt(num_procs));

      //    P  = round(cbrt(num_procs));
      //    Q  = round(cbrt(num_procs));
      //    R  = num_procs / (P*Q);

      //    cx = 1.;
      //    cy = 1.;
      //    cz = 1.;

      //    ax = 1.;
      //    ay = 1.;
      //    az = 1.;

         break;

      default:
         if (myid == 0) printf("Unknown problem.\n");
         MPI_Finalize();
         return 1;
   }

   atype = 3;

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Convection-Diffusion: \n");
      hypre_printf("    -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f\n");
      hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      hypre_printf("    (ax, ay, az) = (%f, %f, %f)\n\n", ax, ay, az);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   hinx = 1./(HYPRE_Real)(nx+1);
   hiny = 1./(HYPRE_Real)(ny+1);
   hinz = 1./(HYPRE_Real)(nz+1);

   HYPRE_BigInt *nx_part;
   HYPRE_BigInt *ny_part;
   hypre_GeneratePartitioning(nx,P,&nx_part);
   hypre_GeneratePartitioning(ny,Q,&ny_part);
   HYPRE_Real nx_local =(HYPRE_Int)(nx_part[p+1] - nx_part[p]);
   HYPRE_Real ny_local =(HYPRE_Int)(ny_part[q+1] - ny_part[q]);

   HYPRE_Int i;
   HYPRE_Real x_step = 1.0/(nx+1);
   HYPRE_Real y_step = 1.0/(ny+1);
   for (i = 0; i < nx_local; i++)
      probInfo.x_coords.push_back((nx_part[p] + i + 1)*x_step);
   for (i = 0; i < ny_local; i++)
      probInfo.y_coords.push_back((ny_part[q] + i + 1)*y_step);

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/
   /* values[7]:
    *    [0]: center
    *    [1]: X-
    *    [2]: Y-
    *    [3]: Z-
    *    [4]: X+
    *    [5]: Y+
    *    [6]: Z+
    */
   values = hypre_CTAlloc(HYPRE_Real,  7, HYPRE_MEMORY_HOST);

   values[0] = 0.;

   if (0 == atype) /* forward scheme for conv */
   {
      values[1] = -cx/(hinx*hinx);
      values[2] = -cy/(hiny*hiny);
      values[3] = -cz/(hinz*hinz);
      values[4] = -cx/(hinx*hinx) + ax/hinx;
      values[5] = -cy/(hiny*hiny) + ay/hiny;
      values[6] = -cz/(hinz*hinz) + az/hinz;

      if (nx > 1)
      {
         values[0] += 2.0*cx/(hinx*hinx) - 1.*ax/hinx;
      }
      if (ny > 1)
      {
         values[0] += 2.0*cy/(hiny*hiny) - 1.*ay/hiny;
      }
      if (nz > 1)
      {
         values[0] += 2.0*cz/(hinz*hinz) - 1.*az/hinz;
      }
   }
   else if (1 == atype) /* backward scheme for conv */
   {
      values[1] = -cx/(hinx*hinx) - ax/hinx;
      values[2] = -cy/(hiny*hiny) - ay/hiny;
      values[3] = -cz/(hinz*hinz) - az/hinz;
      values[4] = -cx/(hinx*hinx);
      values[5] = -cy/(hiny*hiny);
      values[6] = -cz/(hinz*hinz);

      if (nx > 1)
      {
         values[0] += 2.0*cx/(hinx*hinx) + 1.*ax/hinx;
      }
      if (ny > 1)
      {
         values[0] += 2.0*cy/(hiny*hiny) + 1.*ay/hiny;
      }
      if (nz > 1)
      {
         values[0] += 2.0*cz/(hinz*hinz) + 1.*az/hinz;
      }
   }
   else if (3 == atype) /* upwind scheme */
   {
      sign_prod = sign_double(cx) * sign_double(ax);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values[1] = -cx/(hinx*hinx) - ax/hinx;
         values[4] = -cx/(hinx*hinx);
         if (nx > 1)
         {
            values[0] += 2.0*cx/(hinx*hinx) + 1.*ax/hinx;
         }
      }
      else /* diff sign use forward scheme */
      {
         values[1] = -cx/(hinx*hinx);
         values[4] = -cx/(hinx*hinx) + ax/hinx;
         if (nx > 1)
         {
            values[0] += 2.0*cx/(hinx*hinx) - 1.*ax/hinx;
         }
      }

      sign_prod = sign_double(cy) * sign_double(ay);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values[2] = -cy/(hiny*hiny) - ay/hiny;
         values[5] = -cy/(hiny*hiny);
         if (ny > 1)
         {
            values[0] += 2.0*cy/(hiny*hiny) + 1.*ay/hiny;
         }
      }
      else /* diff sign use forward scheme */
      {
         values[2] = -cy/(hiny*hiny);
         values[5] = -cy/(hiny*hiny) + ay/hiny;
         if (ny > 1)
         {
            values[0] += 2.0*cy/(hiny*hiny) - 1.*ay/hiny;
         }
      }

      sign_prod = sign_double(cz) * sign_double(az);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values[3] = -cz/(hinz*hinz) - az/hinz;
         values[6] = -cz/(hinz*hinz);
         if (nz > 1)
         {
            values[0] += 2.0*cz/(hinz*hinz) + 1.*az/hinz;
         }
      }
      else /* diff sign use forward scheme */
      {
         values[3] = -cz/(hinz*hinz);
         values[6] = -cz/(hinz*hinz) + az/hinz;
         if (nz > 1)
         {
            values[0] += 2.0*cz/(hinz*hinz) - 1.*az/hinz;
         }
      }
   }
   else /* centered difference scheme */
   {
      values[1] = -cx/(hinx*hinx) - ax/(2.*hinx);
      values[2] = -cy/(hiny*hiny) - ay/(2.*hiny);
      values[3] = -cz/(hinz*hinz) - az/(2.*hinz);
      values[4] = -cx/(hinx*hinx) + ax/(2.*hinx);
      values[5] = -cy/(hiny*hiny) + ay/(2.*hiny);
      values[6] = -cz/(hinz*hinz) + az/(2.*hinz);

      if (nx > 1)
      {
         values[0] += 2.0*cx/(hinx*hinx);
      }
      if (ny > 1)
      {
         values[0] += 2.0*cy/(hiny*hiny);
      }
      if (nz > 1)
      {
         values[0] += 2.0*cz/(hinz*hinz);
      }
   }

   A = (HYPRE_ParCSRMatrix) GenerateDifConv(hypre_MPI_COMM_WORLD,
                                            nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}


/*----------------------------------------------------------------------
 * Build 1D Poisson (tridiagonal system)
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
Tridiagonal( HYPRE_ParCSRMatrix  *A_ptr,
                   ProblemOptionsList &options     )
{
   HYPRE_Int                 nx, ny;
   HYPRE_Int                 P, Q;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q;
   HYPRE_Real          eps, alpha;

   alpha =  0.00;
   eps = 0.00;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = options.n;
   ny = 1;

   P  = num_procs;
   Q  = 1;

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Tridiagonal:\n");
      hypre_printf("    n = %d\n", nx);
      hypre_printf("    P = %d\n", P);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q from P,Q and myid */
   p = myid;
   q = 0;

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/

   A = (HYPRE_ParCSRMatrix) GenerateRotate7pt(hypre_MPI_COMM_WORLD,
                                              nx, ny, P, Q, p, q, alpha, eps);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 5-point laplacian in 2D 
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian5pt( HYPRE_ParCSRMatrix  *A_ptr,
                 ProblemOptionsList &options)
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;
   HYPRE_Real          cx, cy, cz;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   HYPRE_Real         *values;
   HYPRE_Real         *mtrx;

   
   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = ceil(sqrt(options.n))*ceil(sqrt(num_procs));
   ny = ceil(sqrt(options.n))*ceil(sqrt(num_procs));
   nz = 1;

   P  = round(sqrt(num_procs));
   Q  = num_procs / P;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 0.;

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("  Laplacian:\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(HYPRE_Real,  4, HYPRE_MEMORY_HOST);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz;
   }

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian(hypre_MPI_COMM_WORLD, 
                                                 nx, ny, nz, P, Q, R, p, q, r, values);
   

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build grid aligned anisotropic 5-point laplacian in 2D 
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParGridAlignedAnisotropic( HYPRE_ParCSRMatrix  *A_ptr,
                 ProblemOptionsList &options,
                 ProblemInfo &probInfo)
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;
   HYPRE_Real          cx, cy, cz;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   HYPRE_Real         *values;
   HYPRE_Real         *mtrx;

   
   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = ceil(sqrt(options.n))*ceil(sqrt(num_procs));
   ny = ceil(sqrt(options.n))*ceil(sqrt(num_procs));
   nz = 1;

   P  = round(sqrt(num_procs));
   Q  = num_procs / P;
   R  = 1;

   cx = 1.;
   cy = 0.001;
   cz = 0.;

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("  Laplacian:\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = 0;

   HYPRE_BigInt *nx_part;
   HYPRE_BigInt *ny_part;
   hypre_GeneratePartitioning(nx,P,&nx_part);
   hypre_GeneratePartitioning(ny,Q,&ny_part);
   HYPRE_Real nx_local =(HYPRE_Int)(nx_part[p+1] - nx_part[p]);
   HYPRE_Real ny_local =(HYPRE_Int)(ny_part[q+1] - ny_part[q]);

   HYPRE_Int i;
   HYPRE_Real x_step = 1.0/(nx+1);
   HYPRE_Real y_step = 1.0/(ny+1);
   for (i = 0; i < nx_local; i++)
      probInfo.x_coords.push_back((nx_part[p] + i + 1)*x_step);
   for (i = 0; i < ny_local; i++)
      probInfo.y_coords.push_back((ny_part[q] + i + 1)*y_step);
   
   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(HYPRE_Real,  4, HYPRE_MEMORY_HOST);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz;
   }

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian(hypre_MPI_COMM_WORLD, 
                                                 nx, ny, nz, P, Q, R, p, q, r, values);
   

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

