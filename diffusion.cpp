#include "include/problemGeneration.hpp"

double PointSource(const Vector &pt, double t);
double JumpCoeffScalarFunc(const Vector &pt, double t);
double FourRegionScalar(const Vector &pt, double t);
void FourRegionMatrix(const Vector &pt, double t, DenseMatrix &mat);

void GetMatrixDiffusion(HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out, ProblemOptionsList &options)
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Get the mesh
   ParMesh *pmesh = GetMesh(options);

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (options.order > 0)
   {
      fec = new H1_FECollection(options.order, options.dim);
   }
   else if (pmesh->GetNodes())
   {
      fec = pmesh->GetNodes()->OwnFEC();
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(options.order = 1, options.dim);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(fespace);
   std::vector<Coefficient*> lf_coeffs;
   switch (options.rhs)
   {
      case -1:
      {
         lf_coeffs.push_back( new ConstantCoefficient(0.0) );
         b->AddDomainIntegrator(new DomainLFIntegrator(*(lf_coeffs[0])));
         break;
      }
      case 0:
      {
         lf_coeffs.push_back( new ConstantCoefficient(0.0) );
         b->AddDomainIntegrator(new DomainLFIntegrator(*(lf_coeffs[0])));
         break;
      }
      case 1:
      {
         lf_coeffs.push_back( new ConstantCoefficient(1.0) );
         b->AddDomainIntegrator(new DomainLFIntegrator(*(lf_coeffs[0])));
         break;
      }
      case 2:
      {
         lf_coeffs.push_back( new FunctionCoefficient(PointSource) );
         b->AddDomainIntegrator(new DomainLFIntegrator(*(lf_coeffs[0])));
         break;
      }
      case 3:
      {
         lf_coeffs.push_back( new ConstantCoefficient(0.0) );
         b->AddDomainIntegrator(new DomainLFIntegrator(*(lf_coeffs[0])));
         break;               
      }
      default:
      {
         if (myid == 0) printf("Unknown rhs.\n");
         MPI_Finalize();
         exit(1);
      }
   }
   b->Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   std::vector<Coefficient*> blf_coeffs;
   std::vector<MatrixCoefficient*> blf_matrix_coeffs;
   switch (options.problem)
   {
      case 0:
      {
         blf_coeffs.push_back( new ConstantCoefficient(1.0) );
         a->AddDomainIntegrator(new DiffusionIntegrator(*(blf_coeffs[0])));
         break;
      }
      case 1:
      {
         blf_coeffs.push_back( new FunctionCoefficient(JumpCoeffScalarFunc) );
         a->AddDomainIntegrator(new DiffusionIntegrator(*(blf_coeffs[0])));
         break;
      }
      case 2:
      {
         if (options.dim == 2)
         {
            DenseMatrix mat(2, 2);
            double theta = M_PI/8.0;
            double c = cos(theta);
            double s = sin(theta);
            double ep = 0.001;
            double data[4] = {c*c + ep*s*s, -c*s + ep*c*s, -c*s + ep*c*s, s*s + ep*c*c};
            mat = data;
            blf_matrix_coeffs.push_back( new MatrixConstantCoefficient(mat) );
         }
         else if (options.dim == 3)
         {
            DenseMatrix mat(3,3);
            double theta = 3.0/16.0*M_PI;
            double c = cos(theta);
            double s = sin(theta);
            double ep = 0.001;
            double data[9] = {c*c + ep*s*s, 
                              c*c*s - ep*c*c*s, 
                              c*s*s - ep*c*s*s, 
                              c*c*s + ep*c*c*c*c + ep*s*s,
                              c*c*s*s + ep*c*c*c*c + ep*s*s,
                              c*s*s*s + ep*c*c*c*s - ep*s*c, 
                              s*s*c - ep*c*s*s, 
                              c*s*s*s + ep*c*c*c*s - ep*c*s, 
                              s*s*s*s + ep*c*c*s*s + ep*c*c};
            mat = data;
            blf_matrix_coeffs.push_back( new MatrixConstantCoefficient(mat) );
         }
         a->AddDomainIntegrator(new DiffusionIntegrator(*(blf_matrix_coeffs[0])));
         break;
      }
      case 3:
      {
         // Four region problem from Compatible Relaxation and Coarsening in Algebraic Multigrid paper
         blf_coeffs.push_back( new FunctionCoefficient(FourRegionScalar) );
         blf_matrix_coeffs.push_back( new MatrixFunctionCoefficient(2, FourRegionMatrix) );
         a->AddDomainIntegrator(new DiffusionIntegrator(*(blf_matrix_coeffs[0])));
         a->AddDomainIntegrator(new MassIntegrator(*(blf_coeffs[0])));

         break;
      }
      default:
      {
         if (myid == 0) printf("Unknown problem.\n");
         MPI_Finalize();
         exit(1);         
      }
   }

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (options.static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   MFEMtoHYPRE(A, B, X, A_out, B_out, X_out);

   // 17. Free the used memory.
   // delete a;
   // delete b;
   // delete fespace;
   // if (options.order > 0) { delete fec; }
   // delete pmesh;
}

double PointSource(const Vector &pt, double t)
{    
   int dim = pt.Size();
   double x = pt(0), y = pt(1), z = 0.3;
   if (dim == 3)
   {
      z = pt(2);
   }
   if (x >= 0.2 && x <= 0.4 && y >= 0.2 && y <= 0.4 && z >= 0.2 && z <= 0.4) return 1.0;
   else return 0.0;
}

double JumpCoeffScalarFunc(const Vector &pt, double t)
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   double f = 1.0;

   int dim = pt.Size();
   double x = pt(0), y = pt(1), z = 0.0;
   double multiplier = 1.0;
   if (dim == 3)
   {
      z = pt(2);
      multiplier = cbrt(num_procs);
      int x_region = floor(x*multiplier);
      int y_region = floor(y*multiplier);
      int z_region = floor(z*multiplier);
      if (( (x_region % 2) == (y_region % 2) && !(z_region % 2) ) || ( (x_region % 2) != (y_region % 2) && (z_region % 2) ))  f = 1000.0;
      else f = 1.0;
   }
   else
   {
      multiplier = sqrt(num_procs);
      int x_region = floor(x*multiplier);
      int y_region = floor(y*multiplier);
      if ( (x_region % 2) == (y_region % 2) ) f = 1000.0;
      else f = 1.0;
   }

   return f;
}

double FourRegionScalar(const Vector &pt, double t)
{
   double f = 1.0;

   int dim = pt.Size();
   double x = pt(0), y = pt(1);//, z = 0.0;

   if (dim == 2)
   {
      if (0.0 <= x && x <= 0.5 && 0.0 <= y && y <= 0.5)
      {
         f = 10000;
      }
      else if (0.5 < x && x <= 1.0 && 0.0 <= y && y <= 0.5)
      {
         f = 0.0;
      }
      else if (0.0 <= x && x <= 0.5 && 0.5 < y && y <= 1.0)
      {
         f = 0.0;
      }
      else if (0.5 < x && x <= 1.0 && 0.5 < y && y <= 1.0)
      {
         f = 0.0;
      }
   }

   return f;
}

void FourRegionMatrix(const Vector &pt, double t, DenseMatrix &mat)
{
   double data[4] = {1.0, 0.0, 0.0, 1.0};

   int dim = pt.Size();
   double x = pt(0), y = pt(1);//, z = 0.0;

   if (dim == 2)
   {
      if (0.0 <= x && x <= 0.5 && 0.0 <= y && y <= 0.5)
      {

      }
      else if (0.5 < x && x <= 1.0 && 0.0 <= y && y <= 0.5)
      {

      }
      else if (0.0 <= x && x <= 0.5 && 0.5 < y && y <= 1.0)
      {
         data[3] = 0.01;
      }
      else if (0.5 < x && x <= 1.0 && 0.5 < y && y <= 1.0)
      {
         double theta = 0.5*M_PI;
         double c = cos(theta);
         double s = sin(theta);
         double ep = 0.001;
         data[0] = c*c + ep*s*s;
         data[1] = -c*s + ep*c*s;
         data[2] = -c*s + ep*c*s;
         data[3] = s*s + ep*c*c;         
      }
   }

   mat = data;
}