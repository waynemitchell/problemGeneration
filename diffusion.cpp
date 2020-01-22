#include "include/problemGeneration.hpp"

using namespace std;
using namespace mfem;

double PointSource(const Vector &pt, double t);
double JumpCoeffScalarFunc(const Vector &pt, double t);
double FourRegionScalar(const Vector &pt, double t);
void FourRegionMatrix(const Vector &pt, double t, DenseMatrix &mat);

void GetMatrixDiffusion(HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out, ProblemOptionsList &options, ProblemInfo &probInfo)
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   if (myid == 0) cout << "Generating diffusion matrix ..." << endl;

   // Get the mesh
   probInfo.pmesh = GetMesh(options);

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   if (options.order > 0)
   {
      probInfo.fec = new H1_FECollection(options.order, options.dim);
   }
   else if (probInfo.pmesh->GetNodes())
   {
      probInfo.fec = probInfo.pmesh->GetNodes()->OwnFEC();
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << probInfo.fec->Name() << endl;
      }
   }
   else
   {
      probInfo.fec = new H1_FECollection(options.order = 1, options.dim);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(probInfo.pmesh, probInfo.fec);
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
   if (probInfo.pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(probInfo.pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   probInfo.b = new ParLinearForm(fespace);
   switch (options.rhs)
   {
      case -1:
      {
         probInfo.lf_coeffs.push_back( new ConstantCoefficient(0.0) );
         probInfo.b->AddDomainIntegrator(new DomainLFIntegrator(*(probInfo.lf_coeffs[0])));
         break;
      }
      case 0:
      {
         probInfo.lf_coeffs.push_back( new ConstantCoefficient(0.0) );
         probInfo.b->AddDomainIntegrator(new DomainLFIntegrator(*(probInfo.lf_coeffs[0])));
         break;
      }
      case 1:
      {
         probInfo.lf_coeffs.push_back( new ConstantCoefficient(1.0) );
         probInfo.b->AddDomainIntegrator(new DomainLFIntegrator(*(probInfo.lf_coeffs[0])));
         break;
      }
      case 2:
      {
         probInfo.lf_coeffs.push_back( new FunctionCoefficient(PointSource) );
         probInfo.b->AddDomainIntegrator(new DomainLFIntegrator(*(probInfo.lf_coeffs[0])));
         break;
      }
      default:
      {
         if (myid == 0) printf("Unknown rhs.\n");
         MPI_Finalize();
         exit(1);
      }
   }
   probInfo.b->Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   probInfo.x = new ParGridFunction(fespace);
   *(probInfo.x) = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   probInfo.a = new ParBilinearForm(fespace);
   switch (options.problem)
   {
      case 0: // Isotropic diffusion
      {
         probInfo.blf_coeffs.push_back( new ConstantCoefficient(1.0) );
         probInfo.a->AddDomainIntegrator(new DiffusionIntegrator(*(probInfo.blf_coeffs[0])));
         break;
      }
      case 1: // Jump Coefficient diffusion
      {
         probInfo.blf_coeffs.push_back( new FunctionCoefficient(JumpCoeffScalarFunc) );
         probInfo.a->AddDomainIntegrator(new DiffusionIntegrator(*(probInfo.blf_coeffs[0])));
         break;
      }
      case 2: // Non grid aligned anisotropic diffusion (2D and 3D)
      {
         if (options.dim == 2)
         {
            DenseMatrix mat(2, 2);
            double theta = 3.0*M_PI/16.0;
            double c = cos(theta);
            double s = sin(theta);
            double ep = 0.001;
            double data[4] = {c*c + ep*s*s, -c*s + ep*c*s, -c*s + ep*c*s, s*s + ep*c*c};
            mat = data;
            probInfo.blf_matrix_coeffs.push_back( new MatrixConstantCoefficient(mat) );
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
            probInfo.blf_matrix_coeffs.push_back( new MatrixConstantCoefficient(mat) );
         }
         probInfo.a->AddDomainIntegrator(new DiffusionIntegrator(*(probInfo.blf_matrix_coeffs[0])));
         break;
      }
      case 3: // Four region problem from Compatible Relaxation and Coarsening in Algebraic Multigrid paper
      {         
         probInfo.blf_coeffs.push_back( new FunctionCoefficient(FourRegionScalar) );
         probInfo.blf_matrix_coeffs.push_back( new MatrixFunctionCoefficient(2, FourRegionMatrix) );
         probInfo.a->AddDomainIntegrator(new DiffusionIntegrator(*(probInfo.blf_matrix_coeffs[0])));
         probInfo.a->AddDomainIntegrator(new MassIntegrator(*(probInfo.blf_coeffs[0])));

         break;
      }
      case 4: // Grid aligned anisotropic diffusion (2D and 3D)
      {
         if (options.dim == 2)
         {
            DenseMatrix mat(2, 2);
            double theta = 0.0;
            double c = cos(theta);
            double s = sin(theta);
            double ep = 0.001;
            double data[4] = {c*c + ep*s*s, -c*s + ep*c*s, -c*s + ep*c*s, s*s + ep*c*c};
            mat = data;
            probInfo.blf_matrix_coeffs.push_back( new MatrixConstantCoefficient(mat) );
         }
         else if (options.dim == 3)
         {
            DenseMatrix mat(3,3);
            double theta = 0.0;
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
            probInfo.blf_matrix_coeffs.push_back( new MatrixConstantCoefficient(mat) );
         }
         probInfo.a->AddDomainIntegrator(new DiffusionIntegrator(*(probInfo.blf_matrix_coeffs[0])));
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
   if (options.static_cond) { probInfo.a->EnableStaticCondensation(); }
   probInfo.a->Assemble();

   HypreParMatrix A;
   Vector B, X;
   probInfo.a->FormLinearSystem(ess_tdof_list, *(probInfo.x), *(probInfo.b), A, X, B);

   MFEMtoHYPRE(A, B, X, A_out, B_out, X_out);
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
   
   double cross_pt[2] = {0.0,0.0};

   int dim = pt.Size();
   double x = pt(0), y = pt(1);//, z = 0.0;

   if (dim == 2)
   {
      if (x <= cross_pt[0] && y <= cross_pt[1])
      {
         f = 10000;
      }
      else if (cross_pt[0] < x && y <= cross_pt[1])
      {
         f = 0.0;
      }
      else if (x <= cross_pt[0] && cross_pt[1] < y)
      {
         f = 0.0;
      }
      else if (cross_pt[0] < x && cross_pt[1] < y)
      {
         f = 0.0;
      }
   }

   return f;
}

void FourRegionMatrix(const Vector &pt, double t, DenseMatrix &mat)
{
   double data[4] = {1.0, 0.0, 0.0, 1.0};

   double cross_pt[2] = {0.0,0.0};

   int dim = pt.Size();
   double x = pt(0), y = pt(1);//, z = 0.0;

   if (dim == 2)
   {
      if (x <= cross_pt[0] && y <= cross_pt[1])
      {

      }
      else if (cross_pt[0] < x && y <= cross_pt[1])
      {

      }
      else if (x <= cross_pt[0] && cross_pt[1] < y)
      {
         data[3] = 0.01;
      }
      else if (cross_pt[0] < x && cross_pt[1] < y)
      {
         double theta = 3.0/16.0*M_PI;
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
