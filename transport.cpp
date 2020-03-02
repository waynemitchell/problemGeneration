#include "include/problemGeneration.hpp"

using namespace std;
using namespace mfem;


class CoefficientWithState : public Coefficient
{
protected:
    double (*Function)(const Vector &, const Vector &);

public:
    /// Define a time-independent coefficient from a C-function
    CoefficientWithState(double (*f)(const Vector &, const Vector &))
    {
      Function = f;
    }

    /// Evaluate coefficient
    virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip) {
        double x[3];
        Vector transip(x, 3);
        T.Transform(ip, transip);
        return ((*Function)(state_, transip));
    }
    
    void SetState(Vector state) { 
        state_.SetSize(state.Size());
        state_ = state;
    }

private:
   Vector state_;
};


// freq used in definition of psi_function2(omega,x)
#define PI 3.14159265358979323846
double freq = 1.52;
double sigma_s_function(const Vector &x);
double sigma_t_function(const Vector &x);
double psi_function2(const Vector &omega, const Vector &x);
double Q_function2(const Vector &x);
double inflow_function2(const Vector &x);


void GetMatrixTransport(HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out, ProblemOptionsList &options, ProblemInfo &probInfo)
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   if (myid == 0) cout << "Generating transport matrix ..." << endl;

   // Get the mesh
   probInfo.pmesh = GetMesh(options);

   int blocksize = (options.order+1)*(options.order+1);

   // Define finite element space
   int basis_type = 1; // DG finite element basis type. 0 for G-Leg, 1 for G-Lob
   probInfo.fec = new DG_FECollection(options.order, options.dim, basis_type);
   ParFiniteElementSpace *pfes = new ParFiniteElementSpace(probInfo.pmesh, probInfo.fec);

   // Generate a grid function for storing solutions later
   probInfo.x = new ParGridFunction(pfes);
   *(probInfo.x) = 0.0;

   /* Define angle of flow, coefficients and integrators */
   double theta = 3*PI/16.0;
   std::vector<double> omega0 {cos(theta), sin(theta)};
   Vector omega(&omega0[0],2);

   probInfo.blf_coeffs.push_back( new FunctionCoefficient(sigma_t_function) );
   probInfo.blf_vector_coeffs.push_back( new VectorConstantCoefficient(omega) );

   probInfo.lf_coeffs.push_back( new FunctionCoefficient(inflow_function2) );
   probInfo.lf_coeffs.push_back( new FunctionCoefficient(Q_function2) );

   /* Set up the bilinear form for this angle */
   probInfo.a = new ParBilinearForm(pfes);
   probInfo.a->AddDomainIntegrator(new MassIntegrator(*(probInfo.blf_coeffs[0])));
   probInfo.a->AddDomainIntegrator(new TransposeIntegrator(new ConvectionIntegrator(*(probInfo.blf_vector_coeffs[0]), -1.0)));
   probInfo.a->AddInteriorFaceIntegrator(new DGTraceIntegrator(*(probInfo.blf_vector_coeffs[0]), 1.0, 0.5));  // Interior face integrators
   probInfo.a->AddBdrFaceIntegrator(new DGTraceIntegrator(*(probInfo.blf_vector_coeffs[0]), 1.0, 0.5));       // Boundary face integrators
   probInfo.a->Assemble();
   probInfo.a->Finalize();

   /* Form the right-hand side */
   probInfo.b = new ParLinearForm(pfes);
   probInfo.b->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(*(probInfo.lf_coeffs[0]), *(probInfo.blf_vector_coeffs[0]), -1.0, -0.5));
   probInfo.b->AddDomainIntegrator(new DomainLFIntegrator(*(probInfo.lf_coeffs[1])));
   probInfo.b->Assemble();

   // Generate matrices and vectors from assembled forms
   HypreParMatrix *A = probInfo.a->ParallelAssemble();
   HypreParVector *B = probInfo.b->ParallelAssemble();
   Vector X(pfes->GetVSize());

   // Do block scaling
   HypreParMatrix A_s;
   HypreParVector B_s;
   BlockInvScal(A, &A_s, B, &B_s, blocksize, 1);

   delete A;
   delete B;

   // Convert mfem to hypre
   MFEMtoHYPRE(A_s, B_s, X, A_out, B_out, X_out);
   B_s.SetData(NULL);
}

////////////////////////////////////////////////////////////////////////////////

double psi_function2(const Vector &omega, const Vector &x) {
   double x1 = x(0);
   double x2 = x(1);
   double psi = .5 * (x1*x1 + x2*x2 + 1.) + std::cos(freq*(x1+x2));
   psi = psi * (omega(0)*omega(0) + omega(1));
   return psi;
}

double sigma_s_function(const Vector &x) {
    return 0.;
}

double sigma_t_function(const Vector &x) {
   double x1 = x(0);
   double x2 = x(1);
   // double val_abs = x1*x2 + x1*x1 + 1.;
   double val_abs = x1*x2 + x1*x1;
   double sig_s = sigma_s_function(x);
   return val_abs + sig_s;
}

double Q_function2(const Vector &x) {
   double theta = 3*PI/16.0;
   std::vector<double> omega0 {cos(theta), sin(theta)};
   Vector omega(&omega0[0],2);
   double x1 = x(0);
   double x2 = x(1);
   double sig = sigma_t_function(x);
   double val_sin = freq * std::sin(freq*(x1+x2));
   double psi_dx_dot_v = omega(0)*(x1-val_sin) + omega(1)*(x2-val_sin);
   psi_dx_dot_v = psi_dx_dot_v * (omega(0)*omega(0) + omega(1));
   double psi = psi_function2(omega, x);
   return psi_dx_dot_v + sig * psi;
}

double inflow_function2(const Vector &x) {
   double theta = 3*PI/16.0;
   std::vector<double> omega0 {cos(theta), sin(theta)};
   Vector omega(&omega0[0],2);
   double psi = psi_function2(omega, x);
   return psi;
}
