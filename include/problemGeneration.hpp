#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <chrono>

struct ProblemOptionsList : public mfem::OptionsParser
{
   // Default options list
   int problem = 0;
   int rhs = 1;
   int random_initial = 0;
   int dirichlet_bcs = 1;
   int n = 10000;
   const char *mesh = "square";
   int mesh_partitioning = 0;
   int order = 1;
   bool static_cond = false;
   bool visualization = 0;
   const char *dump_problem_to_dir = "*";
   const char *read_problem_from_dir = "*";
   bool read_matrix_market = false;
   bool read_exact_solution = false;
   int dim = 2;

   ProblemOptionsList(int argc, char *argv[]);
};

struct ProblemInfo
{
   // Store coefficients used for linear and bilinear forms
   std::vector<mfem::Coefficient*> blf_coeffs;
   std::vector<mfem::VectorCoefficient*> blf_vector_coeffs;
   std::vector<mfem::MatrixCoefficient*> blf_matrix_coeffs;

   std::vector<mfem::Coefficient*> lf_coeffs;
   std::vector<mfem::VectorCoefficient*> lf_vector_coeffs;
   std::vector<mfem::MatrixCoefficient*> lf_matrix_coeffs;

   // Store the linear and bilinear forms
   mfem::ParBilinearForm *a;
   mfem::ParLinearForm *b;

   // Store the grid function for the solution
   mfem::ParGridFunction *x;

   // Store the mesh
   mfem::ParMesh *pmesh;

   // Store the finite element collection and space
   mfem::FiniteElementCollection *fec;
   mfem::ParFiniteElementSpace *fespace;
};

// Main problem generation call
void GenerateProblem(HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out, ProblemOptionsList &options, ProblemInfo &probInfo);

// Mesh
mfem::ParMesh* GetMesh(ProblemOptionsList &options);

// MFEM generated problems
void GetMatrixDiffusion(HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out, ProblemOptionsList &options, ProblemInfo &probInfo);
void GetMatrixTransport(HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out, ProblemOptionsList &options, ProblemInfo &probInfo);

// hypre generated problems
HYPRE_Int BuildParLaplacian27pt(HYPRE_ParCSRMatrix *A_ptr, ProblemOptionsList &options );
HYPRE_Int BuildParRotate7pt(HYPRE_ParCSRMatrix *A_ptr, ProblemOptionsList &options );
HYPRE_Int BuildParDifConv(HYPRE_ParCSRMatrix *A_ptr, ProblemOptionsList &options);
HYPRE_Int Tridiagonal(HYPRE_ParCSRMatrix *A_ptr, ProblemOptionsList &options);
HYPRE_Int BuildParLaplacian5pt(HYPRE_ParCSRMatrix  *A_ptr, ProblemOptionsList &options);
HYPRE_Int BuildParGridAlignedAnisotropic(HYPRE_ParCSRMatrix  *A_ptr, ProblemOptionsList &options);

// Conversion from hypre objects to hypre
void MFEMtoHYPRE(mfem::HypreParMatrix &A, mfem::Vector &B, mfem::Vector &X,  HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out);

// Visualization
void VisualizeSolution(HYPRE_ParCSRMatrix A, HYPRE_ParVector B, HYPRE_ParVector X, ProblemOptionsList &options, ProblemInfo &probInfo, std::string custom_sol_name);