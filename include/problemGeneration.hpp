#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <chrono>

using namespace std;
using namespace mfem;

// Collection of options for problem generation
class ProblemOptionsList
{
public:
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
   bool dump_fine_grid_matrix = false;
   bool read_exact_solution = false;
   int dim = 2;
};
void SetProblemOptions(ProblemOptionsList &options, int argc, char *argv[]);

// Main problem generation call
void GenerateProblem(HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out, ProblemOptionsList &options);

// Mesh
ParMesh* GetMesh(ProblemOptionsList &options);

// MFEM generated problems
void GetMatrixDiffusion(HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out, ProblemOptionsList &options);

// hypre generated problems
HYPRE_Int BuildParLaplacian27pt(HYPRE_ParCSRMatrix *A_ptr, ProblemOptionsList &options );
HYPRE_Int BuildParRotate7pt(HYPRE_ParCSRMatrix *A_ptr, ProblemOptionsList &options );
HYPRE_Int BuildParDifConv(HYPRE_ParCSRMatrix *A_ptr, ProblemOptionsList &options);

// Conversion from hypre objects to hypre
void MFEMtoHYPRE(HypreParMatrix &A, Vector &B, Vector &X,  HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out);