#include "include/problemGeneration.hpp"

using namespace std;
using namespace mfem;

// Conversion from hypre objects to hypre
void MFEMtoHYPRE(HypreParMatrix &A, Vector &B, Vector &X,  HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out)
{
   /* *A_out = A.StealData(); */
   *A_out = (HYPRE_ParCSRMatrix) A;

   HYPRE_Int global_m, global_n;
   HYPRE_ParCSRMatrixGetDims(*A_out, &global_m, &global_n);
   HYPRE_ParVectorCreate(hypre_MPI_COMM_WORLD, global_n, hypre_ParCSRMatrixRowStarts(*A_out), B_out);
   HYPRE_ParVectorCreate(hypre_MPI_COMM_WORLD, global_m, hypre_ParCSRMatrixColStarts(*A_out), X_out);
   hypre_ParVectorSetPartitioningOwner(*B_out, 0);
   hypre_ParVectorSetPartitioningOwner(*X_out, 0);
   hypre_VectorData(hypre_ParVectorLocalVector(*B_out)) = B.StealData();
   hypre_VectorData(hypre_ParVectorLocalVector(*X_out)) = X.StealData();
   HYPRE_ParVectorInitialize(*B_out);
   HYPRE_ParVectorInitialize(*X_out);
}
