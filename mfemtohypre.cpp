#include "include/problemGeneration.hpp"

// Conversion from hypre objects to hypre
void MFEMtoHYPRE(HypreParMatrix &A, Vector &B, Vector &X,  HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out)
{
   *A_out = A.StealData();

   HYPRE_Int *partitioning;
   HYPRE_Int global_m, global_n;
   HYPRE_ParCSRMatrixGetRowPartitioning(*A_out, &partitioning);
   HYPRE_ParCSRMatrixGetDims(*A_out, &global_m, &global_n);
   HYPRE_ParVectorCreate(hypre_MPI_COMM_WORLD, global_m, partitioning, B_out);
   HYPRE_ParVectorCreate(hypre_MPI_COMM_WORLD, global_m, partitioning, X_out);
   hypre_VectorData(hypre_ParVectorLocalVector(*B_out)) = B.StealData();
   hypre_VectorData(hypre_ParVectorLocalVector(*X_out)) = X.StealData();
   HYPRE_ParVectorInitialize(*B_out);
   HYPRE_ParVectorInitialize(*X_out);
}