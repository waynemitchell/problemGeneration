#include "include/problemGeneration.hpp"

void GenerateProblem(HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out, ProblemOptionsList &options)
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   auto start = chrono::system_clock::now();

   HYPRE_ParCSRMatrix A;
   HYPRE_ParVector B;
   HYPRE_ParVector X;

   if (options.problem < 0)
   {   
      switch (options.problem)
      {
         case -1:
            BuildParLaplacian27pt(&A, options);
            break;
         case -2:
            BuildParRotate7pt(&A, options);
            break;
         default:
            BuildParDifConv(&A, options);
      }
      HYPRE_Int *partitioning;
      HYPRE_Int global_m, global_n;
      HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
      HYPRE_ParCSRMatrixGetDims(A, &global_m, &global_n);
      HYPRE_ParVectorCreate(hypre_MPI_COMM_WORLD, global_m, partitioning, &B);
      HYPRE_ParVectorCreate(hypre_MPI_COMM_WORLD, global_m, partitioning, &X);
      HYPRE_ParVectorInitialize(B);
      HYPRE_ParVectorInitialize(X);
   }
   else 
   {

      if (options.problem < 10) GetMatrixDiffusion(&A,&B,&X,options);
      // else if (options.problem < 20) GetMatrixAdvectionDiffusionReaction(&A,&B,&X,options)
      // else if (options.problem < 30) GetMatrixElasticity(&A,&B,&X,options);
      else
      {
         if (myid == 0) cout << "Unknown problem" << endl;
         MPI_Finalize();
         exit(0);
      }

   }

   if (options.random_initial) HYPRE_ParVectorSetRandomValues(X, myid);
   if (options.rhs == -1) HYPRE_ParVectorSetRandomValues(B, myid+1);

   // Print matrix
   if (options.dump_fine_grid_matrix)
   {
      char filename[1024];
      sprintf(filename, "outputs/A");
      hypre_ParCSRMatrixPrint( A, filename );
      sprintf(filename, "outputs/b");
      hypre_ParVectorPrint( B, filename );
      sprintf(filename, "outputs/x");
      hypre_ParVectorPrint( X, filename );
   }

   if (myid == 0)
   {
      cout << "Size of linear system: " << hypre_ParCSRMatrixGlobalNumRows(A) << endl;
      cout << "Dofs per processor: " << hypre_ParCSRMatrixNumRows(A) << endl;
   }

   *A_out = A;
   *B_out = B;
   *X_out = X;

   auto end = chrono::system_clock::now();
   std::chrono::duration<double> elapsed = end - start;
   if (myid == 0) cout << "   Generate problem: " << elapsed.count() << endl;
}


