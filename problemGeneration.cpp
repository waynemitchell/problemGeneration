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

   // Read problem
   if (options.read_problem_from_dir[0] != '*')
   {
      char filename[1024];
      sprintf(filename, "%s/A_problem%dP%dn%d", options.read_problem_from_dir, options.problem, num_procs, options.n);
      A = hypre_ParCSRMatrixRead(MPI_COMM_WORLD, filename);
      sprintf(filename, "%s/b_problem%dP%dn%d", options.read_problem_from_dir, options.problem, num_procs, options.n);
      B = hypre_ParVectorRead(MPI_COMM_WORLD, filename);
      // sprintf(filename, "outputs/x");
      // hypre_ParVectorPrint( X, filename );
      HYPRE_Int *partitioning;
      HYPRE_Int global_m, global_n;
      HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
      HYPRE_ParCSRMatrixGetDims(A, &global_m, &global_n);
      HYPRE_ParVectorCreate(hypre_MPI_COMM_WORLD, global_m, partitioning, &X);
      HYPRE_ParVectorInitialize(X);
   }
   else
   {
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

      // Dump problem
      if (options.dump_problem_to_dir[0] != '*')
      {
         char filename[1024];
         sprintf(filename, "%s/A_problem%dP%dn%d", options.dump_problem_to_dir, options.problem, num_procs, options.n);
         hypre_ParCSRMatrixPrint( A, filename );
         sprintf(filename, "%s/b_problem%dP%dn%d", options.dump_problem_to_dir, options.problem, num_procs, options.n);
         hypre_ParVectorPrint( B, filename );
         // sprintf(filename, "outputs/x");
         // hypre_ParVectorPrint( X, filename );
      }
   }

   if (options.random_initial) HYPRE_ParVectorSetRandomValues(X, myid);
   if (options.rhs == -1) HYPRE_ParVectorSetRandomValues(B, myid+1);
   
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


