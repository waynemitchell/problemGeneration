#include "include/problemGeneration.hpp"

void SetProblemOptions(ProblemOptionsList &options, int argc, char *argv[])
{
   int myid, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   OptionsParser args(argc, argv);
   args.AddOption(&(options.problem), "-P", "--problem",
                  "Select problem");
   args.AddOption(&(options.rhs), "-rhs", "--right-hand-side",
                  "Select right-hand side");
   args.AddOption(&(options.random_initial), "-rand", "--random-initial",
                  "Use random initial guess");
   args.AddOption(&(options.n), "-n", "--num-dofs",
                  "Number of degrees of freedom per processor");
   args.AddOption(&(options.mesh), "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&(options.mesh_partitioning), "-mp", "--mesh-partitioning",
                  "Mesh partitioning");
   args.AddOption(&(options.order), "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&(options.static_cond), "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&(options.visualization), "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&(options.dump_fine_grid_matrix), "-dump-mat", "--dump-fine-grid-matrix",
                  "-no-dump-mat", "--no-dump-fine-grid-matrix",
                  "Dump fine grid matrix to file");
   args.AddOption(&(options.read_exact_solution), "-read-exact", "--read-exact-solution",
                  "-no-read-exact", "--no-read-exact-solution",
                  "Read exact solution from file");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      exit(1);
   }
   if (myid == 0)
   {
      cout << "num_procs = " << num_procs << endl;
      args.PrintOptions(cout);
   }
}

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


