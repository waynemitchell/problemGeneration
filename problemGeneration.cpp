#include "include/problemGeneration.hpp"

using namespace std;
using namespace mfem;

void ReadMatrixMarket(HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out, ProblemOptionsList &options);
void WriteMatrixMarket(HYPRE_ParCSRMatrix A_par, ProblemOptionsList &options);

void GenerateProblem(HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out, ProblemOptionsList &options, ProblemInfo &probInfo)
{
   int num_procs, myid;
   MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   if (myid == 0) cout << "Generating problem ..." << endl;

   auto start = chrono::system_clock::now();

   HYPRE_ParCSRMatrix A;
   HYPRE_ParVector B;
   HYPRE_ParVector X;

   // Read problem
   if (options.read_problem_from_dir[0] != '*')
   {
      if (options.read_matrix_market) ReadMatrixMarket(&A, &B, &X, options);
      else
      {
         char filename[1024];
         sprintf(filename, "%s/A_problem%dP%dn%d", options.read_problem_from_dir, options.problem, num_procs, options.n);
         A = hypre_ParCSRMatrixRead(hypre_MPI_COMM_WORLD, filename);
         sprintf(filename, "%s/b_problem%dP%dn%d", options.read_problem_from_dir, options.problem, num_procs, options.n);
         B = hypre_ParVectorRead(hypre_MPI_COMM_WORLD, filename);
         HYPRE_ParVectorCreate(hypre_MPI_COMM_WORLD, hypre_ParVectorGlobalSize(B), hypre_ParVectorPartitioning(B), &X);
         HYPRE_ParVectorInitialize(X);
         hypre_ParVectorSetPartitioningOwner(X, 0);
      }
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
               BuildParRotate7pt(&A, options, probInfo);
               break;
            case -3:
               BuildParDifConv(&A, options, probInfo);
               break;
            case -4:
               BuildParLaplacian5pt(&A, options);
               break;
            case -5:
               BuildParGridAlignedAnisotropic(&A, options, probInfo);
               break;
            default:
               Tridiagonal(&A, options);
         }
         HYPRE_BigInt *partitioning;
         HYPRE_BigInt global_m, global_n;
         HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
         HYPRE_ParCSRMatrixGetDims(A, &global_m, &global_n);
         HYPRE_ParVectorCreate(hypre_MPI_COMM_WORLD, global_m, partitioning, &B);
         HYPRE_ParVectorCreate(hypre_MPI_COMM_WORLD, global_m, partitioning, &X);
         HYPRE_ParVectorInitialize(B);
         HYPRE_ParVectorInitialize(X);
        if (options.rhs == 1) HYPRE_ParVectorSetConstantValues(B, 1.0);
      }
      else 
      {

         if (options.problem < 10) GetMatrixDiffusion(&A,&B,&X,options,probInfo);
         // else if (options.problem < 20) GetMatrixAdvectionDiffusionReaction(&A,&B,&X,options,probInfo)
         // else if (options.problem < 30) GetMatrixElasticity(&A,&B,&X,options,probInfo);
         /* else if (options.problem < 40) GetMatrixTransport(&A,&B,&X,options,probInfo); */
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
         if (options.dump_matrix_market)
         {
            WriteMatrixMarket(A, options);
         }
         else
         {
             char filename[1024];
             sprintf(filename, "%s/A_problem%dP%dn%d%s", options.dump_problem_to_dir, options.problem, num_procs, options.n, options.mesh);
             hypre_ParCSRMatrixPrint( A, filename );
             sprintf(filename, "%s/b_problem%dP%dn%d%s", options.dump_problem_to_dir, options.problem, num_procs, options.n, options.mesh);
             HYPRE_ParVectorPrint( B, filename );
         }
      }
   }

   if (options.random_initial) HYPRE_ParVectorSetRandomValues(X, myid);
   if (options.rhs == -1) HYPRE_ParVectorSetRandomValues(B, myid+1);
   if (options.rhs == 0) HYPRE_ParVectorSetConstantValues(B, 0.0);

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

void WriteMatrixMarket(HYPRE_ParCSRMatrix A_par, ProblemOptionsList &options)
{
   int num_procs, myid;
   MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   if (num_procs != 1)
   {
      printf("Writing of matrix market only suppported for one processor\n");
      MPI_Finalize();
      exit(1);
   }

   // Write out the matrix
   char filename[256];
   sprintf(filename, "%s/A_problem%dn%d%s.mtx", options.dump_problem_to_dir, options.problem, options.n, options.mesh);

   cout << "Writing matrix market " << filename << endl;

   std::fstream ifs(filename, std::ios::out);
   if (!ifs.is_open())
   {
      printf("Failed to open matrix market file.\n");
      MPI_Finalize();
      exit(1);
   }

   // Coordinate real general type 
   ifs << "%%MatrixMarket matrix coordinate real general" << endl;

   // Num rows, cols, nonzeros
   hypre_CSRMatrix *A = hypre_ParCSRMatrixDiag(A_par);
   ifs << hypre_CSRMatrixNumRows(A) << " "
       << hypre_CSRMatrixNumCols(A) << " "
       << hypre_CSRMatrixNumNonzeros(A) << " " << endl;

   // Write in COO format (not necessarily sorted by row)
   for (auto i = 0; i < hypre_CSRMatrixNumRows(A); i++)
   {
       for (auto j = hypre_CSRMatrixI(A)[i]; j < hypre_CSRMatrixI(A)[i+1]; j++)
       {
           ifs << i+1 << " " 
               << hypre_CSRMatrixJ(A)[j] + 1 << " "
               << hypre_CSRMatrixData(A)[j] << endl;
       }
   }
}

void ReadMatrixMarket(HYPRE_ParCSRMatrix *A_out, HYPRE_ParVector *B_out, HYPRE_ParVector *X_out, ProblemOptionsList &options)
{
   int num_procs, myid;
   MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   if (num_procs != 1)
   {
      printf("Reading of matrix market only suppported for one processor\n");
      MPI_Finalize();
      exit(1);
   }

   // Read in the matrix
   char filename[256];
   sprintf(filename, "%s.mtx", options.read_problem_from_dir);

   cout << "Reading matrix market " << filename << endl;

   std::fstream ifs(filename, std::ios::in);
   if (!ifs.is_open())
   {
      printf("Failed to open matrix market file.\n");
      MPI_Finalize();
      exit(1);
   }
   std::string line;
   std::getline(ifs, line);


   // Only general and symmetric supported 
   if (!(line == "%%MatrixMarket matrix coordinate real general" || line == "%%MatrixMarket matrix coordinate real symmetric"))
   {
      printf("Only general or symmetric matrices supported\n");
      MPI_Finalize();
      exit(1);
   }
   const bool symmetric =
       line == "%%MatrixMarket matrix coordinate real symmetric";

   // skip all comments
   while (line[0] == '%') {
      std::getline(ifs, line);
   }

   int num_rows = 0;
   int num_cols = 0;
   int num_nonzeros = 0;
   sscanf(line.c_str(), "%d %d %d", &num_rows, &num_cols, &num_nonzeros);
   if (!(num_rows >= 0 && num_cols >= 0 && num_nonzeros >= 0))
   {
      printf("Unexpected matrix size\n");
      MPI_Finalize();
      exit(1);
   }

   // Read in COO format (not necessarily sorted by row)
   HYPRE_Int *matrix_i = hypre_CTAlloc(HYPRE_Int, num_nonzeros, HYPRE_MEMORY_HOST);
   HYPRE_Int *matrix_j = hypre_CTAlloc(HYPRE_Int, num_nonzeros, HYPRE_MEMORY_HOST);
   HYPRE_Complex *matrix_coo_data = hypre_CTAlloc(HYPRE_Complex, num_nonzeros, HYPRE_MEMORY_HOST);

   int cnt = 0;
   while (std::getline(ifs, line)) 
   {
      if (!line.empty()) 
      {
         stringstream line_stream;
         line_stream.str(line);
         int i, j;
         double data;
         line_stream >> i;
         line_stream >> j;
         line_stream >> data;
         matrix_i[cnt] = (HYPRE_Int) i - 1;
         matrix_j[cnt] = (HYPRE_Int) j - 1;
         matrix_coo_data[cnt] = (HYPRE_Complex) data;
         cnt++;
      }
   }

   HYPRE_Int *row_ptr = hypre_CTAlloc(HYPRE_Int, num_rows + 1, HYPRE_MEMORY_HOST);
   HYPRE_Int *col_ind = hypre_CTAlloc(HYPRE_Int, num_nonzeros, HYPRE_MEMORY_HOST);
   HYPRE_Complex *matrix_csr_data = hypre_CTAlloc(HYPRE_Complex, num_nonzeros, HYPRE_MEMORY_HOST);
   HYPRE_Int row = matrix_i[0];

   // Get row sums
   for (auto i = 0; i < num_nonzeros; i++)
   {
      row_ptr[matrix_i[i]]++;
   }
   // Setup row pointer
   int sum = 0;
   for (auto i = 0; i < num_rows; i++)
   {
      int row_size = row_ptr[i];
      row_ptr[i] = sum;
      sum += row_size;
   }
   row_ptr[num_rows] = num_nonzeros;
   // Set col ind and data
   for(auto i = 0; i < num_nonzeros; i++)
   {
      int row  = matrix_i[i];
      int dest = row_ptr[row];

      col_ind[dest] = matrix_j[i];
      matrix_csr_data[dest] = matrix_coo_data[i];

      row_ptr[row]++;
   }
   // Fix up row pointer
   int last = 0;
   for(auto i = 0; i <= num_rows; i++)
   {
      int temp = row_ptr[i];
      row_ptr[i] = last;
      last = temp;
   }
   hypre_TFree(matrix_i, HYPRE_MEMORY_HOST);
   hypre_TFree(matrix_j, HYPRE_MEMORY_HOST);
   hypre_TFree(matrix_coo_data, HYPRE_MEMORY_HOST);

   hypre_CSRMatrix *A_seq = hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   hypre_CSRMatrixI(A_seq) = row_ptr;
   hypre_CSRMatrixJ(A_seq) = col_ind;
   hypre_CSRMatrixData(A_seq) = matrix_csr_data;

   HYPRE_BigInt *row_starts = hypre_CTAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);
   row_starts[1] = num_rows;
   HYPRE_ParCSRMatrix A = hypre_CSRMatrixToParCSRMatrix(hypre_MPI_COMM_WORLD, A_seq, row_starts, NULL);

   if (symmetric)
   {
      // Get transpose
      hypre_CSRMatrix *AT;
      hypre_CSRMatrixTranspose(A_seq, &AT, 1);
      // Zero diagonal for AT
      for (auto i = 0; i < num_cols; i++)
      {
         for (auto j = hypre_CSRMatrixI(AT)[i]; j < hypre_CSRMatrixI(AT)[i+1]; j++)
         {
            if (hypre_CSRMatrixJ(AT)[j] == i) hypre_CSRMatrixData(AT)[j] = 0;
         }
      }
      // Add to get symmetrized A
      hypre_CSRMatrix *A_sym = hypre_CSRMatrixAdd(A_seq, AT);
   }

   // Read in the RHS
   sprintf(filename, "%s_b_uniform-min-1-max-10.mtx", options.read_problem_from_dir);

   std::fstream ifs_b(filename, std::ios::in);
   if (!ifs.is_open())
   {
      printf("Failed to open rhs file.\n");
      MPI_Finalize();
      exit(1);
   }
   std::getline(ifs_b, line);

   if (!(line == "%%MatrixMarket matrix array real general"))
   {
      printf("Unexpected rhs matrix type\n");
      MPI_Finalize();
      exit(1);
   }

   // skip all comments
   while (line[0] == '%') {
      std::getline(ifs_b, line);
   }

   int b_num_rows = 0;
   int b_num_cols = 0;
   sscanf(line.c_str(), "%d %d", &b_num_rows, &b_num_cols);
   if (!(b_num_rows >= 0 && b_num_cols == 1))
   {
      printf("Unexpected rhs size: num rows = %d, num cols = %d\n", b_num_rows, b_num_cols);
      MPI_Finalize();
      exit(1);
   }

   HYPRE_Complex *b_data = hypre_CTAlloc(HYPRE_Complex, b_num_rows, HYPRE_MEMORY_HOST);

   cnt = 0;
   while (std::getline(ifs_b, line)) 
   {
      if (!line.empty()) 
      {
         stringstream line_stream;
         line_stream.str(line);
         double data;
         line_stream >> data;
         b_data[cnt++] = (HYPRE_Complex) data;
      }
   }

   hypre_Vector *b_seq = hypre_SeqVectorCreate(b_num_rows);
   hypre_VectorData(b_seq) = b_data;

   HYPRE_ParVector B = hypre_VectorToParVector (hypre_MPI_COMM_WORLD, b_seq, row_starts);
   hypre_ParVectorSetPartitioningOwner(B, 0);

   // Setup X with zero initialization
   HYPRE_ParVector X;
   HYPRE_ParVectorCreate(hypre_MPI_COMM_WORLD, hypre_ParVectorGlobalSize(B), hypre_ParVectorPartitioning(B), &X);
   HYPRE_ParVectorInitialize(X);
   hypre_ParVectorSetPartitioningOwner(X, 0);

   *A_out = A;
   *B_out = B;
   *X_out = X;
}
