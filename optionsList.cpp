#include "include/problemGeneration.hpp"

using namespace std;
using namespace mfem;

ProblemOptionsList::ProblemOptionsList(int argc, char *argv[]) : OptionsParser(argc, argv)
{
   AddOption(&problem, "-P", "--problem",
                  "Select problem");
   AddOption(&epsilon, "-ep", "--epsilon",
                  "Set epsilon (different meanings for different problems)");
   AddOption(&a, "-a", "--a",
                  "Set a (different meanings for different problems)");
   AddOption(&b, "-b", "--b",
                  "Set b (different meanings for different problems)");
   AddOption(&rhs, "-rhs", "--right-hand-side",
                  "Select right-hand side");
   AddOption(&random_initial, "-rand", "--random-initial",
                  "Use random initial guess");
   AddOption(&n, "-n", "--num-dofs",
                  "Number of degrees of freedom per processor");
   AddOption(&mesh, "-m", "--mesh",
                  "Mesh file to use.");
   AddOption(&mesh_partitioning, "-mp", "--mesh-partitioning",
                  "Mesh partitioning");
   AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   AddOption(&dump_problem_to_dir, "-dump-prob", "--dump-problem-dir",
                  "Directory to dump the problem to");
   AddOption(&dump_matrix_market, "-dump-mtx", "--dump-matrix-market",
                  "-no-dump-mtx", "--no-dump-matrix-market",
                  "Writing matrix market files (default is HYPRE_ParCSRMatrix)");
   AddOption(&read_problem_from_dir, "-read-prob", "--read-problem-dir",
                  "Directory to read the problem from");
   AddOption(&read_matrix_market, "-read-mtx", "--read-matrix-market",
                  "-no-read-mtx", "--no-read-matrix-market",
                  "Reading matrix market files (default is HYPRE_ParCSRMatrix)");
   AddOption(&read_exact_solution, "-read-exact", "--read-exact-solution",
                  "-no-read-exact", "--no-read-exact-solution",
                  "Read exact solution from file");
}
