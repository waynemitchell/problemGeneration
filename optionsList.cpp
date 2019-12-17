#include "include/problemGeneration.hpp"

ProblemOptionsList::ProblemOptionsList(int argc, char *argv[]) : OptionsParser(argc, argv)
{
   AddOption(&problem, "-P", "--problem",
                  "Select problem");
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
   AddOption(&read_problem_from_dir, "-read-prob", "--read-problem-dir",
                  "Directory to read the problem from");
   AddOption(&read_exact_solution, "-read-exact", "--read-exact-solution",
                  "-no-read-exact", "--no-read-exact-solution",
                  "Read exact solution from file");
}