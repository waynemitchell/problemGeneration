#include "include/problemGeneration.hpp"

using namespace std;
using namespace mfem;

void VisualizeSolution(HYPRE_ParCSRMatrix A, HYPRE_ParVector B, HYPRE_ParVector X, ProblemOptionsList &options, ProblemInfo &probInfo, string custom_sol_name)
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   if (options.problem >= 0)
   {
      // Recover the parallel grid function corresponding to X. This is the
      // local finite element solution on each processor.

      // Save the refined mesh and the solution in parallel. This output can
      // be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
      ostringstream mesh_name, sol_name, res_name, rhs_name, suffix;
      suffix << "_problem" << options.problem << "P" << num_procs << "n" << options.n << options.mesh;
      mesh_name << "outputs/mesh" << suffix.str() << "." << setfill('0') << setw(6) << myid;
      if (custom_sol_name == "*") sol_name << "outputs/sol" << suffix.str() << "." << setfill('0') << setw(6) << myid;
      else sol_name << "outputs/" << custom_sol_name << suffix.str() << "." << setfill('0') << setw(6) << myid;
      // res_name << "outputs/res" << suffix.str() << "." << setfill('0') << setw(6) << myid;
      // rhs_name << "outputs/rhs" << suffix.str() << "." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      probInfo.pmesh->Print(mesh_ofs);

      HypreParVector X_mfem(X);
      probInfo.a->RecoverFEMSolution(X_mfem, *(probInfo.b), *(probInfo.x));
      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      probInfo.x->Save(sol_ofs);

      // probInfo.a->RecoverFEMSolution(B, *b, *x);
      // ofstream rhs_ofs(rhs_name.str().c_str());
      // rhs_ofs.precision(8);
      // probInfo.x->Save(rhs_ofs);

      // Plot square of residual
      // hypre_ParCSRMatrixMatvec(-1.0, A, X, 1.0, B);
      // hypre_ParVector *B_hypre = (hypre_ParVector*) B;
      // for (auto i = 0; i < hypre_VectorSize(hypre_ParVectorLocalVector(B_hypre)); i++) hypre_VectorData(hypre_ParVectorLocalVector(B_hypre))[i] = fabs(hypre_VectorData(hypre_ParVectorLocalVector(B_hypre))[i]);
      // a->RecoverFEMSolution(B, *b, *x);
      // ofstream res_ofs(res_name.str().c_str());
      // res_ofs.precision(8);
      // x->Save(res_ofs);
   }
   else
   {
      ostringstream sol_name, suffix;
      suffix << "_problem" << options.problem << "P" << num_procs << "n" << options.n;
      if (custom_sol_name == "*") sol_name << "outputs/sol" << suffix.str();
      else sol_name << "outputs/" << custom_sol_name << suffix.str();
      hypre_ParVectorPrint(X, sol_name.str().c_str());
      if (probInfo.x_coords.size())
      {
         ostringstream coord_name;
         coord_name << "outputs/x_coords" << suffix.str() << "." << myid; 
         ofstream coords_out(coord_name.str().c_str());
         for (auto coord = probInfo.x_coords.begin(); coord != probInfo.x_coords.end(); ++coord)
            coords_out << *coord << endl;
      }
      if (probInfo.y_coords.size())
      {
         ostringstream coord_name;
         coord_name << "outputs/y_coords" << suffix.str() << "." << myid; 
         ofstream coords_out(coord_name.str().c_str());
         for (auto coord = probInfo.y_coords.begin(); coord != probInfo.y_coords.end(); ++coord)
            coords_out << *coord << endl;
      }
      if (probInfo.z_coords.size())
      {
         ostringstream coord_name;
         coord_name << "outputs/z_coords" << suffix.str() << "." << myid; 
         ofstream coords_out(coord_name.str().c_str());
         for (auto coord = probInfo.z_coords.begin(); coord != probInfo.z_coords.end(); ++coord)
            coords_out << *coord << endl;
      }
      // hypre_ParCSRMatrixMatvec(-1.0, A, X, 1.0, B);
      // hypre_ParVectorPrint(B, );
   }
}
