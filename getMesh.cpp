#include "include/problemGeneration.hpp"

ParMesh* GetMesh(ProblemOptionsList &options)
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh;
   int ref_levels = 1;
   if (!strcmp(options.mesh,"square") || !strcmp(options.mesh,"cube"))
   {
      if (!strcmp(options.mesh,"square")) options.dim = 2;
      else if (!strcmp(options.mesh,"cube"))  options.dim = 3;
      int N = options.n * num_procs;
      int M = 0;
      int Ns = N;
      int ref_factor = 2;
      int Ms = round(M / ref_factor);
      if (options.dim == 2) M = round(sqrt(N));
      if (options.dim == 3) M = round(cbrt(N));
      while (M % (ref_factor*2) == 0 || Ns > options.n)
      {
         ref_levels++;
         ref_factor *= 2;
         Ms = round(M / ref_factor);
         if (options.dim == 2) Ns = Ms*Ms;
         if (options.dim == 3) Ns = Ms*Ms*Ms;
      }
      if (!strcmp(options.mesh,"square")) mesh = new Mesh(Ms, Ms, Element::QUADRILATERAL);
      else if (!strcmp(options.mesh,"cube")) mesh = new Mesh(Ms, Ms, Ms, Element::QUADRILATERAL);
   }
   else
   {
      char mesh_file[256] = "meshes/";
      strcat(mesh_file, options.mesh);
      strcat(mesh_file, ".mesh");
      mesh = new Mesh(mesh_file, 1, 1);
      int N = options.n * num_procs;
      int Ns = mesh->GetNE();
      int ref_factor = 1;
      for (auto i = 0; i < options.dim; i++) ref_factor *= 2;
      while (Ns * ref_factor < N) ref_levels++;
   }

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   int serial_ref_levels = 0;
   int Ns = mesh->GetNE();
   int ref_factor = 1;
   for (auto i = 0; i < options.dim; i++) ref_factor *= 2;
   while (Ns * ref_factor < options.n)
   {
      serial_ref_levels++;
      Ns *= ref_factor;
   }
   int par_ref_levels = ref_levels - serial_ref_levels;
   if (myid == 0) cout << "serial_ref_levels = " << serial_ref_levels << ", par_ref_levels = " << par_ref_levels << endl;
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }

   return pmesh;
}