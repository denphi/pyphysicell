// Reference oracle: run BioFVM's own LOD diffusion solver on a fixed test case
// and dump the field, so the open-physicell kernel can be compared against it.
// Built against the vendored PhysiCell checkout; serial (the pragmas are
// no-ops without -fopenmp), which does not change the result.
#include <cstdio>
#include <cstdlib>
#include <string>
#include "BioFVM.h"

using namespace BioFVM;

int main(int argc, char** argv) {
  const bool two_d = (argc > 1 && std::string(argv[1]) == "2d");
  const int nsteps = argc > 2 ? atoi(argv[2]) : 100;
  const char* out = argc > 3 ? argv[3] : "biofvm_ref.bin";

  Microenvironment M;
  M.name = "ref";
  if (two_d) {
    default_microenvironment_options.simulate_2D = true;
    M.resize_space(-200.0, 200.0, -200.0, 200.0, -10.0, 10.0, 20.0, 20.0, 20.0);
    M.diffusion_decay_solver = diffusion_decay_solver__constant_coefficients_LOD_2D;
  } else {
    M.resize_space(-200.0, 200.0, -200.0, 200.0, -200.0, 200.0, 20.0, 20.0, 20.0);
    M.diffusion_decay_solver = diffusion_decay_solver__constant_coefficients_LOD_3D;
  }
  M.set_density(0, "oxygen", "mmHg");
  M.diffusion_coefficients[0] = 100000.0;
  M.decay_rates[0] = 0.1;
  M.add_density("debris", "none");
  M.diffusion_coefficients[1] = 1000.0;
  M.decay_rates[1] = 0.01;

  // a reproducible, non-symmetric initial state
  for (unsigned int n = 0; n < M.number_of_voxels(); ++n) {
    const std::vector<double>& c = M.mesh.voxels[n].center;
    M.density_vector(n)[0] = 10.0 + 0.01 * c[0] + 0.02 * c[1] + 0.003 * c[2];
    M.density_vector(n)[1] = (n % 7 == 0) ? 5.0 : 0.0;
  }

  // BioFVM's Dirichlet activation defaults to false: switch oxygen on, leave debris off
  M.set_substrate_dirichlet_activation(0, true);
  M.set_substrate_dirichlet_activation(1, false);

  // Dirichlet on the x-faces for oxygen only, as a typical model does
  for (unsigned int n = 0; n < M.number_of_voxels(); ++n) {
    const std::vector<double>& c = M.mesh.voxels[n].center;
    if (c[0] < M.mesh.bounding_box[0] + M.mesh.dx ||
        c[0] > M.mesh.bounding_box[3] - M.mesh.dx) {
      std::vector<double> v(M.number_of_densities(), 0.0);
      v[0] = 38.0;
      M.add_dirichlet_node(n, v);
    }
  }

  const double dt = 0.01;
  for (int s = 0; s < nsteps; ++s) M.simulate_diffusion_decay(dt);

  FILE* f = fopen(out, "wb");
  const long nv = M.number_of_voxels(), nd = M.number_of_densities();
  fwrite(&nv, sizeof(long), 1, f);
  fwrite(&nd, sizeof(long), 1, f);
  for (long n = 0; n < nv; ++n)
    fwrite(&(M.density_vector(n)[0]), sizeof(double), nd, f);
  fclose(f);
  printf("wrote %ld voxels x %ld substrates to %s\n", nv, nd, out);
  return 0;
}
