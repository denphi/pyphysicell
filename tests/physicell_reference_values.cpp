// For a fresh cell of each cell definition, print every behavior value and
// every signal value PhysiCell reports, so the Python library can be compared
// on values rather than on names.
#include <cstdio>
#include <iostream>
#include "core/PhysiCell.h"
#include "modules/PhysiCell_standard_modules.h"

using namespace BioFVM;
using namespace PhysiCell;

int main(int argc, char* argv[]) {
  if (argc < 2) return 2;
  if (!load_PhysiCell_config_file(argv[1])) return 1;
  initialize_microenvironment();
  create_cell_container_for_microenvironment(microenvironment, 30.0);
  initialize_default_cell_definition();
  cell_defaults.phenotype.secretion.sync_to_microenvironment(&microenvironment);
  initialize_cell_definitions_from_pugixml();
  build_cell_definitions_maps();
  setup_signal_behavior_dictionaries();

  // a deterministic field: substrate s in voxel n holds 1 + s + n/1000
  for (unsigned int n = 0; n < microenvironment.number_of_voxels(); n++)
    for (unsigned int s = 0; s < microenvironment.number_of_densities(); s++)
      microenvironment.density_vector(n)[s] = 1.0 + s + n / 1000.0;
  // the gradient stencil uses thomas_*_jump, which only the diffusion solver
  // initialises, so a harness must step the solver once before asking
  microenvironment.simulate_diffusion_decay(1e-12);
  for (unsigned int n = 0; n < microenvironment.number_of_voxels(); n++)
    for (unsigned int s = 0; s < microenvironment.number_of_densities(); s++)
      microenvironment.density_vector(n)[s] = 1.0 + s + n / 1000.0;
  if (default_microenvironment_options.calculate_gradients)
    microenvironment.compute_all_gradient_vectors();   // as a run would

  const int n_sig = find_signal_index("time") + 1 > 0 ? 0 : 0;  // silence unused
  (void)n_sig;

  for (int k = 0; k < (int)cell_definitions_by_index.size(); k++) {
    Cell_Definition* pCD = cell_definitions_by_index[k];
    Cell* pC = create_cell(*pCD);
    pC->assign_position(0.0, 0.0, 0.0);
    std::cout << "#CELL\t" << pCD->name << std::endl;
    for (int i = 0; ; i++) {
      std::string nm = behavior_name(i);
      if (nm == "not found") break;
      printf("B\t%s\t%.10g\n", nm.c_str(), get_single_behavior(pC, i));
    }
    for (int i = 0; ; i++) {
      std::string nm = signal_name(i);
      if (nm == "not found") break;
      printf("S\t%s\t%.10g\n", nm.c_str(), get_single_signal(pC, i));
    }
  }
  return 0;
}
