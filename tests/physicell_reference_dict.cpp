// Dump PhysiCell's own signal and behavior dictionaries for a given config, so
// the Python library's dictionary can be diffed against it exactly.
#include <cstdio>
#include <iostream>
#include "core/PhysiCell.h"
#include "modules/PhysiCell_standard_modules.h"

using namespace BioFVM;
using namespace PhysiCell;

int main(int argc, char* argv[]) {
  if (argc < 2) { std::cerr << "usage: ref_dict <PhysiCell_settings.xml>\n"; return 2; }
  if (!load_PhysiCell_config_file(argv[1])) return 1;
  initialize_microenvironment();
  create_cell_container_for_microenvironment(microenvironment, 30.0);
  initialize_default_cell_definition();
  cell_defaults.phenotype.secretion.sync_to_microenvironment(&microenvironment);
  initialize_cell_definitions_from_pugixml();
  build_cell_definitions_maps();
  setup_signal_behavior_dictionaries();

  std::cout << "#SIGNALS" << std::endl;
  display_signal_dictionary(std::cout);
  std::cout << "#BEHAVIORS" << std::endl;
  display_behavior_dictionary(std::cout);
  return 0;
}
