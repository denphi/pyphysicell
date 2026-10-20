// Dump the structure of PhysiCell's standard cycle and death models, so the
// Python library's handling of each can be checked rather than assumed.
#include <iostream>
#include "core/PhysiCell.h"
#include "modules/PhysiCell_standard_modules.h"

using namespace PhysiCell;

static void dump(Cycle_Model& m) {
  std::cout << "MODEL\t" << m.code << "\t" << m.name << "\tphases=" << m.phases.size() << "\n";
  for (size_t i = 0; i < m.phases.size(); i++) {
    std::cout << "  PHASE\t" << i << "\t" << m.phases[i].name
              << "\tdivision_at_exit=" << m.phases[i].division_at_phase_exit
              << "\tremoval_at_exit=" << m.phases[i].removal_at_phase_exit
              << "\tentry_fn=" << (m.phases[i].entry_function ? 1 : 0) << "\n";
    for (size_t k = 0; k < m.phase_links[i].size(); k++) {
      std::cout << "    LINK\t" << i << "->" << m.phase_links[i][k].end_phase_index
                << "\tfixed=" << m.phase_links[i][k].fixed_duration
                << "\trate=" << m.data.transition_rates[i][k]
                << "\tarrest=" << (m.phase_links[i][k].arrest_function ? 1 : 0) << "\n";
    }
  }
}

int main() {
  create_standard_cell_cycle_models();
  create_standard_cell_death_models();
  Cycle_Model* models[] = {&Ki67_advanced, &Ki67_basic, &live, &flow_cytometry_cycle_model,
                           &flow_cytometry_separated_cycle_model, &cycling_quiescent,
                           &apoptosis, &necrosis};
  for (auto* m : models) dump(*m);
  return 0;
}
