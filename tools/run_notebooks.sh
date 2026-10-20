#!/bin/bash
# Execute every generated notebook in place, 4 at a time, and report.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
run_one() {
  nb="$1"; t0=$(date +%s)
  if $PY -m nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.timeout=900 --ExecutePreprocessor.kernel_name=pyphysicell "$nb" >/dev/null 2>"/tmp/nbrun_$(basename "$nb").err"; then
    echo "PASS $(basename "$nb" .ipynb) $(( $(date +%s) - t0 ))s"
  else
    echo "FAIL $(basename "$nb" .ipynb) $(( $(date +%s) - t0 ))s :: $(grep -oE '[A-Za-z]*Error:.*' "/tmp/nbrun_$(basename "$nb").err" | head -1 | cut -c1-90)"
  fi
}
export -f run_one; export PY
ls notebooks/*.ipynb | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {} | sort | tee notebooks/.run_results
