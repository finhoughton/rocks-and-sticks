#!/bin/sh
# Build the rs_engine Python extension (fast path; setup.py builds it too), the frozen older
# versions in engine_src/versions/v<n> (as modules rs_engine_v<n>, used for rated comparisons),
# and the self-play tool.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
INC=$($PY -m pybind11 --includes)
SUFFIX=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
FLAGS="-O3 -DNDEBUG -std=c++17 -fPIC -fvisibility=hidden -shared -undefined dynamic_lookup"
# shellcheck disable=SC2086
c++ $FLAGS $INC engine_src/rs_bindings.cpp -o "rs_engine$SUFFIX"
echo "built rs_engine$SUFFIX"
for dir in engine_src/versions/v*/; do
    [ -f "$dir/rs_bindings.cpp" ] || continue
    name="rs_engine_$(basename "$dir")"
    out="$name$SUFFIX"
    # rebuild a frozen version only when its sources are newer than the module
    if [ -f "$out" ] && [ -z "$(find "$dir" -name '*.?pp' -newer "$out")" ]; then
        continue
    fi
    # shellcheck disable=SC2086
    c++ $FLAGS $INC -DRS_MODULE_NAME="$name" "$dir/rs_bindings.cpp" -o "$out"
    echo "built $out"
done
c++ -O3 -DNDEBUG -std=c++17 -pthread engine_src/rs_selfplay.cpp -o engine_src/rs_selfplay
echo "built engine_src/rs_selfplay"
