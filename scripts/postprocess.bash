#!/bin/bash
# Post-processing for one finished run: make every plot, then move them into plots/ renamed
# <stem>_nfp<NFP>_<target><qs|noqs>.<ext>. Run from the run dir (t<N>_qs/ or t<N>_noqs/), with the python env
# already set up (batch_optim.slurm and replot.slurm call it inside singularity).
#
# Usage: postprocess.bash <target> <aspect> <iota> <nfp> [<helicity_m> <helicity_n>]
#   m and n given -> QS run (plot_specific_qs.py), omitted -> no-QS run (plot_specific_noqs.py)

if [ "$#" -ne 4 ] && [ "$#" -ne 6 ]; then
    echo "Usage: $0 <target> <aspect> <iota> <nfp> [<helicity_m> <helicity_n>]" >&2
    exit 1
fi

TARGET=$1
ASPECT=$2
IOTA=$3
NFP=$4

SRC_DIR="$(dirname "$(dirname "$(realpath "$0")")")/src"

if [ "$#" -eq 6 ]; then
    QS_TAG=qs
    PLOT_SCRIPT=plot_specific_qs.py
    PLOT_ARGS="$5 $6"
else
    QS_TAG=noqs
    PLOT_SCRIPT=plot_specific_noqs.py
    PLOT_ARGS=""
fi

# Suffix for the plot files moved into plots/, e.g. nfp4_3qs
SUFFIX="nfp${NFP}_${TARGET}${QS_TAG}"

echo "starting plotting in $(pwd) (${SUFFIX})"
python "${SRC_DIR}/plot_total_obj.py" "${TARGET}"
python "${SRC_DIR}/${PLOT_SCRIPT}" "${ASPECT}" "${IOTA}" "${TARGET}" ${PLOT_ARGS}
python "${SRC_DIR}/make_paraview.py"
python "${SRC_DIR}/plot_surfaces.py" "${TARGET}"

# Move plots into plots/, renamed as <stem>_${SUFFIX}.<ext>
mkdir -p plots
for pair in total_obj.pdf:total_obj targets.pdf:targets objectives_paraview.vts:paraview boozer.pdf:boozer heatmap.pdf:heatmap plot3.pdf:flux_surfaces; do
    src=${pair%%:*}
    stem=${pair##*:}
    ext=${src##*.}
    if [[ -e "$src" ]]; then
        mv "$src" "plots/${stem}_${SUFFIX}.${ext}"
    else
        echo "skipping $src (not found)"
    fi
done
