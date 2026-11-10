#!/bin/bash
# P5 assets into the lane, read-only from their sources; cmp before admission.
# base config/tokenizer (small) and the DSpark drafter from `$DSPARK_SRC` (one reader),
# then the production IQ4_NL Engram from production's dir (nvme, idle I/O class,
# rate-limited). Nothing under $PROD/1cat-1277 is written.
: "${LANE:?set LANE to the root directory of this build}"
: "${PROD:?set PROD to the Volta sPLIt checkout}"
set -euo pipefail
cd $LANE/models
B=${BASE_SRC:?set BASE_SRC to a local copy of the DeepSeek-V4.1-Flash-1017 config and tokenizer files}
mkdir -p base dspark-v41 engram
for f in config.json LICENSE README.md model.safetensors.index.json tokenizer_config.json tokenizer.json; do
  cp --preserve=timestamps "$B/$f" "base/$f"; cmp "$B/$f" "base/$f"
done
cp -r --preserve=timestamps "$B/encoding" base/ && diff -r "$B/encoding" base/encoding
D=${DSPARK_SRC:?set DSPARK_SRC to a local copy of the DSpark drafter files}
for f in config.json model.safetensors.index.json; do cp --preserve=timestamps "$D/$f" "dspark-v41/$f"; cmp "$D/$f" "dspark-v41/$f"; done
for f in model-00044-of-00048.safetensors model-00045-of-00048.safetensors model-00046-of-00048.safetensors; do
  src=$(readlink -f "$D/$f")
  ionice -c3 cp "$src" "dspark-v41/$f.part"; cmp "$src" "dspark-v41/$f.part"; mv "dspark-v41/$f.part" "dspark-v41/$f"
  echo "ADMITTED dspark-v41/$f $(stat -c %s "dspark-v41/$f")"
done
E=$PROD/1cat-1277/engram/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
ionice -c3 rsync --bwlimit=200m "$E" "engram/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf.part"
ionice -c3 cmp "$E" "engram/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf.part"
mv engram/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf.part engram/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
echo "ADMITTED engram $(stat -c %s engram/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf)"
echo ALL-ADMITTED
