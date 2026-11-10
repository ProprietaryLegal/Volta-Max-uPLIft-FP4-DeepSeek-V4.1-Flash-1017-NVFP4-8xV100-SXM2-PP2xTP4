#!/bin/bash
# P1's isolated GGUF base: copy from NVMe, then compare every byte.
: "${LANE:?set LANE to the root directory of this build}"
set -euo pipefail
cd $LANE
mkdir -p models/gguf
for shard in 00001 00002 00003 00004; do
    name="DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-${shard}-of-00004.gguf"
    source="$GGUF_SRC/$name"
    destination="models/gguf/$name"
    test ! -e "$destination"
    test ! -e "$destination.part"
    cp --reflink=never "$source" "$destination.part"
    cmp "$source" "$destination.part"
    mv "$destination.part" "$destination"
    stat -c 'BYTE-EQUAL %n %s bytes' "$destination"
done
printf 'GGUF_COPY_EXIT=0\n'
