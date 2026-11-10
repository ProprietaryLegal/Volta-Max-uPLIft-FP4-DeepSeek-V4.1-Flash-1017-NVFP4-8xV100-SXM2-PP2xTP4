#!/bin/bash
# Copy NVFP4 weights HDD->NVMe, verifying each LFS file's sha256 against the HF oid. Fails fast.
: "${LANE:?set LANE to the root directory of this build}"
set -euo pipefail
SRC=${WEIGHTS_SRC:?set WEIGHTS_SRC to a local download of nvidia/DeepSeek-V4.1-Flash-1017-NVFP4}
DST=$LANE/models/DeepSeek-V4.1-Flash-1017-NVFP4
REV=480550d349ede47a048eb65a276c5c01a06256e2
mkdir -p "$DST"
python3 -I - "$SRC" "$DST" "$REV" <<'PY'
import sys,os,hashlib,shutil
from huggingface_hub import HfApi
src,dst,rev=sys.argv[1:]
info=HfApi().model_info("nvidia/DeepSeek-V4.1-Flash-1017-NVFP4",revision=rev,files_metadata=True)
for s in sorted(info.siblings,key=lambda x:x.rfilename):
    sp=os.path.join(src,s.rfilename); dp=os.path.join(dst,s.rfilename)
    os.makedirs(os.path.dirname(dp),exist_ok=True)
    h=hashlib.sha256()
    with open(sp,'rb') as f, open(dp+'.part','wb') as g:
        while True:
            b=f.read(64<<20)
            if not b: break
            h.update(b); g.write(b)
    want=s.lfs.sha256 if s.lfs else None
    got=h.hexdigest()
    if os.path.getsize(dp+'.part')!=s.size: raise SystemExit(f"SIZE MISMATCH {s.rfilename}")
    if want and got!=want: raise SystemExit(f"SHA MISMATCH {s.rfilename} {got} != {want}")
    os.rename(dp+'.part',dp)
    print("OK",s.rfilename,s.size,"sha" if want else "size-only(non-LFS)",flush=True)
print("ALL-VERIFIED",len(info.siblings),flush=True)
PY
