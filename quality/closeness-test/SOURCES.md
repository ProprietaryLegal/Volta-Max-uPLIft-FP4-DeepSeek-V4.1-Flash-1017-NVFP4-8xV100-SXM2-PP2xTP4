# Sources of the closeness test items

Every item is cut from a public source at a pinned revision or date. The texts are not stored in this folder. The
file `items/plan-v1.ids.json` names each item, its source file, its offset and the sha256 of its text, and also the
sha256 of every source file, so `build_items.py` rebuilds the same items and the hashes prove it. The rebuilt plan
has sha256 `19f597f3a7ba379f990212fc5306a97cc4b58f34513cd9b99830885375b4d341`.

None of these texts was used to choose which experts the build keeps at full NVFP4 precision. That choice was made
from route counts on Python 3.12 standard library source, United States Code title 28, six short prompts and
private text. No Python standard library source and no United States Code is used here.

## Next-token contexts (320, 64 of each kind)

| Kind | Source | Revision or date | Contexts | Licence |
|---|---|---|---|---|
| code | Go `src/strings/strings.go` | go1.23.0 | 7 | BSD-3-Clause, the Go Authors |
| code | Go `src/bufio/bufio.go` | go1.23.0 | 6 | BSD-3-Clause, the Go Authors |
| code | lodash `lodash.js` | 4.17.21 | 13 | MIT, OpenJS Foundation and contributors |
| code | requests `src/requests/sessions.py` | v2.32.3 | 7 | Apache-2.0, Kenneth Reitz and contributors |
| code | click `src/click/core.py` | 8.1.7 | 6 | BSD-3-Clause, Pallets |
| code | ripgrep `crates/globset/src/glob.rs` | 14.1.0 | 13 | MIT or Unlicense, Andrew Gallant |
| code | Redis `src/sds.c` | 7.2.4 | 12 | BSD-3-Clause, Salvatore Sanfilippo and contributors |
| prose | Project Gutenberg 944, Darwin, The Voyage of the Beagle | cache file as fetched | 16 | public domain in the United States |
| prose | Project Gutenberg 205, Thoreau, Walden | cache file as fetched | 16 | public domain in the United States |
| prose | Project Gutenberg 2641, Forster, A Room with a View | cache file as fetched | 16 | public domain in the United States |
| prose | Project Gutenberg 3300, Smith, The Wealth of Nations | cache file as fetched | 16 | public domain in the United States |
| legal | eCFR 16 CFR part 310 (Telemarketing Sales Rule) | as of 2026-10-30 | 8 | public domain, US government work |
| legal | eCFR 24 CFR part 100 (Fair Housing) | as of 2026-10-28 | 8 | public domain, US government work |
| legal | eCFR 29 CFR part 825 (Family and Medical Leave) | as of 2026-11-06 | 8 | public domain, US government work |
| legal | eCFR 45 CFR part 164 (Security and Privacy) | as of 2026-11-05 | 8 | public domain, US government work |
| legal | U.S. Reports volume 550, Caselaw Access Project static files, 4 cases chosen by the seed | as fetched | 16 | public domain, court opinions |
| legal | U.S. Reports volume 560, Caselaw Access Project static files, 4 cases chosen by the seed | as fetched | 16 | public domain, court opinions |
| structured | FiveThirtyEight `college-majors/recent-grads.csv` | commit 4c1ff5e3 | 10 | CC BY 4.0, FiveThirtyEight |
| structured | FiveThirtyEight `candy-power-ranking/candy-data.csv` | commit 4c1ff5e3 | 6 | CC BY 4.0, FiveThirtyEight |
| structured | GitHub Linguist `lib/linguist/languages.yml` | v7.30.0 | 16 | MIT, GitHub, Inc. |
| structured | VS Code `extensions/typescript-language-features/package.json` | 1.90.0 | 16 | MIT, Microsoft Corporation |
| structured | Apache Commons Lang `pom.xml` | rel/commons-lang-3.14.0 | 16 | Apache-2.0, Apache Software Foundation |
| chat | databricks-dolly-15k, 8 instructions from each of its 8 categories | commit bdd27f4d | 64 | CC BY-SA 3.0, Databricks |

The exact download URLs are in `build_items.py` and printed by `build_items.py --dry`. Gutenberg and Caselaw
Access Project files are not pinned by their hosts. The ids file records the sha256 of every downloaded file, so a
later download that differs shows up as a different file hash and a different ids file.

## Multiple choice (513 questions)

MMLU (Hendrycks et al., "Measuring Massive Multitask Language Understanding", 2021), the `cais/mmlu` dataset at
revision `c30699e8356da336a370243923dbaf21066bb9fe`, test split, MIT licence. Nine questions from each of the 57
subjects, chosen by the seed 20261110. The questions are not copied here. The ids file names each by subject and
row.

## Attribution

FiveThirtyEight data is used under CC BY 4.0. databricks-dolly-15k is used under CC BY-SA 3.0, and only its row
numbers and text hashes are in this folder. The code sources are used under their licences named above. Nothing
here is changed from the source text except cutting it into windows of whole tokens.
