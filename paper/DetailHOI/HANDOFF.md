# DetailHOI — project handoff

Written for an agent picking this up cold. Everything below was verified against the
code and logs in `/mnt/fast/DetailHOI` unless a line says otherwise. Where something is
inferred rather than recorded, it is labelled.

**Read this before touching anything:** the repo is **not** under version control
(`git status` fails — there is no `.git`). File modification times are the only history.
Edits are unrecoverable. Copy before you rewrite.

---

## 1. What the project is

A two-stage Human–Object Interaction (HOI) detector. The baseline is **PViC**
(Zhang et al., ICCV 2023) — a frozen DETR supplies instances, and a small trained
interaction head scores every human–object pair. PViC steers the head's cross-attention
with a **positional query** built from the two box centres.

**The contribution:** replace the *human half* of that positional query with an embedding
of the person's **skeleton** instead of their box centre. Nothing else in the pipeline
changes — same decoder, same width, same loss, same inference cost.

Author: Chunyu Jiang (Monash University). Supervisors: A/Prof. Chao Chen, Prof. Michael Wang.

### The method, precisely as implemented

Let `PE : R^2 -> R^256` be the sinusoidal point embedding (`ops.compute_sinusoidal_pe`,
temperature 20).

| | PViC baseline (`pvic.py`) | DetailHOI (`detailhoi*.py`) |
|---|---|---|
| cross-attn `q_pos` ("centre") | `[PE(c_i) ; PE(c_j)]`, 256+256 = 512 | `[W_h·vec(PE(p_i,1..K)) ; W_o·PE(c_j)]`, `D_h + D_o = 512` |
| self-attn `q_pos` ("box") | `[c_pe;wh_pe]_i ⊕ [c_pe;wh_pe]_j`, 1024 | **identical — untouched** |
| content query `q_ij` | visual ⊕ 36-d spatial | **identical — untouched** |

Four details that matter:

1. **Extent modulation.** DAB-DETR modulates the box-centre embedding by the box's own
   `(w,h)`. DetailHOI does the same for the skeleton, but the scale is the bounding
   extent of the *confident* joints (`conf > 0.3`), falling back to `(1,1)`. Reference
   scale comes from `ref_keypoint_head(embeds).sigmoid()`.
2. **Confidences are not a soft mask.** They only decide which joints set the skeleton's
   scale. Every keypoint is embedded regardless.
3. **`W_h` is a single `nn.Linear(K*256, D_h)`.** This is why interpolated keypoints
   cannot help — a linear map already forms any convex combination of its inputs. That
   observation is a load-bearing argument in the paper.
4. **Everything upstream is frozen.** DETR and RTMPose both run under `no_grad`.
   `model.freeze_detector()` is called in both entry points.

---

## 2. What is already produced (this folder)

| File | What it is |
|---|---|
| `HANDOFF.md` | This file. Start here. |
| `README.md` | Short index of the folder; points back here. |
| `DetailHOI.docx` | Word manuscript, ~6,400 words, figures + 5 tables embedded. A4, Times New Roman 11pt, booktabs-style rules, page numbers. |
| `detailhoi.tex` | Same paper as self-contained LaTeX. `pdflatex detailhoi.tex` twice. Inline `thebibliography` — no BibTeX run needed. **Never compiled** (no TeX in the authoring environment); structure was validated programmatically (envs balanced, `$` parity, all refs/cites resolve). |
| `detailhoi.html` | Web version, figures inlined as base64. Published as an Artifact at `https://claude.ai/code/artifact/406b1935-0320-4af4-8f11-6da737136123`. Republish the same path to update that URL. |
| `figures/flowchart.{pdf,png}` | Fig. 1, simplified before/after schematic. Drawn in matplotlib (`mkflow.py`). Added 2026-08-24; see §11. |
| `figures/architecture.{pdf,png}` | Fig. 2, pipeline diagram. Drawn in matplotlib. |
| `figures/convergence.{pdf,png}` | Fig. 3, from the training logs. |
| `figures/ablation.{pdf,png}` | Fig. 4, from the training logs. |
| `figures/attention.{pdf,png}` | Fig. 5, qualitative attention-decomposition figure. Assembled by `mkattn.py` from `figures/_attn_source/`. Added 2026-08-24; see §11. |
| `figures/curves.json` | Per-epoch mAP scraped from **every** wandb run. The figures rebuild from this alone. |
| `mkfigs.py`, `mkarch.py`, `mkflow.py` | Regenerate the drawn figures. `python3 mkfigs.py && python3 mkarch.py && python3 mkflow.py` from this folder. |
| `mkattn.py` | Rebuilds Fig. 5 from `figures/_attn_source/`. |
| `Detailed human object interation.docx` | The author's **original draft**. Superseded but kept — some numbers exist only here (§6). |
| `Progress Review .pptx` | Candidature-review talk. Source of the §11 additions. Treat its numbers as **unverified** relative to this repo's code/logs (§11). |

The three formats carry identical content. If you change a number, change it in all three
plus `README.md`.

---

## 3. Repository map

### Entry points
| File | Imports | Purpose |
|---|---|---|
| `main.py` | `pvic.build_detector` | **PViC baseline.** |
| `main_d.py` | `detailhoi1.build_detector` | **DetailHOI.** The import on line 16 is the switch — the ablations were run by editing it. |

Both entry points contain an identical `get_filtered_indices` / `FilteredDataFactory`
block that silently subsets the data **if the whitelist `.txt` files exist**. See §5.

`configs.py` serves `main.py`, `configs_d.py` serves `main_d.py`. Defaults in
`configs_d.py`: lr 1e-4, wd 1e-4, batch 92, 30 epochs, LR drop ×0.2 at epoch 20.
Runtime args in `main_d.py`: repr-dim 384, triplet enc 1 / dec 2, alpha .5, gamma .1,
box-score-thresh .05, min/max instances 3/15, raw-lambda 2.8, seed 140, world-size 2.

### Model variants — the ablation matrix

All five are the same architecture with different `K` and different `(D_h, D_o)`.
**`D_h + D_o` must equal 512** or the decoder's `qk_attn_qpos_proj` (`Linear(512, 384)`)
will not accept the input.

| File | K | D_h / D_o | pose module | reads `rp[...]` | state |
|---|---|---|---|---|---|
| `detailhoi.py` | 6 | 439 / 73 | `build_model_pose` | `body_keypoints` | parses; never traced to a logged run |
| `detailhoi1.py` | 17 | **384 / 128** | `build_model_pose_17` | `keypoints` | **BROKEN — see §7** |
| `detailhoi2.py` | 31 | 256 / 256 | `build_model_pose1` + `keypoint_utils` | `body_keypoints` | parses |
| `detailhoi3.py` | 23 | 256 / 256 | `build_model_pose_17` | `keypoints` | parses; interpolates 17→23 inline |
| `detailhoi4.py` | 3 | 256 / 256 | `build_model_pose_17` | `keypoints` | **BROKEN — see §7** |

`K=6` = head centroid, both wrists, both ankles, torso centre.
`K=3` = head centroid + both wrists.
`K=23` = 17 joints + 6 midpoints (upper arms, thighs, shoulder line, hip line).
`K=31` = 17 joints + 14 midpoints, one per edge of `keypoint_utils.SKELETON_CONNECTIONS`.

### Pose modules
| File | Output written into `region_props[i]` |
|---|---|
| `build_model_pose_17.py` | `['keypoints']` = `[N,17,3]` |
| `build_model_pose.py` | `['keypoints']` = `[N,17,3]` **and** `['body_keypoints']` = `[N,6,3]` |
| `build_model_pose1.py` | uses `keypoint_utils.attach_body_keypoints_to_region_props` → `['body_keypoints']` = `[N,31,3]`, IoU-matched |

All wrap **RTMPose-L** via `mmpose.apis.inference_topdown`, config
`rtmpose-l_8xb256-420e_coco-256x192.py`, weights
`checkpoints/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth`.

Note the association difference: `build_model_pose*.py` assign pose results to human
proposals **by index order**; `keypoint_utils.py` assigns **by box IoU**. The paper says
"associated by construction", which is true of the top-down crop but glosses the index
ordering. If you tighten this, `keypoint_utils`'s IoU matching is the safer path.

### Unchanged infrastructure (inherited from PViC — do not rewrite)
`transformers.py` (decoder; `q_pos` is a dict with `"box"` and `"centre"` keys),
`ops.py` (sinusoidal PE, spatial encodings, prior scores, focal loss),
`utils.py` (`CustomisedDLE` engine, `DataFactory`, `test_vcoco` / `test_hico`),
`detr/`, `h_detr/`, `pocket/`, `hicodet/`, `vcoco/`.

`attention.py` / `attn.py` are unreferenced by any entry point. Dead as far as I can tell.

---

## 4. Environment

From `wandb/run-20260416_224029-o7i1bbcy/files/`:

```
conda env: headhoi        python 3.8.20
torch 1.8.0+cu111         torchvision 0.9.0+cu111
mmpose 1.3.2  mmcv 2.1.0  mmengine 0.10.7  mmdet 3.2.0
ultralytics 8.3.202  pocket 0.5  numpy 1.24.4  wandb 0.24.2
hardware: 2× NVIDIA RTX A6000
```

Launch (see `test_comd.txt` for the full set):
```bash
WANDB_PROJECT="detailhoi" WANDB_ENTITY="cj-monash-university" \
python main_d.py --dataset vcoco --data-root vcoco/ --partitions trainval test \
  --pretrained checkpoints/detr-r50-vcoco.pth \
  --output-dir outputs/<name> --use-wandb
```
Add `--eval` to evaluate a `--resume` checkpoint. HICO-DET: drop `--dataset/--data-root/
--partitions` (defaults are hicodet) and use `checkpoints/detr-r50-hicodet.pth`.

One V-COCO epoch ≈ 19 min on 2× A6000 (from the logs' `Time[Data/Iter.]` ≈ 1140 s).

---

## 5. The human-clarity filter — the biggest trap in this repo

`filter_dataset_vcoco.py` / `filter_dataset_hico.py` run YOLOv8-n over every image and
write a whitelist of filenames. **Both entry points load those whitelists automatically
if the files exist** and silently train and evaluate on the subset.

Present on disk:
```
vcoco/filtered_human_images_train2014.txt     62,911 names
vcoco/filtered_human_images_val2014.txt       30,354 names
hicodet/filtered_human_images_train2015.txt   30,083 names
hicodet/filtered_human_images_test2015.txt     7,658 names
```

Effect on the HOI splits (computed, not quoted):

| Dataset | Split | images with ≥1 interaction | kept | % |
|---|---|---|---|---|
| V-COCO | trainval | 4,969 | 4,404 | 88.6 |
| V-COCO | test | 4,532 | 3,971 | 87.6 |
| HICO-DET | train2015 | 38,118 | 30,083 | 78.9 |
| HICO-DET | test2015 | 9,658 | 7,658 | 79.3 |

**Consequences you must respect:**

- Subset mAP (~0.63) and full-set mAP (~0.71) are **different evaluation sets** and are
  not comparable. Never put them in one column.
- To reproduce a *full-set* number you must move or rename the whitelist files.
- The filter keeps an image if **any class** clears conf 0.8 — `len(res.boxes) > 0`, with
  no person check. It is a proxy for image quality, not a person-specific test. The paper
  states this in the limitations. A person-specific version gates on `res.boxes.cls == 0`.

---

## 6. Experimental record

Scraped from `wandb/*/files/output.log` + `config.yaml` + `wandb-metadata.json`.
V-COCO mAP throughout, in-repo diagnostic evaluator (see §8).

| # | wandb run | program | output-dir | bs | eval set | best | ep30 |
|---|---|---|---|---|---|---|---|
| 1 | `20260303_110635-la2u2snp` | main_d | pvic-detr-r50-vcoco | 96 | full | 0.7148 | 0.7137 |
| 2 | `20260304_112041-tdeuibxc` | main_d | pvic-detr-r50-vcoco | 96 | full | 0.7143 | 0.7137 |
| 3 | `20260412_111858-hxlvflrd` | main_d | pvic-detr-r50-vcoco | 96 | full | 0.7113 | — (stopped, 24 ep) |
| 4 | `20260412_234238-h10290e5` | main_d | pvic-detr-r50-vcoco | 92 | full | 0.7128 | 0.7120 |
| 5 | `20260413_100300-1d6bw3h9` | **main** | pvic-detr-r50-vcoco | 64 | filtered | 0.6395 | 0.6357 |
| 6 | `20260413_214328-4uf4991d` | main_d | pvic-detr-r50-vcoco | 92 | filtered | 0.6367 | 0.6367 |
| 7 | `20260414_072257-3hbovxgv` | main_d | pvic-detr-r50-vcoco | 92 | filtered | 0.6366 | 0.6366 |
| 8 | `20260414_214937-zd4kr4u6` | main_d | pvic-detr-r50-vcoco | 92 | filtered | 0.6322 | 0.6322 |
| 9 | `20260415_092858-q123cho0` | main_d | **new_kept17_ratio3** | 92 | filtered | **0.6396** | **0.6395** |
| 10 | `20260415_213716-l3xzhljz` | main_d | **new_kept31_ratio1** | 92 | filtered | 0.6365 | 0.6351 |
| 11 | `20260416_143629-qdjs1ro6` | main_d | **new_kept23_ratio1** | 92 | filtered | 0.6353 | 0.6351 |
| 12 | `20260416_224029-o7i1bbcy` | main_d | **new_kept23_ratio1** | 92 | filtered | 0.6363 | 0.6344 |
| 13 | `20260417_112427-u7qfdmfo` | main_d | new_kept3_ratio1 | 92 | filtered | — | — |

**Confidence in the run → variant mapping.** `new_kept<K>_ratio<r>` names are
unambiguous: `ratio3` = 384/128, `ratio1` = 256/256. So rows **9–12 are solid** and they
are what the paper's ablation table reports. Row 5 is solid too — `main.py` imports
`pvic`, so it is a genuine baseline. **Rows 1–4 and 6–8 all went to the same generic
output dir with no variant recorded**; which `detailhoi*.py` was imported at the time is
not recoverable. Treat them as unlabelled.

**Why the level breaks at Apr 13.** The whitelists were written Apr 12 00:30–01:16.
`main.py`'s mtime is Apr 13 09:53 and the run using it started Apr 13 10:03 — the filter
block was added minutes before. Runs 1–4 predate the filter reaching the code path they
used, which is why they score ~0.71. *(Inference from timestamps + the mAP break, not
from a record.)*

**Row 13** was stopped by hand — the log tail is `KeyboardInterrupt` during DataLoader
shutdown, not a model error. Separately, `detailhoi4.py` is broken *today* (§7); its mtime
is 6 min after the run started, so it was edited after the interrupt.

**No HICO-DET run appears anywhere in `wandb/`.** Only `outputs/main1-detr-r50-hicodet/
best.pth` (Feb 23) exists. Every HICO-DET number in the paper comes from the author's
original `.docx`.

### Where every number in the paper came from

| Number | Source | Status |
|---|---|---|
| Subset ablation: 63.95 / 63.96 / 63.63 / 63.65 | runs 5, 9, 11/12, 10 | **verified from logs** |
| Convergence + ablation figures | `figures/curves.json` ← all 13 logs | **verified** |
| Dataset / filter counts (Table 2) | computed from the annotation JSONs | **verified** |
| Parameter counts (Table 1) | computed analytically from the layer shapes | **verified** |
| Hyper-parameters, runtime, env | `config.yaml` + `wandb-metadata.json` | **verified** |
| V-COCO 71.28 / 71.42 | the `.docx` draft | **unverified** — no clean full-set PViC run exists |
| All HICO-DET numbers (33.35 / 33.74 / 28.59 / 28.23 / 34.77 / 35.39 / 33.51 / 28.79 / 34.92) | the `.docx` draft | **unverified** — no logs at all |
| Published QPIC / UPT / PViC rows (Table 5) | quoted from memory of the source papers | **must be checked against the PDFs** |

---

## 7. Known defects in the code

**A. `main_d.py` will crash as it currently stands. Fix this first.**
`detailhoi1.py` calls `self.ref_keypoint_head(embeds)` at line 155 but never defines it in
`HumanObjectMatcher.__init__` — `grep -c "self.ref_keypoint_head = " detailhoi1.py` → `0`.
Every other variant defines it. It raises `AttributeError` on the first forward pass.
The fix is to copy the block from `detailhoi3.py:90`:
```python
self.ref_keypoint_head = nn.Sequential(
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 2)
)
```
`detailhoi1.py`'s mtime (Apr 19 05:39) is **two days after the last successful run**, so
this is an unfinished edit, not what run 9 executed. Run 9's result stands; the file no
longer reproduces it.

**B. `detailhoi4.py` does not parse.** `IndentationError: unexpected indent` at line 50 —
`extract_head_and_hands` sits at module level under a comment reading "Add this method to
KeypointEstimator class". It was never added. Two fixes are needed: move the method into
`KeypointEstimator`, **and** make the forward actually reduce 17 keypoints to 3 —
`forward` reads `rp["keypoints"]` (`[N,17,3]`) while `kpts_pe_proj` expects `3*256`, so
even after the syntax fix it will fail on a shape mismatch. K=3 has never run.

**C. `torch.meshgrid` without `indexing='ij'`** in `detailhoi1.py` despite a comment
claiming otherwise. Warning only on torch 1.8; an error on newer torch.

**D. Per-instance Python loops** in every `compute_box_pe` (`for i in range(N)` to compute
the keypoint extent). Vectorisable with a masked min/max. Not a correctness issue.

Verified: everything else in the repo compiles — `detailhoi{,2,3}.py`, `pvic.py`,
`main.py`, `main_d.py`, all pose modules, `utils.py`, `ops.py`, `transformers.py`.

---

## 8. Evaluation protocol — state this in any writeup

`utils.CustomisedDLE.test_vcoco` uses `DetectionAPMeter(24, algorithm='11P')` with
`BoxPairAssociation(min_iou=0.5)`. `main_d.py` carries an explicit warning:

> "NOTE This evaluation results on V-COCO do not necessarily follow the protocol as the
> official evaluation code, and so are only used for diagnostic purposes."

So **~0.71 is not V-COCO `AP_role`** and must never be compared to published V-COCO
figures. The paper handles this with a boxed protocol note in §4.3 and a `†` on the
column. Keep that. HICO-DET uses `DetectionAPMeter(600, algorithm='11P')` with the
standard 138 rare / 462 non-rare split — that one *is* comparable to the literature.

---

## 9. Open work, in priority order

1. **Fix defect A**, then re-run `new_kept17_ratio3` to confirm 0.6396 still reproduces.
   Nothing else is trustworthy until the headline config runs from the current source.
2. **Regenerate the HICO-DET numbers.** `--eval --resume outputs/main1-detr-r50-hicodet/
   best.pth`, with the whitelists moved aside for the full-set number. Four table cells
   in the paper depend on this.
3. **Run a clean full-set PViC baseline** with `main.py` (whitelists moved aside) to
   replace the unverified 71.28.
4. **Remove the batch-size confound.** Run 5 (baseline) used bs 64; runs 9–12 used 92.
   Re-run the baseline at 92.
5. **Multi-seed variance.** Every ablation is seed 140 and the margins are ~0.03–0.3 mAP,
   the same order as the epoch-to-epoch oscillation. Three seeds per row would make the
   ablation table defensible; without it a reviewer can dismiss the whole result.
6. **Verify Table 5's published numbers** against the QPIC / UPT / PViC PDFs.
7. **Verify 6 references** marked `% [verify]` in `detailhoi.tex` and
   `[DETAILS TO VERIFY]` in `DetailHOI.docx`: HORP, HOIGaze, HAGI++, PAIN, GeoVis-GNN,
   VRDiff. They came from the author's draft without full bibliographic detail.
8. **Optional: finish K=3 and K=6.** Both would strengthen the granularity curve at the
   sparse end, where the paper currently has no data.

---

## 10. Editorial decisions already made — do not silently reverse

**The gaze / head-orientation module was cut from the results.** The original `.docx`
claims a "256-dimensional face orientation feature (`face_feat`)" fused with
`centre_feat`. It **does not exist in this repo** — `grep -rn "face_feat\|gaze"` over all
Python outside `detr/ pocket/ h_detr/` returns nothing. (`.vscode/settings.json` has a
stale `./deep_head_pose/code` path, so it may exist elsewhere.) The paper therefore
describes it in §6 as a designed extension and says explicitly that no result depends on
it. **If you find that code, this becomes a real experiment and the paper's framing can
change. If you do not, do not re-introduce the claim.**

**The paper's thesis was moved from the headline gain to the ablations.** +0.39 mAP is
thin. The defensible findings are (a) deterministic interpolation along the skeleton graph
*hurts*, with a mechanistic explanation (a linear map already spans convex combinations),
and (b) the 512-d budget split is a real hyper-parameter — 384/128 is the only
configuration that beats the baseline. A third argument carries weight: on the subset the
baseline peaks 63.95 → ends 63.57 while K=17 peaks 63.96 → ends 63.95, so the prior buys
end-of-schedule *stability* even where peak accuracy ties.

**Everything unfavourable is stated, not hidden**: the rare-class regression
(28.59 → 28.23), the subset tie, the single seed, the batch-size confound, the proxy
filter, the non-standard V-COCO protocol. Keep it that way — each of these is the first
thing a reviewer would find anyway.

---

## 11. Quick reference

```bash
# per-epoch mAP for every run
for d in wandb/run-*/; do echo "$d $(grep -oP 'mAP: \K[0-9.]+' $d/files/output.log | sort -rn | head -1)"; done

# which model variant an entry point currently uses
grep -n "^from detailhoi\|^from pvic" main.py main_d.py

# does the source still compile
for f in detailhoi*.py pvic.py main*.py; do python3 -m py_compile $f && echo "OK $f" || echo "FAIL $f"; done

# disable the human-clarity filter (restore full-set evaluation)
mkdir -p _whitelists_off && mv vcoco/filtered_human_images_*.txt hicodet/filtered_human_images_*.txt _whitelists_off/

# rebuild the paper figures
cd paper && python3 mkfigs.py && python3 mkarch.py && python3 mkflow.py
```

---

## 12. Content folded in from the Progress Review deck (2026-08-24)

At the user's request, material from `Progress Review .pptx` (Chunyu Jiang's candidature
review talk, 18 slides) was added to all three manuscript formats. Full detail is in
`README.md`; summary here so it's not missed:

- **New Fig. 1** (Introduction): a simplified before/after schematic, *not* a reproduction of
  the deck's slide-10 flow chart. That slide includes a "Face Direction" input and a "Human
  Object matcher" enhancement box implementing the head-orientation extension — see §10 above,
  **this is the cut `face_feat` component**. The new figure deliberately omits it and shows
  only the implemented pipeline.
- **New equation** (end of §3.1): the four-term cross-attention decomposition from Conditional
  DETR, in this paper's notation. Renumbers every later equation by +1 in the .tex/.docx.
- **New Fig. 5** (§5.3, new subsection): the deck's attention-decomposition visualisation,
  reassembled from `figures/_attn_source/`. Single-example qualitative figure.
- **§3.3 enrichment**: `ratio1`/`ratio3` shorthand (matches the `new_kept<K>_ratio<r>` wandb run
  names in §6) and the K=6 variable-split note.

**Checked and deliberately not applied**: the deck's slide 9 states the human-clarity filter
as RTMPose-L with different counts (trainval 5,814→5,404, test 4,532→4,332) than what's in the
paper. §5 above marks the paper's YOLOv8-n numbers **verified from logs**, and every ablation
mAP in §6's table was measured on that exact subset — swapping in the deck's numbers would
have made the method description and the results tables inconsistent. Don't make that swap
without first re-deriving which filter/counts the *current* checkpoints actually used.
