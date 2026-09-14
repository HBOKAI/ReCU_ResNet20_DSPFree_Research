# H2B Finite-Width Optimization Report

> H2B optimizes only conservative internal widths around the frozen PASS H2A-v2 datapath. No retraining, QAT, architecture, weight, pow2 exponent, RTL, or synthesis work was performed.

## 1. Frozen source and search discipline

- H2A-v2 result: `C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\H2_FINITE_WIDTH_RESULTS.json`
- R8B checkpoint: `C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\experiments\recu_r8b\recu_r8b_r7_pow2_fc_20260912_225827\best.pt`
- H0 result: `C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\H0_INTEGER_BIAS_SWEEP_RESULTS.json` (INT6 frozen)
- H1MP result: `C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\H1MP_SEARCH_RESULTS.json` (58-node bits/shifts/policies frozen)
- H2A-v2 official TEST reference: **85.03%**.
- Official TEST was not evaluated during H2B search. It was evaluated once after the final TRAIN-derived plan was frozen.
- Search split: CIFAR-10 TRAIN calibration subset and non-overlapping TRAIN search-validation subset.
- Family order: `inner -> output -> fc_accumulator -> final_logits -> gap`.

## 2. Fixed invariants

| Item | Frozen value |
|---|---:|
| Stem binary accumulator | INT11 |
| Cin=16 binary accumulator | INT9 |
| Cin=32 binary accumulator | INT10 |
| Cin=64 binary accumulator | INT11 |
| H0 bias | INT6 |
| H1MP plan | 58 nodes / 18 residual alignments |
| GAP semantics | q_gap=q_sum; s_gap=s_input+6 |
| FC accumulator before H2B | INT24 |
| Final logits before H2B | INT24 |
| Binary overflow | 0 by exact-width contract |

## 3. H2A-v2 TRAIN validation baseline

- Search-validation accuracy: **88.95%**.
- H2A-v2 official TEST reference: **85.03%**; this value was not used for candidate decisions.
- H2A-v2 GAP verification: `scale_metadata_adjustment`; arithmetic right shift applied = `False`.

## 4. Width sweep results

### QRPReLU inner

| Candidate width | Action | Validation accuracy | Delta vs H2A validation | Validation saturation | Overflow |
|---:|---|---:|---:|---:|---:|
| 29 | `accept` | 88.95% | +0.00 pp | 108157730 | 0 |
| ↳ `qrprelu.layer1.0.inner_add` | safe [-67112264, 73736042]; cal [-67109246, 71970373]; val [-67109246, 67776069]; sat 0 | | |
| ↳ `qrprelu.layer1.1.inner_add` | safe [-67132350, 74744782]; cal [-67132350, 74744782]; val [-67132350, 74744782]; sat 0 | | |
| ↳ `qrprelu.layer1.2.inner_add` | safe [-134245237, 141233231]; cal [-117468021, 75469963]; val [-119565173, 75469963]; sat 0 | | |
| ↳ `qrprelu.layer2.0.inner_add` | safe [-68209433, 70730438]; cal [-67109802, 67372454]; val [-67109802, 67372454]; sat 0 | | |
| ↳ `qrprelu.layer2.1.inner_add` | safe [-268437176, 273088227]; cal [-121636272, 98564688]; val [-117441968, 104856144]; sat 0 | | |
| ↳ `qrprelu.layer2.2.inner_add` | safe [-66747385, 68607409]; cal [-66510335, 67735646]; val [-66510335, 67735646]; sat 0 | | |
| ↳ `qrprelu.layer3.0.inner_add` | safe [-67108845, 67713386]; cal [-67108845, 66846956]; val [-67108845, 66846956]; sat 0 | | |
| ↳ `qrprelu.layer3.1.inner_add` | safe [-134218465, 135076398]; cal [-134217644, 125829204]; val [-134217644, 132120660]; sat 0 | | |
| ↳ `qrprelu.layer3.2.inner_add` | safe [-134218051, 132130797]; cal [-134213809, 132130344]; val [-134211346, 132126505]; sat 0 | | |
| 28 | `accept` | 88.95% | +0.00 pp | 108157730 | 0 |
| ↳ `qrprelu.layer1.0.inner_add` | safe [-67112264, 73736042]; cal [-67109246, 71970373]; val [-67109246, 67776069]; sat 0 | | |
| ↳ `qrprelu.layer1.1.inner_add` | safe [-67132350, 74744782]; cal [-67132350, 74744782]; val [-67132350, 74744782]; sat 0 | | |
| ↳ `qrprelu.layer1.2.inner_add` | safe [-134245237, 141233231]; cal [-117468021, 75469963]; val [-119565173, 75469963]; sat 0 | | |
| ↳ `qrprelu.layer2.0.inner_add` | safe [-68209433, 70730438]; cal [-67109802, 67372454]; val [-67109802, 67372454]; sat 0 | | |
| ↳ `qrprelu.layer2.1.inner_add` | safe [-268437176, 273088227]; cal [-121636272, 98564688]; val [-117441968, 104856144]; sat 0 | | |
| ↳ `qrprelu.layer2.2.inner_add` | safe [-66747385, 68607409]; cal [-66510335, 67735646]; val [-66510335, 67735646]; sat 0 | | |
| ↳ `qrprelu.layer3.0.inner_add` | safe [-67108845, 67713386]; cal [-67108845, 66846956]; val [-67108845, 66846956]; sat 0 | | |
| ↳ `qrprelu.layer3.1.inner_add` | safe [-134218465, 135076398]; cal [-134217644, 125829204]; val [-134217644, 132120660]; sat 0 | | |
| ↳ `qrprelu.layer3.2.inner_add` | safe [-134218051, 132130797]; cal [-134213809, 132130344]; val [-134211346, 132126505]; sat 0 | | |
| 27 | `accept` | 88.90% | +0.05 pp | 110149642 | 2004981 |
| ↳ `qrprelu.layer1.0.inner_add` | safe [-67112264, 73736042]; cal [-67109246, 71970373]; val [-67109246, 67776069]; sat 7 | | |
| ↳ `qrprelu.layer1.1.inner_add` | safe [-67132350, 74744782]; cal [-67132350, 74744782]; val [-67132350, 74744782]; sat 61546 | | |
| ↳ `qrprelu.layer1.2.inner_add` | safe [-134245237, 141233231]; cal [-117468021, 75469963]; val [-119565173, 75469963]; sat 805974 | | |
| ↳ `qrprelu.layer2.0.inner_add` | safe [-68209433, 70730438]; cal [-67109802, 67372454]; val [-67109802, 67372454]; sat 31147 | | |
| ↳ `qrprelu.layer2.1.inner_add` | safe [-268437176, 273088227]; cal [-121636272, 98564688]; val [-117441968, 104856144]; sat 382504 | | |
| ↳ `qrprelu.layer2.2.inner_add` | safe [-66747385, 68607409]; cal [-66510335, 67735646]; val [-66510335, 67735646]; sat 613 | | |
| ↳ `qrprelu.layer3.0.inner_add` | safe [-67108845, 67713386]; cal [-67108845, 66846956]; val [-67108845, 66846956]; sat 0 | | |
| ↳ `qrprelu.layer3.1.inner_add` | safe [-134218465, 135076398]; cal [-134217644, 125829204]; val [-134217644, 132120660]; sat 278449 | | |
| ↳ `qrprelu.layer3.2.inner_add` | safe [-134218051, 132130797]; cal [-134213809, 132130344]; val [-134208681, 132126505]; sat 444741 | | |
| 26 | `reject_rollback` | 88.32% | +0.63 pp | 173233413 | 65386160 |
| ↳ `qrprelu.layer1.0.inner_add` | safe [-67112264, 73736042]; cal [-67109246, 71970373]; val [-67109246, 67776069]; sat 6761869 | | |
| ↳ `qrprelu.layer1.1.inner_add` | safe [-67132350, 74744782]; cal [-67132350, 74744782]; val [-67132350, 74744782]; sat 11924125 | | |
| ↳ `qrprelu.layer1.2.inner_add` | safe [-134245237, 141233231]; cal [-117468021, 75469963]; val [-117468021, 75469963]; sat 7424753 | | |
| ↳ `qrprelu.layer2.0.inner_add` | safe [-68209433, 70730438]; cal [-67109802, 67372454]; val [-67109802, 67372454]; sat 4874591 | | |
| ↳ `qrprelu.layer2.1.inner_add` | safe [-268437176, 273088227]; cal [-121636272, 98564688]; val [-113247664, 104856144]; sat 12873445 | | |
| ↳ `qrprelu.layer2.2.inner_add` | safe [-66747385, 68607409]; cal [-66510335, 67735646]; val [-66510335, 67735646]; sat 6500961 | | |
| ↳ `qrprelu.layer3.0.inner_add` | safe [-67108845, 67713386]; cal [-67108845, 66846956]; val [-67108845, 66846956]; sat 3574553 | | |
| ↳ `qrprelu.layer3.1.inner_add` | safe [-134218465, 135076398]; cal [-134217644, 125829204]; val [-134217644, 132120660]; sat 5563892 | | |
| ↳ `qrprelu.layer3.2.inner_add` | safe [-134218051, 132130797]; cal [-134213809, 132130344]; val [-130017777, 132126505]; sat 5887971 | | |

- Frozen QRPReLU inner width: **INT27**.

Internal QRPReLU trace coverage per candidate includes shifted input, xi1 offset, xi2 offset, inner add, negative branch, output-before-H1MP-requantization, and final finite output.

### QRPReLU output

| Candidate width | Action | Validation accuracy | Delta vs H2A validation | Validation saturation | Overflow |
|---:|---|---:|---:|---:|---:|
| 37 | `accept` | 88.90% | +0.05 pp | 110149642 | 2004981 |
| ↳ `qrprelu.layer1.0.negative_branch` | safe [-18628667904, 19902298368]; cal [-14777063168, 17666962944]; val [-14777063168, 17666962944]; sat 0 | | |
| ↳ `qrprelu.layer1.0.output` | safe [-18628667904, 19902298368]; cal [-14777063168, 16642998272]; val [-14777063168, 16642998272]; sat 0 | | |
| ↳ `qrprelu.layer1.1.negative_branch` | safe [-18293655040, 19661949952]; cal [-16807598592, 17333649664]; val [-16807598592, 17015268864]; sat 0 | | |
| ↳ `qrprelu.layer1.1.output` | safe [-18293655040, 19661949952]; cal [-16807598592, 16642998272]; val [-16807598592, 16642998272]; sat 0 | | |
| ↳ `qrprelu.layer1.2.negative_branch` | safe [-38130886656, 36682345216]; cal [-16345495808, 18234432000]; val [-15808624896, 17370499840]; sat 0 | | |
| ↳ `qrprelu.layer1.2.output` | safe [-38130886656, 36682345216]; cal [-16345495808, 19327352832]; val [-15808624896, 19327352832]; sat 0 | | |
| ↳ `qrprelu.layer2.0.negative_branch` | safe [-18877615616, 18854442496]; cal [-16569144064, 16096307200]; val [-16569144064, 16096307200]; sat 0 | | |
| ↳ `qrprelu.layer2.0.output` | safe [-18877615616, 18854442496]; cal [-16569144064, 16642998272]; val [-16569144064, 16642998272]; sat 0 | | |
| ↳ `qrprelu.layer2.1.negative_branch` | safe [-69344604160, 71044129024]; cal [-18402727168, 17567623936]; val [-17012980480, 17346757632]; sat 0 | | |
| ↳ `qrprelu.layer2.1.output` | safe [-69344604160, 71044129024]; cal [-18402727168, 25232932864]; val [-17012980480, 26843545600]; sat 0 | | |
| ↳ `qrprelu.layer2.2.negative_branch` | safe [-134387948, 146732004]; cal [-109214234, 134055398]; val [-105019930, 134055398]; sat 0 | | |
| ↳ `qrprelu.layer2.2.output` | safe [-134387948, 146732004]; cal [-109214234, 130023424]; val [-105019930, 130023424]; sat 0 | | |
| ↳ `qrprelu.layer3.0.negative_branch` | safe [-17179869184, 16642998272]; cal [-8505189120, 9015967744]; val [-8505189120, 9015967744]; sat 0 | | |
| ↳ `qrprelu.layer3.0.output` | safe [-17179869184, 16642998272]; cal [-8505189120, 16642998272]; val [-8505189120, 16642998272]; sat 0 | | |
| ↳ `qrprelu.layer3.1.negative_branch` | safe [-34579987712, 35374260736]; cal [-16679634816, 17179628032]; val [-11811401216, 16642757120]; sat 0 | | |
| ↳ `qrprelu.layer3.1.output` | safe [-34579987712, 35374260736]; cal [-16679634816, 32212254720]; val [-11811401216, 33822867456]; sat 0 | | |
| ↳ `qrprelu.layer3.2.negative_branch` | safe [-34359738368, 33822867456]; cal [-88758507, 1095871575]; val [-32135424, 1078755839]; sat 0 | | |
| ↳ `qrprelu.layer3.2.output` | safe [-34359738368, 33822867456]; cal [-88758507, 33822867456]; val [-32135424, 33822867456]; sat 0 | | |
| 36 | `accept` | 88.90% | +0.05 pp | 110149642 | 2004981 |
| ↳ `qrprelu.layer1.0.negative_branch` | safe [-18628667904, 19902298368]; cal [-14777063168, 17666962944]; val [-14777063168, 17666962944]; sat 0 | | |
| ↳ `qrprelu.layer1.0.output` | safe [-18628667904, 19902298368]; cal [-14777063168, 16642998272]; val [-14777063168, 16642998272]; sat 0 | | |
| ↳ `qrprelu.layer1.1.negative_branch` | safe [-18293655040, 19661949952]; cal [-16807598592, 17333649664]; val [-16807598592, 17015268864]; sat 0 | | |
| ↳ `qrprelu.layer1.1.output` | safe [-18293655040, 19661949952]; cal [-16807598592, 16642998272]; val [-16807598592, 16642998272]; sat 0 | | |
| ↳ `qrprelu.layer1.2.negative_branch` | safe [-38130886656, 36682345216]; cal [-16345495808, 18234432000]; val [-15808624896, 17370499840]; sat 0 | | |
| ↳ `qrprelu.layer1.2.output` | safe [-38130886656, 36682345216]; cal [-16345495808, 19327352832]; val [-15808624896, 19327352832]; sat 0 | | |
| ↳ `qrprelu.layer2.0.negative_branch` | safe [-18877615616, 18854442496]; cal [-16569144064, 16096307200]; val [-16569144064, 16096307200]; sat 0 | | |
| ↳ `qrprelu.layer2.0.output` | safe [-18877615616, 18854442496]; cal [-16569144064, 16642998272]; val [-16569144064, 16642998272]; sat 0 | | |
| ↳ `qrprelu.layer2.1.negative_branch` | safe [-69344604160, 71044129024]; cal [-18402727168, 17567623936]; val [-17012980480, 17346757632]; sat 0 | | |
| ↳ `qrprelu.layer2.1.output` | safe [-69344604160, 71044129024]; cal [-18402727168, 25232932864]; val [-17012980480, 26843545600]; sat 0 | | |
| ↳ `qrprelu.layer2.2.negative_branch` | safe [-134387948, 146732004]; cal [-109214234, 134055398]; val [-105019930, 134055398]; sat 0 | | |
| ↳ `qrprelu.layer2.2.output` | safe [-134387948, 146732004]; cal [-109214234, 130023424]; val [-105019930, 130023424]; sat 0 | | |
| ↳ `qrprelu.layer3.0.negative_branch` | safe [-17179869184, 16642998272]; cal [-8505189120, 9015967744]; val [-8505189120, 9015967744]; sat 0 | | |
| ↳ `qrprelu.layer3.0.output` | safe [-17179869184, 16642998272]; cal [-8505189120, 16642998272]; val [-8505189120, 16642998272]; sat 0 | | |
| ↳ `qrprelu.layer3.1.negative_branch` | safe [-34579987712, 35374260736]; cal [-16679634816, 17179628032]; val [-11811401216, 16642757120]; sat 0 | | |
| ↳ `qrprelu.layer3.1.output` | safe [-34579987712, 35374260736]; cal [-16679634816, 32212254720]; val [-11811401216, 33822867456]; sat 0 | | |
| ↳ `qrprelu.layer3.2.negative_branch` | safe [-34359738368, 33822867456]; cal [-88758507, 1095871575]; val [-32135424, 1078755839]; sat 0 | | |
| ↳ `qrprelu.layer3.2.output` | safe [-34359738368, 33822867456]; cal [-88758507, 33822867456]; val [-32135424, 33822867456]; sat 0 | | |
| 35 | `accept` | 88.80% | +0.15 pp | 110221465 | 2077119 |
| ↳ `qrprelu.layer1.0.negative_branch` | safe [-18628667904, 19902298368]; cal [-14777063168, 17666962944]; val [-14777063168, 17666962944]; sat 4 | | |
| ↳ `qrprelu.layer1.0.output` | safe [-18628667904, 19902298368]; cal [-14777063168, 16642998272]; val [-14777063168, 16642998272]; sat 0 | | |
| ↳ `qrprelu.layer1.1.negative_branch` | safe [-18293655040, 19661949952]; cal [-16807598592, 17333649664]; val [-16807598592, 17015268864]; sat 0 | | |
| ↳ `qrprelu.layer1.1.output` | safe [-18293655040, 19661949952]; cal [-16807598592, 16642998272]; val [-16807598592, 16642998272]; sat 0 | | |
| ↳ `qrprelu.layer1.2.negative_branch` | safe [-38130886656, 36682345216]; cal [-16345495808, 18234432000]; val [-15808624896, 17370499840]; sat 6 | | |
| ↳ `qrprelu.layer1.2.output` | safe [-38130886656, 36682345216]; cal [-16345495808, 19327352832]; val [-15808624896, 19327352832]; sat 151 | | |
| ↳ `qrprelu.layer2.0.negative_branch` | safe [-18877615616, 18854442496]; cal [-16569144064, 16096307200]; val [-16569144064, 16096307200]; sat 0 | | |
| ↳ `qrprelu.layer2.0.output` | safe [-18877615616, 18854442496]; cal [-16569144064, 16642998272]; val [-16569144064, 16642998272]; sat 0 | | |
| ↳ `qrprelu.layer2.1.negative_branch` | safe [-69344604160, 71044129024]; cal [-18402727168, 17567623936]; val [-17012980480, 17346757632]; sat 5 | | |
| ↳ `qrprelu.layer2.1.output` | safe [-69344604160, 71044129024]; cal [-18402727168, 25232932864]; val [-17012980480, 26843545600]; sat 2464 | | |
| ↳ `qrprelu.layer2.2.negative_branch` | safe [-134387948, 146732004]; cal [-109214234, 134055398]; val [-105019930, 134055398]; sat 0 | | |
| ↳ `qrprelu.layer2.2.output` | safe [-134387948, 146732004]; cal [-109214234, 130023424]; val [-105019930, 130023424]; sat 0 | | |
| ↳ `qrprelu.layer3.0.negative_branch` | safe [-17179869184, 16642998272]; cal [-8505189120, 9015967744]; val [-8505189120, 9015967744]; sat 0 | | |
| ↳ `qrprelu.layer3.0.output` | safe [-17179869184, 16642998272]; cal [-8505189120, 16642998272]; val [-8505189120, 16642998272]; sat 0 | | |
| ↳ `qrprelu.layer3.1.negative_branch` | safe [-34579987712, 35374260736]; cal [-16679634816, 17179628032]; val [-11811401216, 16642757120]; sat 0 | | |
| ↳ `qrprelu.layer3.1.output` | safe [-34579987712, 35374260736]; cal [-16679634816, 32212254720]; val [-11811401216, 33822867456]; sat 11947 | | |
| ↳ `qrprelu.layer3.2.negative_branch` | safe [-34359738368, 33822867456]; cal [-88758507, 1095871575]; val [-32135424, 1078755839]; sat 0 | | |
| ↳ `qrprelu.layer3.2.output` | safe [-34359738368, 33822867456]; cal [-88758507, 33822867456]; val [-32135424, 33822867456]; sat 57561 | | |
| 34 | `reject_rollback` | 85.44% | +3.51 pp | 130482969 | 24044060 |
| ↳ `qrprelu.layer1.0.negative_branch` | safe [-18628667904, 19902298368]; cal [-14777063168, 17666962944]; val [-14777063168, 17666962944]; sat 2812842 | | |
| ↳ `qrprelu.layer1.0.output` | safe [-18628667904, 19902298368]; cal [-14777063168, 16642998272]; val [-14777063168, 16642998272]; sat 4884039 | | |
| ↳ `qrprelu.layer1.1.negative_branch` | safe [-18293655040, 19661949952]; cal [-16807598592, 17333649664]; val [-16807598592, 16598283264]; sat 2389006 | | |
| ↳ `qrprelu.layer1.1.output` | safe [-18293655040, 19661949952]; cal [-16807598592, 16642998272]; val [-16807598592, 16642998272]; sat 6718485 | | |
| ↳ `qrprelu.layer1.2.negative_branch` | safe [-38130886656, 36682345216]; cal [-16345495808, 18234432000]; val [-15540189440, 12865722880]; sat 567957 | | |
| ↳ `qrprelu.layer1.2.output` | safe [-38130886656, 36682345216]; cal [-16345495808, 19327352832]; val [-15540189440, 15569256448]; sat 551706 | | |
| ↳ `qrprelu.layer2.0.negative_branch` | safe [-18877615616, 18854442496]; cal [-16569144064, 16096307200]; val [-16569144064, 13495627008]; sat 842062 | | |
| ↳ `qrprelu.layer2.0.output` | safe [-18877615616, 18854442496]; cal [-16569144064, 16642998272]; val [-16569144064, 16642998272]; sat 1132404 | | |
| ↳ `qrprelu.layer2.1.negative_branch` | safe [-69344604160, 71044129024]; cal [-18402727168, 17567623936]; val [-17012980480, 13809527552]; sat 558058 | | |
| ↳ `qrprelu.layer2.1.output` | safe [-69344604160, 71044129024]; cal [-18402727168, 25232932864]; val [-17012980480, 25769803776]; sat 880006 | | |
| ↳ `qrprelu.layer2.2.negative_branch` | safe [-134387948, 146732004]; cal [-109214234, 134055398]; val [-105019930, 104695270]; sat 0 | | |
| ↳ `qrprelu.layer2.2.output` | safe [-134387948, 146732004]; cal [-109214234, 130023424]; val [-105019930, 130023424]; sat 0 | | |
| ↳ `qrprelu.layer3.0.negative_branch` | safe [-17179869184, 16642998272]; cal [-8505189120, 9015967744]; val [-8505189120, 8995596416]; sat 1162 | | |
| ↳ `qrprelu.layer3.0.output` | safe [-17179869184, 16642998272]; cal [-8505189120, 16642998272]; val [-8505189120, 16642998272]; sat 151085 | | |
| ↳ `qrprelu.layer3.1.negative_branch` | safe [-34579987712, 35374260736]; cal [-16679634816, 17179628032]; val [-11811401216, 13421531648]; sat 5725 | | |
| ↳ `qrprelu.layer3.1.output` | safe [-34579987712, 35374260736]; cal [-16679634816, 32212254720]; val [-11811401216, 33822867456]; sat 249425 | | |
| ↳ `qrprelu.layer3.2.negative_branch` | safe [-34359738368, 33822867456]; cal [-88758507, 1095871575]; val [-32135424, 1078755839]; sat 0 | | |
| ↳ `qrprelu.layer3.2.output` | safe [-34359738368, 33822867456]; cal [-88758507, 33822867456]; val [-32135424, 33822867456]; sat 740087 | | |

- Frozen QRPReLU output width: **INT35**.

Internal QRPReLU trace coverage per candidate includes shifted input, xi1 offset, xi2 offset, inner add, negative branch, output-before-H1MP-requantization, and final finite output.

### FC accumulator

| Candidate width | Action | Validation accuracy | Delta vs H2A validation | Validation saturation | Overflow |
|---:|---|---:|---:|---:|---:|
| 23 | `accept` | 88.80% | +0.15 pp | 110221465 | 2077119 |
| ↳ `fc.accumulator` | safe [-273654, 274068]; cal [-35580, 59376]; val [-36762, 60532]; sat 0 | | |
| 22 | `accept` | 88.80% | +0.15 pp | 110221465 | 2077119 |
| ↳ `fc.accumulator` | safe [-273654, 274068]; cal [-35580, 59376]; val [-36762, 60532]; sat 0 | | |
| 21 | `accept` | 88.80% | +0.15 pp | 110221465 | 2077119 |
| ↳ `fc.accumulator` | safe [-273654, 274068]; cal [-35580, 59376]; val [-36762, 60532]; sat 0 | | |
| 20 | `accept` | 88.80% | +0.15 pp | 110221465 | 2077119 |
| ↳ `fc.accumulator` | safe [-273654, 274068]; cal [-35580, 59376]; val [-36762, 60532]; sat 0 | | |
| 19 | `accept` | 88.80% | +0.15 pp | 110221465 | 2077119 |
| ↳ `fc.accumulator` | safe [-273654, 274068]; cal [-35580, 59376]; val [-36762, 60532]; sat 0 | | |
| 18 | `accept` | 88.80% | +0.15 pp | 110221465 | 2077119 |
| ↳ `fc.accumulator` | safe [-273654, 274068]; cal [-35580, 59376]; val [-36762, 60532]; sat 0 | | |
| 17 | `accept` | 88.80% | +0.15 pp | 110221465 | 2077119 |
| ↳ `fc.accumulator` | safe [-273654, 274068]; cal [-35580, 59376]; val [-36762, 60532]; sat 0 | | |
| 16 | `accept` | 88.75% | +0.20 pp | 110222512 | 2078166 |
| ↳ `fc.accumulator` | safe [-273654, 274068]; cal [-35580, 59376]; val [-36762, 60532]; sat 1047 | | |
| 15 | `reject_rollback` | 81.69% | +7.26 pp | 110233942 | 2089596 |
| ↳ `fc.accumulator` | safe [-273654, 274068]; cal [-35580, 59376]; val [-36762, 60532]; sat 12477 | | |

- Frozen FC accumulator width: **INT16**.

### Final logits

| Candidate width | Action | Validation accuracy | Delta vs H2A validation | Validation saturation | Overflow |
|---:|---|---:|---:|---:|---:|
| 23 | `accept` | 88.75% | +0.20 pp | 110222512 | 2078166 |
| ↳ `final_logits` | safe [-273910, 273812]; cal [-35868, 59088]; val [-33056, 33343]; sat 0 | | |
| 22 | `accept` | 88.75% | +0.20 pp | 110222512 | 2078166 |
| ↳ `final_logits` | safe [-273910, 273812]; cal [-35868, 59088]; val [-33056, 33343]; sat 0 | | |
| 21 | `accept` | 88.75% | +0.20 pp | 110222512 | 2078166 |
| ↳ `final_logits` | safe [-273910, 273812]; cal [-35868, 59088]; val [-33056, 33343]; sat 0 | | |
| 20 | `accept` | 88.75% | +0.20 pp | 110222512 | 2078166 |
| ↳ `final_logits` | safe [-273910, 273812]; cal [-35868, 59088]; val [-33056, 33343]; sat 0 | | |
| 19 | `accept` | 88.75% | +0.20 pp | 110222512 | 2078166 |
| ↳ `final_logits` | safe [-273910, 273812]; cal [-35868, 59088]; val [-33056, 33343]; sat 0 | | |
| 18 | `accept` | 88.75% | +0.20 pp | 110222512 | 2078166 |
| ↳ `final_logits` | safe [-273910, 273812]; cal [-35868, 59088]; val [-33056, 33343]; sat 0 | | |
| 17 | `accept` | 88.75% | +0.20 pp | 110222512 | 2078166 |
| ↳ `final_logits` | safe [-273910, 273812]; cal [-35868, 59088]; val [-33056, 33343]; sat 0 | | |
| 16 | `accept` | 88.75% | +0.20 pp | 110222842 | 2078496 |
| ↳ `final_logits` | safe [-273910, 273812]; cal [-35868, 59088]; val [-33056, 33343]; sat 330 | | |
| 15 | `reject_rollback` | 83.40% | +5.55 pp | 110235205 | 2090859 |
| ↳ `final_logits` | safe [-273910, 273812]; cal [-35868, 59088]; val [-33056, 33343]; sat 12693 | | |

- Frozen Final logits width: **INT16**.

### GAP accumulator

| Candidate width | Action | Validation accuracy | Delta vs H2A validation | Validation saturation | Overflow |
|---:|---|---:|---:|---:|---:|
| 15 | `accept` | 88.75% | +0.20 pp | 110222842 | 2078496 |
| ↳ `gap.sum` | safe [-32768, 32704]; cal [0, 10683]; val [0, 9627]; sat 0 | | |
| 14 | `accept` | 88.75% | +0.20 pp | 110222851 | 2078505 |
| ↳ `gap.sum` | safe [-32768, 32704]; cal [0, 10683]; val [0, 9627]; sat 9 | | |
| 13 | `accept` | 88.65% | +0.30 pp | 110229555 | 2085270 |
| ↳ `gap.sum` | safe [-32768, 32704]; cal [0, 10683]; val [0, 9627]; sat 6894 | | |
| 12 | `reject_rollback` | 87.37% | +1.58 pp | 110279318 | 2135178 |
| ↳ `gap.sum` | safe [-32768, 32704]; cal [0, 10683]; val [0, 9627]; sat 57599 | | |

- Frozen GAP accumulator width: **INT13**.

## 5. Final frozen plan

1. QRPReLU inner final maximum width: **INT27**.
2. QRPReLU output final maximum width: **INT35**.
3. FC accumulator final width: **INT16**.
4. Final logits final width: **INT16**.
5. GAP sum accumulator final width: **INT13**; deferred scaling remains `q_gap=q_sum`, `s_gap=s_input+6`.
- Final TRAIN validation accuracy: **88.65%**.
- Final TRAIN validation delta vs H2A baseline: **+0.30 pp**.
- Final TRAIN validation H2 finite saturation: **2085270**; overflow: **2085270**.

## 6. Official CIFAR-10 TEST after freeze

- Official TEST accuracy: **84.93%**.
- Delta vs H2A-v2 / 85.03%: **-0.10 pp**.
- Final H2 finite saturation: **2078706**; total reported saturation including frozen H1MP boundaries: **110040676**.
- Final H2 finite overflow: **2078706**; binary overflow remains **0**.
- Residual alignment: **18/18**, equivalence all = `True`.

## 7. Verification and stopping condition

- Full tests: **92 / 92 PASS**.
- H2B width decisions used TRAIN-derived data only; official TEST was not used for accept/reject.
- H2B completed and stopped. No RTL or synthesis was started.
