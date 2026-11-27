### Nov 8 Decision: split by lesion_id, not by image

- Alternatives: random split by image; group by lesion but without stratifying by class
- Why: 10,015 images but only 7,470 lesions, some lesions have 2-6 photos
- Because: photos of one lesion look almost the same. If some are in train and some in test, the model can recognize the lesion instead of learning to generalize, so the test score is inflated and no longer measures performance on new lesions. I stratified by class as well so every class (even df, 115 images) appears in train, validation and test.
- Evidence: split 7153 / 1431 / 1431 images; assertions show no lesion_id is shared between parts and all 7 classes appear in each.
- Requirement: R6

### Nov 8 Decision: fit imputer and scaler on train only

- Alternatives: fit preprocessing on the whole dataset
- Because: the median age and the mean and standard deviation used for scaling are learned from data. If they were computed using validation or test rows, information about unseen data would leak into the pipeline and results would look better than reality. A new patient in a clinic is never available when the model is built. I put every step in one sklearn Pipeline and call fit() on train only. handle_unknown="ignore" stops a category unseen in training from crashing the encoder.
- Requirement: R6

### Nov 9 Decision: report macro F1 and referral sensitivity, not only accuracy

- Alternatives: accuracy only
- Evidence: always-predict-nv gets 0.669 accuracy but macro F1 0.115 and referral sensitivity 0; image-only logistic regression had almost the same accuracy (0.674) but macro F1 0.469
- Because: nv is about 67% of the data, so accuracy rewards ignoring rare classes. Macro F1 gives every class the same weight. Referral sensitivity measures what the clinic cares about: missing a high-concern lesion (mel, bcc, akiec) is worse than an unnecessary referral.

### Nov 10 Decision: choose logistic-regression C on validation macro F1

- Alternatives: default C, choose on test
- Evidence: metadata-only underfits (train F1 about 0.2, validation about 0.19 at every C); image-only overfits as C grows (train F1 0.53 to 0.84, validation peaks at C = 0.01); chosen C: 0.001 metadata, 0.01 image, 0.01 hybrid
- Because: validation is the only set we may tune on. The test set stays locked.

### Nov 19 Decision: evaluate per lesion (average the 7 probabilities of a lesion's images)

- Alternatives: evaluate per image
- Why: the brief defines referral rate as lesions / all lesions; my first scripts counted images
- Because: the system decides about lesions, and photos of one lesion are not independent. Counting images would count a lesion several times in the referral rate and sensitivity and make bootstrap intervals too narrow. Averaging the probabilities gives one decision per lesion. The rule was fixed before looking at any test result.
- Effect: validation has 1068 lesions (168 high-concern). Lesion-level numbers are not comparable with the earlier image-level numbers.

### Nov 19 Decision: bootstrap lesions, not images, for confidence intervals

- Because: resampling whole lesions respects the dependence between photos of one lesion. Comparisons between two models use a paired bootstrap (same resampled lesions for both), because the interval of the difference is what matters.
- Result (baselines): hybrid vs image-only logistic regression intervals overlap on every measure, so I did not claim the metadata helps.

### Nov 20 Decision: fine-tune EfficientNet-B0 as main design 1 (4,016,515 parameters, 16.4 MB file)

- Alternatives: frozen ResNet18 + logistic regression (my baseline); MobileNetV3; ResNet50 or transformers (larger and slower on CPU)
- Why: the instructor asked for improving existing deep-learning approaches; the baseline reached lesion-level macro F1 about 0.51-0.55, fine-tuned runs reached about 0.70-0.77; EfficientNet-B0 is small enough for R5 (100 MB limit, CPU) and trains in about 70 s per epoch on a free T4 GPU.
- Still to check: full-pipeline CPU time per lesion (preprocessing + network + decision), must be at most 1 second.

### Nov 20 Decision: training settings (augmentation on train only, class weights from train counts, best epoch by validation macro F1)

- Because: validation and test must show real unchanged images, so augmentation is a training aid only. Class weights computed from validation or test counts would leak information. Epoch choice uses validation only; the test rows are never loaded in the training script.
- Known weakness: picking the best of 12 noisy validation scores is slightly optimistic, and macro F1 is noisy because rare classes have few validation lesions (df about 10-16).
- Known weakness 2: the first epoch logs compared a weighted or focal train loss with a plain validation loss, so those curves are not used in the report. Learning curves in the report use the plain-loss runs (same loss for train and validation).

### Nov 21 Decision: compare three losses (plain, weighted, focal) and repeat with 3 seeds

- Evidence from seed 42: all three had AUROC of p_high about 0.95 and every paired interval against plain included 0, so one run could not separate them. Focal looked best on seed 42 (macro F1 0.767) but was 0.699 on seeds 1 and 2.
- Because: a single seed can make a loss look best by chance.
- Selection rule (written BEFORE running p10): primary measure is the mean sensitivity at a 25% referral cap over 3 seeds (AUROC as a check). Weighted or focal replaces plain only if the paired 95% interval of the difference excludes 0. If both do, take the one with the higher mean. If neither does, keep weighted loss as the standard, simplest way to handle imbalance.

### Nov 21 Result of p10 and decision: base loss = plain

- 3-seed means, plain / weighted / focal: sensitivity at 25% cap 0.891 / 0.867 / 0.881; macro F1 0.751 / 0.734 / 0.721; mel recall 0.548 / 0.724 / 0.705; AUROC of p_high 0.952 / 0.947 / 0.952; accuracy 0.876 / 0.831 / 0.844.
- Paired bootstrap, sensitivity at 25%: weighted - plain -0.025 (-0.055 to 0.007); focal - plain -0.007 (-0.036 to 0.023); focal - weighted 0.018 (-0.012 to 0.055). AUROC differences about 0. Neither weighted nor focal replaces plain.
- Deviation: my written fallback said "keep weighted", which defaults to the intervention although it showed no benefit. I changed it to "keep plain" before running any further experiment, and I record the deviation here.
- Report: a mixed result for the minority-class ablation (task D).
- Because: weighted and focal loss raise mel recall at the argmax, but they do not improve the ranking (AUROC of p_high 0.947 / 0.952 vs 0.952) or the sensitivity at the 25% cap (paired intervals include 0; weighted is even 0.025 lower on average). They also lower accuracy (0.831 and 0.844 vs 0.876) and nv recall, so they mostly move the operating point. Plain loss is the simplest, has the best accuracy and macro F1, and there is no evidence that the extras help at the decision point that matters. (Drafted with Claude's help; rehearse in my own words.)

### Nov 22 Result: per-class recall, 3-seed means (plain / weighted / focal)

- akiec 0.566 / 0.677 / 0.657; bcc 0.812 / 0.826 / 0.806; bkl 0.683 / 0.788 / 0.731; df 0.700 / 0.667 / 0.733 (10 lesions); mel 0.548 / 0.724 / 0.705; nv 0.960 / 0.857 / 0.887; vasc 0.769 / 0.872 / 0.846 (13 lesions). Balanced accuracy 0.720 / 0.773 / 0.766.
- Reading: weighted and focal raise rare-class recall at the argmax (mel about +15 lesions) at the cost of nv recall (about 56-80 more nv lesions misclassified) and accuracy. At a fixed 25% referral cap they show no significant gain, so the loss mostly moves the operating point.
- Caution: df and vasc are too small to interpret; no per-class paired bootstrap.
- Because: the argmax recall depends on where the decision boundary sits (the implicit threshold). A weighted loss pushes that boundary towards the rare classes, so more mel lesions are labelled mel (and more nv are wrongly labelled). Sensitivity at a fixed cap only uses the ranking of p_high, and the ranking (AUROC) did not change, so the same lesions are on top of the list. The loss moved the threshold, not the quality of the ranking. (Drafted with Claude's help.)

### Nov 22 Decision: hybrid CNN design (main design 2)

- Design: EfficientNet-B0 image branch (1280 features) + metadata branch (19 inputs to 32, ReLU, dropout) + 7-class head; 4,017,379 parameters, 16.4 MB.
- Metadata features: scaled age, one-hot sex and location, plus 3 missing-indicator flags; preprocessing fitted on train rows only (saved as meta_prep.joblib for the app).
- Because: a filled-in median age looks like a real age, so the indicator flags tell the model which details were missing.
- Rule written BEFORE p12: primary measure sensitivity at the 25% cap (3-seed mean, paired bootstrap); secondary macro F1 and AUROC; I say the hybrid or metadata helps only if the interval excludes 0; argmax measures are reported next to the referral rate.

### Nov 22 Result of p12: hybrid vs image-only (plain loss, 3-seed means)

- Sensitivity at 25%: 0.891 vs 0.891, difference 0.000 (-0.028 to 0.024). Macro F1 0.020 (-0.005 to 0.050). AUROC 0.005 (0.000 to 0.010).
- Argmax: mel recall +0.103 (0.045 to 0.160), referral sensitivity +0.071 (0.038 to 0.108), but referral rate +0.027 (0.018 to 0.036): part of the gain is a shifted operating point.
- Reading: no demonstrated gain at the primary measure.
- Because: the hybrid's argmax gains come with a higher referral rate (+0.027), so part of the gain is a shifted operating point, not better ranking. At a fixed 25% cap the two designs refer the same share of lesions, and the sensitivity is identical (0.891 vs 0.891), so the metadata did not make the ranking of high-concern lesions better in a way I can show. (Drafted with Claude's help.)

### Nov 23 Result of p13: missing-metadata test (whole lesions hidden, nested fixed masks, 3-seed means)

- Normal hybrid 0% to 100% hidden: macro F1 0.771 to 0.757, mel recall 0.651 to 0.582, sens@25% 0.891 to 0.877. Dropout hybrid (30% of training samples with all details hidden): 0.772 to 0.763, 0.655 to 0.636, 0.899 to 0.889. Image only: 0.751 / 0.548 / 0.891.
- Pre-registered claim rule: dropout improves robustness only if it beats the normal hybrid at 100% hidden on sens@25% AND macro F1 with paired 95% intervals excluding 0. Result: +0.009 (-0.019 to 0.035) and +0.007 (-0.022 to 0.036). Rule not met: robustness gain not demonstrated. With nothing hidden, dropout costs nothing visible (macro F1 +0.001, -0.028 to 0.032).
- Metadata effect inside one model (details present minus hidden): normal macro F1 +0.015 (0.004 to 0.027), mel recall +0.069 (0.033 to 0.104), sens@25% +0.011 (-0.003 to 0.027); dropout macro F1 +0.009 (0.002 to 0.020), mel recall +0.020 (0.007 to 0.036), sens@25% +0.008 (-0.002 to 0.019).
- Dropout hybrid with details hidden minus image only: mel recall +0.089 (0.039 to 0.141), sens@25% -0.002 (-0.027 to 0.022). A model without metadata cannot gain from metadata, so this gap is not a metadata effect (architecture, training noise, epoch selection).
- Caveats: 15 intervals so some may exclude 0 by chance; dropout hid all details together, single fields were not trained; small validation set; metadata alone reaches macro F1 about 0.19.
- Because: most of the information is in the image (metadata alone reaches macro F1 of only about 0.19), so with details hidden the hybrid falls back on its image branch, and the missing-indicator flags tell it the details are absent. Dropout cannot be called better because the paired intervals against the normal hybrid at 100% hidden include 0 (+0.009 and +0.007), so the pre-registered rule was not met; the point estimates may be noise. (Drafted with Claude's help.)

### Nov 23 Decision: models for the app tabs

- Tab 2 (image): plain EfficientNet-B0. Tab 3 (hybrid): dropout hybrid, chosen as a robustness design decision (equal performance with details present, point estimates degrade less when hidden), not as a proven improvement. Tab 1 (metadata): best metadata model.
- Comparison models in each tab are labelled "comparison only, not used for the decision".

### Nov 24 Result of p14: decision layer (validation, calibrated)

- Calibration (T / ECE before / ECE after cross-fitted): image seed 42 1.78 / 0.059 / 0.018; image seed 1 1.51 / 0.045 / 0.020; image seed 2 1.39 / 0.036 / 0.030; hybrid seed 42 1.01 / 0.015 / 0.017; hybrid seed 1 1.26 / 0.033 / 0.024; hybrid seed 2 1.60 / 0.041 / 0.025. Image-only models were overconfident (T above 1); the deployed hybrid was already calibrated and scaling did not help it. ECE with 10 bins on 1068 lesions is noisy (differences of about 0.01 are not meaningful). Temperature scaling does not change accuracy or the top class; it changes confidence only.
- Deferral at a 25% cap (image seed 1; hybrid seed 42): 0% deferred: sensitivity 0.899 and 0.917, error among kept 0.122 and 0.116; 10% deferred: 0.893 and 0.911, error 0.079 and 0.079, errors in deferred 54 and 48; 20% deferred: 0.851 and 0.792 (the cap becomes binding because deferred lesions count as referred).
- Risk-coverage: error among kept lesions falls from about 0.12 at full coverage to about 0.05 (image) and 0.04 (hybrid) at 80% coverage; the most confident 40% have almost no errors.
- Cost (r x missed + needless referrals): best referral rate rises with r (about 0.17-0.19 for r = 2, 0.23-0.24 for r = 5, 0.33-0.34 for r = 10, 0.37-0.41 for r = 20). A 25% cap is cost-optimal only for r of about 5.
- Because: the lowest-confidence lesions hold many of the errors (54 and 48 errors among the 10% deferred), and they are rarely the high-concern lesions that p_high already ranks near the top, so deferring them removes errors at almost no cost in sensitivity. Deferred lesions also count as referred, so they use up the cap. At 20% deferral they take so much capacity that fewer lesions can be referred by p_high, and sensitivity falls to 0.85 and 0.79. (Drafted with Claude's help.)

### Nov 24 Decision: referral cap, deferral share and target sensitivity (YOUR DECISION)

- Evidence: sensitivity at 20 / 25 / 30% caps is about 0.81 / 0.89 / 0.93 (image only, validation); deferral 10% costs at most one high-concern lesion; high-concern lesions are about 15.7% of lesions, so a 25% cap leaves about 9 points for false alarms and deferrals.
- Suggested values (confirm or change): cap = 0.25, deferral share = 0.10, target sensitivity about 0.90 at the cap.
- My choice (confirmed Nov 27): cap = 0.25, deferral share = 0.10, target sensitivity about 0.90 at the cap.
- Because (cap): I assume a clinic can look at about one lesion in four. On validation a 20% cap gives sensitivity of about 0.81, 25% about 0.89 and 30% about 0.93, and the referral curve flattens after about 30%. Going from 25% to 30% gains only 3-4 points for 5 more points of workload, while 20% loses about 8 points. High-concern lesions are about 15.7% of lesions, so 25% leaves about 9 points for false alarms and deferrals. The cost analysis says 25% is cost-optimal at r of about 5; for r = 10 or 20 the best rate would be 33-41%, so the cap is a capacity choice, not the cost-optimal rate for every r.
- Because (deferral share): at 10% deferral sensitivity at the cap barely changes (0.899 to 0.893 image only, 0.917 to 0.911 hybrid), error among kept lesions falls from about 0.12 to about 0.08, and 48-54 errors are in the deferred group. At 20% sensitivity falls to 0.85 and 0.79, so 10% is the largest share that is nearly free.
- Because (target): validation sensitivity at the cap is 0.89-0.91, so about 0.90 is realistic; the image-only model is just below it (0.893 at 10% deferral). The test number will probably be lower because the deployed seeds were picked by their best validation score. If the test misses 0.90 I report the measured value and explain it (R1). (Drafted with Claude's help.)
- If the target is missed: report the best result and explain why (R1).

### Nov 24 Decision: freeze rule (write it BEFORE the test run)

- Deployed models: image-only seed 1 and dropout hybrid seed 42, chosen as the seeds with the highest validation sensitivity at the 25% cap (ties by macro F1). I know this is the maximum of three noisy values, so validation numbers for these two are optimistic.
- Frozen before the test run: temperature T, deferral threshold tau and referral threshold t of every run, stored in decision_rule_FROZEN.json (committed with a message starting "FREEZE:").
- Test protocol: the test set is evaluated once, all 3 seeds of both designs, each with its own frozen rule; I report the mean, the spread and the deployed models' bootstrap intervals. Nothing is changed after seeing test results.
- Because: T, tau and t are fitted or chosen on validation data. If I changed them after seeing the test results, the test set would become a second tuning set and the test numbers would be optimistic and no longer an honest estimate for new lesions. The seed was also picked on validation, so it must be fixed before the test is touched. (Drafted with Claude's help.)

### Nov 25 Result: image-degradation stress test (validation, 3-seed means, details present)

- Clean: image only macro F1 0.751, sens@25% 0.891; dropout hybrid 0.772, 0.899.
- Blur radius 2: 0.360 / 0.744 (image), 0.377 / 0.702 (hybrid), mel recall about 0.02. Blur radius 4, JPEG quality 5 and noise sigma 20 push sens@25% to 0.31-0.53 (random referral gives 0.25). Darkening is tolerated best (0.83). No consistent hybrid advantage; no paired test run.
- Caveats: models trained without blur, noise or compression augmentation; degradations applied to the 224x224 images; sens@25% uses an adaptive threshold, a frozen threshold may behave worse (not measured).
- Design decision: add an image-quality gate to the app (sharpness and brightness) with thresholds chosen on validation images; low-quality images get a retake warning and a deferral. Blur/noise/compression augmentation is listed as future work.
- Because (likely explanation, not tested): melanoma is told apart from the very common nevus by fine dermoscopic structures (network, dots, streaks, irregular borders). Blur removes exactly that detail, so a blurred melanoma looks like a smooth nevus, and nv is about 67% of the training data. The ranking survives better (AUROC 0.86-0.88) than the argmax recall, which is why sensitivity at the cap falls less than mel recall. (Drafted with Claude's help.)

### Nov 27 Result: CPU timing (p15, draft rules)

- Hardware: Intel64 Family 6 Model 78 CPU (exact model name from Task Manager: **\_\_\_\_**), Windows 10, Python 3.13, torch 2.15 CPU with 2 threads, scikit-learn 1.6.1 (needed to load meta_prep.joblib; version 1.9.0 failed with a pickle error, so the environment file pins 1.6.1).
- Image only seed 1: 4,016,515 parameters, 16.4 MB; per lesion mean 81 ms, p95 98 ms, max 106 ms (read+resize 24, network 56, decision 0.11). Dropout hybrid seed 42: 4,017,379 parameters, 16.4 MB; mean 75 ms, p95 92 ms, max 95 ms (read 12, metadata 10, network 53). R5 met: p95 about 10 times under 1 s, file about 6 times under 100 MB.
- Caveats: model already loaded, one image at a time, no Grad-CAM or app overhead; the hybrid's faster read is probably the OS file cache; lesions with several images need several passes; the 60-image decision counts used the draft rules and are not results.

### Nov 27 Result: learning curves (p16, plain loss, 3 seeds)

- Validation loss is lowest at epoch 3-7 (0.41-0.45) while train loss keeps falling to about 0.11-0.13; at epoch 12 validation loss is 0.51-0.59. Both designs overfit after about epoch 4; neither underfits. The hybrid's minimum is slightly lower but the seed bands overlap.
- Checkpoints were chosen by validation macro F1, not loss, so they can come from a later epoch than the loss minimum. Rising loss with a stable ranking fits the overconfidence that temperature scaling fixed (image-only T of 1.51). Figure: figures/fig_learning_curves.png.

### Nov 25 Decision: instructor guidance, finish to the marks breakdown

- The instructor said to focus on finishing the CEP according to the marks breakdown (app, report, knowing the work) and to cater for novelty later.
- Decision: no new experiments. Task D is written up with the experiments already done (metadata, dropout, losses) as an honest, justified effort to improve, including negative results.
- Cut list: segmentation, preprocessing ablation, extra loss or dropout runs, single-field dropout training.

### Nov 8-25 Decision: literature rules

- Only peer-reviewed papers that use HAM10000 (instructor rule), every link opened and checked
- Excluded as core papers: Pacheco and Krohling (PAD-UFES-20, not HAM10000), Gessert et al. (ISIC 2019 challenge), review papers beyond 1-2, preprints without a published version (Adebiyi et al. is used only as a comparison example)
- Kept Guo et al. (2017) only as a method reference for temperature scaling, not as one of the 10 papers.
- Status: the teammate is building the 10-paper table; the stated-gap sentence is checked against that table before it goes into the report.

### Mistakes caught and fixed (for the report's "problems faced" section)

1. Scored images instead of lesions: fixed by averaging probabilities per lesion and evaluating and bootstrapping per lesion.
2. Median-filled age hides missingness: fixed with missing-indicator features.
3. Learning curves compared a weighted or focal train loss with a plain validation loss: fixed by using the plain-loss runs for the report figures.
4. Picked a winner from one noisy run (seed 42 and focal): fixed with 3 seeds and paired bootstrap.
5. A fallback in my own pre-registered rule defaulted to the intervention: changed to "keep plain" and logged openly.
6. Candidate papers included reviews, preprints, non-HAM10000 studies and a page with retraction wording: fixed with five checks and a link-tested reference log.
7. A zip with the wrong image folders would break the matching check: fixed by zipping only the two HAM10000 image folders and keeping the assertion.

### Open questions / still to do

- Test-set results (one run) and whether validation optimism shows up.
- Behaviour of the frozen thresholds under degraded images (referral rate and sensitivity), if time allows.
- Image-quality thresholds for the app gate (chosen on validation images).
- Grad-CAM cases (3 correct, 3 wrong), 2 common confusions, shortcut check.
- 3 "one mistake spoils everything" experiments and 3 trade-offs.
- Whether references and the code appendix count toward the 6-page limit (ask the instructor).
