### Nov 8 Decision: split by lesion_id, not by image
- Alternatives: random split by image
- Why: 10,015 images but only 7,470 lesions, some lesions have 2-6 photos
- Because: ______ (what goes wrong if photos of one lesion are in train and test?)

### Nov 8 Decision: fit imputer and scaler on train only
- Because: ______

### Nov 9 Decision: report macro F1 and referral sensitivity, not only accuracy
- Evidence: always-predict-nv gets 0.669 accuracy but 0 sensitivity
- Because: ______

### Nov 19 Decision: evaluate per lesion (average probabilities of its images)
- Why: the brief defines referral rate as lesions / all lesions, my first scripts counted images
- Because: ______

### Nov 2x Decision: fine-tune EfficientNet-B0 (4.0M parameters, 16.4 MB file)
- Alternatives: frozen ResNet18 + logistic regression (my baseline)
- Why: ______