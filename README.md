# Corn-AI Leaf Disease Detection

Corn-AI uses two MobileNetV2 classifiers in sequence: a corn/not-corn verifier, then a corn disease classifier. It also reports image-quality measurements and abstains when a score is below a decision cutoff. Scores are model outputs, not guarantees of correctness. Quality warnings do not themselves reject an image.

## Current project data and models

The existing project contains a 2-class corn detector, a 3-class disease model, and these disease folders:

- `dataset/healthy/`
- `dataset/rust disease/`
- `dataset/gray leaf disease/`

The required Northern leaf blight class is not present in the current disease training data or test folders. The existing `models/disease_model.keras` has three outputs and cannot predict Northern leaf blight. In addition, the current `test data/` contains images with matching source names and duplicates from `dataset/`, so it is not an independent test set. Evaluation detects this overlap and refuses to report it as independent. No four-class model or valid independent-test result is claimed. The training script below checks for all four classes and stops with a clear error while any are missing.

Training writes candidate models rather than replacing the existing active models. This leaves the current working models and `classes.txt` intact while the new data and results are reviewed.

## Install in VS Code on Windows

Use Python 3.13 for this project. Python 3.14 currently cannot install the stable TensorFlow version pinned in `requirements.txt`. You do not need to activate the environment; invoking its Python executable directly also avoids PowerShell execution-policy errors.

From the project root in PowerShell, create the environment once (skip this if `.venv` already exists):

```powershell
py -3.13 -m venv .venv
```

Install the training and runtime dependencies into that environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
```

Run the dashboard using the same environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If you prefer activating the environment and PowerShell blocks scripts, bypass the policy for this terminal session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\.venv\Scripts\Activate.ps1
```

Do not use bare `pip` or `python` commands after activation fails: they may refer to a different system Python.

Native Windows TensorFlow is CPU-only in recent TensorFlow releases. For NVIDIA GPU training, use a supported Linux/WSL2 TensorFlow setup.

## Dataset folders

Keep disease training data under `dataset/`, with exactly one recognized folder for each class. Existing folder aliases are accepted:

```text
dataset/
  healthy/
  common rust/              # or rust disease/
  gray leaf spot/            # or gray leaf disease/
  northern leaf blight/      # or northern corn leaf blight/
```

Use JPG, JPEG, PNG, or WEBP images. Include representative Indian maize varieties and field conditions, not just clean laboratory leaves. Do not place augmentation copies from one original across different disease labels.

The detector data remains:

```text
corn_detector/
  corn/
  not_corn/
```

Use diverse corn leaves in `corn/`. Include other crop leaves (including tomato, chilli, and rice), soil, people, and unrelated objects in `not_corn/`. A finite negative set cannot represent every possible non-corn image, so rejection is only as reliable as the diversity of training and independent test examples.

For independent disease evaluation, `test data/` must contain one folder per disease class, with folder names matching the class aliases above. It must be collected independently and must not be used to choose the model, training settings, or thresholds. The disease trainer and evaluator check for matching names and perceptual near-duplicates across train/test folders but cannot verify capture provenance. Remove the current duplicated test examples and replace them with genuinely held-out images before using test metrics. To evaluate a detector on an external test set, create a separately collected folder with `corn/` and `not_corn/` subfolders and pass it to the evaluator. Never copy training images into these test folders.

The split groups identical files, common augmentation/copy filename suffixes, matching normalized stems, and perceptual-hash near-duplicates before stratified train/validation/test partitioning. This substantially reduces leakage, but image hashes cannot prove that differently edited or renamed photos have independent original sources; retain source/plant identifiers and use them to group data whenever available.

## Train candidate models

Train the binary verifier:

```powershell
& .\.venv\Scripts\Activate.ps1
python train_detector.py
```

Train the four-class disease classifier:

```powershell
python train.py
```

Each script uses deterministic seeds, source-grouped stratified splits, class weights, training-only augmentation, a frozen MobileNetV2 stage, cautious fine-tuning, learning-rate reduction, early stopping, and best-validation-loss checkpoints. The candidate detector combines a validation-selected corn score cutoff with a nearest-reference feature-distance guard: normalized MobileNetV2 corn embeddings from training are compared against each image, and the distance limit is calibrated on validation corn examples. Both guards are evaluated together for false acceptance/rejection. The disease cutoff is selected from validation data to meet a configurable minimum precision while retaining coverage. These are rejection cutoffs, not probability calibration or a guarantee for every unseen image.

Training writes candidate artifacts under `models/` and plots under `outputs/`. It refuses to overwrite existing candidate artifacts unless explicitly told to do so:

```powershell
python train_detector.py --overwrite-candidate
python train.py --overwrite-candidate
```

The current disease data is missing Northern leaf blight, so `python train.py` currently reports the missing class and exits before training or replacing any model.

## Independent evaluation

Evaluate active models on the disease images in `test data/`. Active detector metrics are not reported unless a separate detector test folder is provided, because the existing detector's training partition history is not guaranteed:

```powershell
python evaluate.py --detector-test-dir "C:\path\to\independent-detector-test"
```

The detector test directory must contain `corn/` and `not_corn/`. Evaluate new candidate models using the source-grouped holdout split reserved by their training script, or supply the separate detector test directory:

```powershell
python evaluate.py --candidate --detector-test-dir "C:\path\to\independent-detector-test"
```

The active disease model evaluation currently stops because matching train/test sources are present. The candidate disease model also cannot be evaluated until `test data/` has all four matching classes and the train/test overlap is removed. Once valid data is provided, reports include accuracy, macro F1, per-class precision/recall/F1, confusion matrices, disease confidence coverage, detector non-corn false acceptance and corn false rejection, and accuracy grouped by measured brightness. Outputs are saved to `outputs/`. Evaluation never tunes thresholds.

Do not repeatedly use the independent test results to select thresholds or model variants. Select settings using validation data and reserve the independent test for final reporting.

## Predict a local image

```powershell
python predict.py "C:\path\to\corn-leaf.jpg"
```

Use candidate models explicitly after both candidate models have been trained:

```powershell
python predict.py "C:\path\to\corn-leaf.jpg" --candidate
```

The output includes the corn decision, up to three disease candidates, their scores, image dimensions, brightness, contrast, Laplacian-variance blur score, and quality warnings. Candidate detectors also report the feature-distance guard. Low brightness, exposure, or blur is not automatically treated as proof that the image is not corn.

## Deploy real inference on Streamlit Community Cloud

The Streamlit app runs the active TensorFlow models on the server. Streamlit Community Cloud deploys from GitHub and installs the runtime dependencies in `requirements.txt`; training and plotting dependencies are kept in `requirements-training.txt`.

1. Make sure the repository is pushed to GitHub and that you can connect your GitHub account to [share.streamlit.io](https://share.streamlit.io/). For a public app, use a public GitHub repository. The current `origin` remote is the project's GitHub repository.
2. In PowerShell, from the project folder, stage the application, its dependencies and helpers, the documented scripts, and both active model weights. Do not stage datasets, test images, candidate models, or `.venv`:

   ```powershell
   git add README.md .gitignore .gitattributes app.py evaluate.py predict.py train.py train_detector.py modules/vision.py modules/dataset_utils.py classes.txt models/corn_classes.txt requirements.txt requirements-training.txt
   git add -f models/corn_detector.keras models/disease_model.keras
   git status --short
   git diff --cached --name-only
   git diff --cached --stat
   ```

   Confirm the staged list does **not** include `dataset/`, `test data/`, `corn_detector/`, `.venv/`, `index.html`, candidate model files, or secrets. The two active model files are each about 10 MB. A public GitHub repository makes both model files downloadable.

3. Commit and push the reviewed files to the branch you want Streamlit to deploy:

   ```powershell
   git commit -m "Prepare Corn-AI Streamlit deployment"
   git push origin main
   ```

   If your current branch is not `main`, push that branch name instead. Do not use `git add -A` here; review `git status` so datasets and unrelated files are not published.

4. Open [Streamlit Community Cloud](https://share.streamlit.io/), sign in with GitHub, and choose **Create app**.
5. Select the GitHub repository, branch you pushed, and main file path `app.py`. Open **Advanced settings** and choose Python **3.11** if that option is shown; then deploy.
6. Wait for package installation and app startup. The first install and model load can take several minutes. If deployment fails, inspect **Manage app → Logs**. Missing model errors usually mean the `.keras` files were not committed; dependency errors will also appear in the build logs.

The model weights live in the deployed repository. Anyone with access to a public repository can download them. Image uploads are sent to the hosted app for inference; do not upload sensitive images. The app code does not save uploads to disk.

The existing disease model supports only healthy, common rust, and gray leaf spot; it cannot predict Northern leaf blight. The active models also use legacy fallback thresholds rather than newly validation-calibrated cutoffs. Keep the on-screen limitations visible and do not present this demo as a validated four-class or field-ready diagnostic system. Streamlit's free service has resource limits and may restart or sleep; check its current plan and usage limits before relying on it.

## Streamlit dashboard

Run the application from the project root:

```powershell
streamlit run app.py
```

Upload a JPG/JPEG/PNG/WEBP image in the browser. Select active or candidate models, review quality measurements and warnings, and inspect the real model result. Candidate models are explicitly labeled as not promoted. A model classifies the image as a whole; it does not locate lesions, identify lesion boundaries, or estimate disease severity. Grad-CAM can be added later for qualitative model attention, but it is not lesion segmentation.

## Common issues

- **Missing Northern leaf blight class:** add verified labeled images to the corresponding `dataset/` and independent `test data/` folders. Do not create empty folders or rename another disease to make the checks pass.
- **Not enough distinct sources:** collect more distinct plant/image sources per class; augmentation copies do not count as independent sources.
- **Model/class mapping mismatch:** keep the saved candidate class text and model together. Do not edit the class order by hand.
- **No model file:** confirm the `.keras` files exist in `models/` and run the appropriate training script.
- **Poor field image:** retake with the leaf in focus and without extreme shadow or glare when practical. A warning alone does not invalidate the classifier output.
- **A confident false acceptance:** add representative negative examples, retrain, and measure the false-acceptance rate on a genuinely independent non-corn set. Softmax scores and a finite negative set cannot guarantee open-world rejection.
