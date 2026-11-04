# Reproduction and Usage Guide

This guide contains the dataset preparation, training, evaluation and gesture-recognition application instructions for the project. For an overview, see the [main README](../README.md).

Unless noted otherwise, paths below are relative to the repository root. Run dataset-preparation commands from the repository root and training commands from the experiments directory.

## Data-Level Fusion: Processing Benchmark Datasets

1. **Consiglio Nazionale delle Ricerche (CNR) Hand Gestures Dataset**
   - Download the [CNR dataset](https://github.com/aviogit/dynamic-hand-gesture-classification-datasets/tree/master/dynamic-hand-gestures-new-CNR-dataset-2k-images) and extract it to `datasets/CNR/`.
   - Generate the spatiotemporal dataset using the [CNR preprocessing notebook](../data-pipeline/parseData-CNRd.ipynb). This creates randomized training and validation subsets. The generated dataset is saved to `image-datasets/CNR-3d-original-1920px.1080px-[topdown]/`.

2. **Leap Motion Dynamic Hand Gesture (LMDHG) Database**
   - Download the [LMDHG dataset](https://www-intuidoc.irisa.fr/en/english-leap-motion-dynamic-hand-gesture-lmdhg-database/) and extract it to `datasets/LMDHG/`.
   - Preprocess it using the [LMDHG preprocessing notebook](../data-pipeline/parseData-LMDHGd.ipynb). This creates `datasets/LMDHG_3d_dictPaperSplit_l750_s609.pckl`.
   - Generate the spatiotemporal dataset using the following command:

     ```bash
     uv run python data-pipeline/createImages_LMDHGd_mVOs.py -c "data-pipeline/configs/LMDHG.default.json"
     ```

     The generated dataset is saved to `image-datasets/LMDHG.mVOs-dictPaperSplit-3d.V1-noisy(raw).960px-[allVOs].adaptive-mean`.

3. **First-Person Hand Action (FPHA) Benchmark**
   - Download the [FPHA dataset](https://guiggh.github.io/publications/first-person-hands/) and extract it to `datasets/FPHA/`.
   - Preprocess it using the [FPHA preprocessing notebook](../data-pipeline/parseData-FPHAd.ipynb). This creates `datasets/FPHA_3d_dictPaperSplit_l250_s1175.pckl`.
   - Generate the spatiotemporal dataset using the following command:

     ```bash
     uv run python data-pipeline/createImages_FPHAd_mVOs.py -c "data-pipeline/configs/FPHA.default.json"
     ```

     The generated dataset is saved to `image-datasets/FPHA.mVOs-dictPaperSplit-3d.V1-noisy(raw).960px-[allVOs].adaptive-mean`.

4. **3D Hand Gesture Recognition Using a Depth and Skeleton Dataset (SHREC2017)**
   - Download the [SHREC2017 dataset](http://www-rech.telecom-lille.fr/shrec2017-hand/) and extract it to `datasets/SHREC2017/`.
   - Preprocess it using the [SHREC2017 preprocessing notebook](../data-pipeline/parseData-SHREC2017d.ipynb). This creates `datasets/SHREC2017_3d_dictTVS_l250_s2800.pckl`.
   - To generate the 14G and 28G spatiotemporal datasets:
     - Set `"n_dataset_classes": 14,` in the [SHREC2017 configuration](../data-pipeline/configs/SHREC2017.default.json) for 14G evaluation or `"n_dataset_classes": 28,` for 28G evaluation.
     - Generate the images using:

       ```bash
       uv run python data-pipeline/createImages_SHREC2017d_mVOs.py -c "data-pipeline/configs/SHREC2017.default.json"
       ```

   - The generated 14G and 28G datasets are saved to `image-datasets/SHREC2017.mVOs-3d.14g-noisy(raw).960px-[allVOs].adaptive-mean` and `image-datasets/SHREC2017.mVOs-3d.28g-noisy(raw).960px-[allVOs].adaptive-mean` respectively.

5. **Dynamic Hand Gesture 14/28 (DHG1428) Dataset**
   - Download the [DHG1428 dataset](http://www-rech.telecom-lille.fr/DHGdataset/) and extract it to `datasets/DHG1428/`.
   - Preprocess it using the [DHG1428 preprocessing notebook](../data-pipeline/parseData-DHG1428d.ipynb). This creates `datasets/DHG1428_3d_dictTVS_l250_s2800.pckl`.
   - To generate the 14G and 28G spatiotemporal datasets:
     - Set `"n_dataset_classes": 14,` in the [DHG1428 configuration](../data-pipeline/configs/DHG1428.default.json) for 14G evaluation or `"n_dataset_classes": 28,` for 28G evaluation.
     - Generate the images using:

       ```bash
       uv run python data-pipeline/createImages_DHG1428d_mVOs.py -c "data-pipeline/configs/DHG1428.default.json"
       ```

   - The generated 14G and 28G datasets are saved to `image-datasets/DHG1428.mVOs-3d.14g-noisy(raw).960px-[allVOs].adaptive-mean` and `image-datasets/DHG1428.mVOs-3d.28g-noisy(raw).960px-[allVOs].adaptive-mean` respectively.

6. **SBU Kinect Interaction Dataset (SBUKID)**
   - Download the clean version of the [SBUKID dataset](https://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/README.txt) and extract the zipped files to `datasets/SBUKId/`.
   - Preprocess it using the [SBUKID preprocessing notebook](../data-pipeline/parseData-SBUKId.ipynb). This creates a `datasets/SBUKId.txts/` folder and five files named `SBUKId_3D_dictCVS_f0{1,2,3,4,5}_s282.pckl`.
   - To generate the five cross-validation spatiotemporal datasets:
     - Set the cross-validation fold in lines 43 and 44 of the [SBUKID configuration](../data-pipeline/configs/SBUKId.default.json), using `f01` and `F01` for the first fold, `f02` and `F02` for the second fold, and so on.
     - Run the following command five times, updating the configuration for each fold:

       ```bash
       uv run python data-pipeline/createImages_SBUKId_mVOs.py
       ```

   - This creates five cross-validation datasets under `image-datasets/SBUKId-3D-CVS.F0{1,2,3,4,5}.8G-norm.960px-[allVOs.adaptiveMean]`.
   - Verify the generated datasets using the [SBUKID image verification notebook](../data-pipeline/verifyImages-SBUKId.ipynb).

*Note: Dataset-generation parameters are set in the default JSON configuration files. See the [configuration schema](../data-pipeline/configs/defaultConfigs.schema.json) for details.*

> Alternatively, download the preprocessed .pckl files and generated spatiotemporal datasets from this [drive folder](https://drive.google.com/drive/folders/1BvoxkRDBK86A3_oNdQrnC8TLvp4l0W9x) and extract them to the corresponding `datasets` and `image-datasets` directories.

## Model Training and Evaluation

The [experiments directory](../experiments) contains the training and evaluation code for all benchmark datasets. The [runs directory](../runs) contains TensorBoard event archives linked to their corresponding result logs; see its [archive guide](../runs/README.md) for details.

Set training arguments at the command line. See the [training-parameter schema](../experiments/src/trainingParameters.schema.json) for details. Run training commands from the `experiments` directory.

Before training, ensure that the `datasetDirectories` object in [`helperFunctions.py`](../experiments/src/helperFunctions.py) points to the generated datasets in `image-datasets`.

### Single-Stream and Two-Stream Experiments

[`sVO.mVO.Checkpoints.py`](../experiments/sVO.mVO.Checkpoints.py) tests the effect of different view orientations on classification accuracy across the datasets.

The [notebook](../experiments/sVO.mVO.Checkpoints.ipynb) contains the command-line inputs. The [results log](../results/sVO.mVO.Checkpoints.log) records the 1SA and 2SA experiments. New run logs are written under `experiments/outputs/`; they do not update the checked-in results log. The corresponding TensorBoard event archive is `runs/sVO.mVO.Checkpoints.zip`.

The SBUKId single-stream checkpoint runs cover six view orientations across five cross-validation folds. Their TensorBoard events are in `runs/SBUKId.sVO.Checkpoints.zip`, with the corresponding accuracies in the [SBUKId results log](../results/SBUKId.sVO.Checkpoints.log).

**1SA example:** Train on DHG1428 with 14 classes and the `top-down` view on GPU 0:

```bash
uv run python sVO.mVO.Checkpoints.py -IG 0 -nC 14 -mVOs top-down -dsN DHG1428
```

**2SA example:** Train on DHG1428 with 28 classes and the ordered `custom` and `side-left` views on GPU 1:

```bash
uv run python sVO.mVO.Checkpoints.py -IG 1 -nC 28 -mVOs custom side-left -dsN DHG1428
```

### Multi-Stream Ensemble Tuner Experiments

[`mVO.e2eEnsembleTuning.py`](../experiments/mVO.e2eEnsembleTuning.py) trains the custom Ensemble Tuner Multi-Stream CNN architecture shown in the main README.

The [notebook](../experiments/mVO.e2eEnsembleTuning.ipynb) contains the command-line inputs. The [results log](../results/mVO.e2eEnsembleTuning.log) records the 1SA, 2SA and 3SA experiments used to find the optimal combination and ordering of views for all datasets. New run logs are written under `experiments/outputs/`; they do not update the checked-in results log. The corresponding TensorBoard event archive is `runs/mVO.e2eEnsembleTuning.zip`.

**3SA example:** Train on DHG1428 with 14 classes and the ordered `top-down`, `custom` and `front-away` views on GPU 0:

```bash
uv run python mVO.e2eEnsembleTuning.py -IG 0 -nC 14 -mVOs top-down custom front-away -dsN DHG1428 -IISE 0 -ISIS 0
```

### Final Evaluation on Benchmark Datasets

Final evaluation used the optimal view combinations and training schedules. The summary spreadsheet `results/allDatasets-e2eEnsembleTuning-Summary.xlsx` and its companion [summary log](../results/allDatasets-e2eEnsembleTuning-Summary.log) describe how the final classification accuracies were obtained.

To reproduce the evaluation results reported in the paper, use the command-line inputs in the [evaluation notebook](../experiments/allDatasets-e2eEnsembleTuning-Evaluation.ipynb). The [evaluation log](../results/allDatasets-e2eEnsembleTuning-Evaluation.log) contains the results. The corresponding TensorBoard event archive is `runs/allDatasets-e2eEnsembleTuning-Evaluation.zip`.

## Live Real-Time HGR Application

The real-time application recognises the `Swipe- { Up | Down | Right | Left | + | V | X }` gestures from the DHG1428 dataset in both 14G and 28G modes. Gestures can be performed with either one finger or the whole hand.

The [DHG1428 gesture diagram](./DHG1428-Swipe-Gestures.png) shows how to perform the gestures, as in the original dataset. The [application video](./HGR-Live-Demo-Video.mp4) shows the application recognising all the `Swipe` gestures in one take.

The real-time application requires trained `.pkl` models. Download them from this [drive folder](https://drive.google.com/drive/folders/1BvoxkRDBK86A3_oNdQrnC8TLvp4l0W9x) and extract them to the `gesture-recognition/models` directory.

The application is organised as follows:

```text
gesture-recognition/
├── allConfigs.jsonc
├── liveStreamHGR.py
├── models/          # trained .pkl models, ignored by Git
├── outputs/         # generated sequences, logs and archives, ignored by Git
└── src/             # application modules
```

To launch the application, change to the `gesture-recognition` directory and run:

```bash
uv run python liveStreamHGR.py
```

The application runs on CPU when `cpu_mode` is `true` in the [application configuration](../gesture-recognition/allConfigs.jsonc) and uses the GPU when it is `false`. If CUDA is unavailable, it falls back to CPU automatically.

When the application modules have initialised, the terminal output should include:

```text
INFO: Initialized <vispyOutputGUI.py> ...
INFO: Initialized <dataLevelFusion.py> ...
INFO: Initialized <liveStreamHGR.py> ...
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
INFO: dls.vocab=[0.Swipe+ 1.SwipeDown 2.SwipeLeft 3.SwipeRight 4.SwipeUp 5.SwipeV 6.SwipeX]
INFO: Initialized <gestureClassInference.py> ...
```

With the three windows shown in the [application window image](./Initialized-Application-Windows.png), focus the camera window for these controls:

1. Raise your hand in view of the camera and wait until a skeleton is overlaid on it. This shows that the gesture sequence is being recorded.
2. Perform any of the `Swipe` gestures while maintaining the skeleton overlay, as shown in the [gesture diagram](./DHG1428-Swipe-Gestures.png).
3. Press `Space` to save a sufficiently long gesture sequence and trigger data-level fusion and classification.
4. Read the gesture inference in the output GUI or command line. Inferences are appended to `gesture-recognition/outputs/history.log`.
5. Press lowercase `c` to clear the current gesture sequence without classifying it.
6. Press `Esc` to exit the camera stream and terminate the application and its child processes.

When `debug_mode` is enabled in the [application configuration](../gesture-recognition/allConfigs.jsonc), focus the relevant window before using its controls:

- **Fusion window**
  - `Left` and `Right` cycle through view orientations.
  - `Space` pauses or resumes the visualisation.
  - `C` prints camera details.
  - `Esc` closes the fusion window.
- **Output window**
  - `C` prints scene details.
  - `Esc` closes the output window.

These window-specific `Esc` controls do not exit the camera stream; use `Esc` in the focused camera window to stop the whole application.

### Gesture-Recognition Dataset and Model Training

The optional seven-class dataset used by the application is generated from the 14G DHG1428 spatiotemporal dataset. Extract `DHG1428.mVOs-3d.14g-noisy(raw).960px-[allVOs].adaptive-mean.rar` into `image-datasets`, then run:

```bash
cd gesture-recognition
uv run python -m src.createDemoDataset
uv run python -m src.ensembleTunerTraining -IG 0 --n_classes 7 -IE 30 -IISE 0 -ISE 10 -ISIS 276 328 -CEMC -CETE
```

`createDemoDataset` writes `image-datasets/DHG1428.liveStream-3D.7G-[custom.topdown.frontaway]`. Training writes the model and checkpoints to `gesture-recognition/models` and appends run logs to `gesture-recognition/outputs/ensembleTunerTraining.log`. The [training notebook](../gesture-recognition/ensembleTunerTraining.ipynb) contains the same commands.
