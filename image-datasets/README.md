# e2eET Skeleton Based HGR Using Data-Level Fusion

## Data-Level Fusion: Processing Benchmark Datasets

1. **Consiglio Nazionale delle Ricerche (CNR) Hand Gestures Dataset**
   - Download the [CNR dataset](https://github.com/aviogit/dynamic-hand-gesture-classification-datasets/tree/master/dynamic-hand-gestures-new-CNR-dataset-2k-images) and extract to the directory `./datasets/CNR/`.
   - Generate the spatiotemporal dataset by running the notebook `./data-pipeline/parseData-CNRd.ipynb`. This will create randomized training and validation subsets from the original dataset. The spatiotemporal dataset will be saved to `./image-datasets/CNR-3d-original-1920px.1080px-[topdown]/`.

2. **Leap Motion Dynamic Hand Gesture (LMDHG) Database**
   - Download the [LMDHG dataset](https://www-intuidoc.irisa.fr/en/english-leap-motion-dynamic-hand-gesture-lmdhg-database/) and extract to the directory `./datasets/LMDHG/`.
   - Preprocess the dataset by running the notebook `./data-pipeline/parseData-LMDHGd.ipynb`. This will create a file `./datasets/LMDHG_3d_dictPaperSplit_l750_s609.pckl`.
   - Generate the spatiotemporal dataset using `python data-pipeline/createImages_LMDHGd_mVOs.py -c "data-pipeline/configs/LMDHG.default.json"`. The spatiotemporal dataset will be saved to `./image-datasets/LMDHG.mVOs-dictPaperSplit-3d.V1-noisy(raw).960px-[allVOs].adaptive-mean`.

3. **First-Person Hand Action (FPHA) Benchmark**
   - Download the [FPHA dataset](https://guiggh.github.io/publications/first-person-hands/) and extract to the directory `./datasets/FPHA/`.
   - Preprocess the dataset by running the notebook `./data-pipeline/parseData-FPHAd.ipynb`. This will create a file `./datasets/FPHA_3d_dictPaperSplit_l250_s1175.pckl`.
   - Generate the spatiotemporal dataset using `python data-pipeline/createImages_FPHAd_mVOs.py -c "data-pipeline/configs/FPHA.default.json"`. The spatiotemporal dataset will be saved to `./image-datasets/FPHA.mVOs-dictPaperSplit-3d.V1-noisy(raw).960px-[allVOs].adaptive-mean`.

4. **3D Hand Gesture Recognition Using a Depth and Skeleton Dataset (SHREC2017)**
   - Download the [SHREC2017 dataset](http://www-rech.telecom-lille.fr/shrec2017-hand/) and extract to the directory `./datasets/SHREC2017/`.
   - Preprocess the dataset by running the notebook `./data-pipeline/parseData-SHREC2017d.ipynb`. This will create a file `./datasets/SHREC2017_3d_dictTVS_l250_s2800.pckl`.
   - To generate the 14G and 28G spatiotemporal datasets:
      - Modify line 23 in `./data-pipeline/configs/SHREC2017.default.json` such that `"n_dataset_classes": 14,` for 14G evaluation mode or `"n_dataset_classes": 28,` for 28G evaluation mode.
      - Execute `uv run python data-pipeline/createImages_SHREC2017d_mVOs.py -c "data-pipeline/configs/SHREC2017.default.json"`.
   -  The 14G and 28G spatiotemporal datasets will be saved to `./image-datasets/SHREC2017.mVOs-3d.14g-noisy(raw).960px-[allVOs].adaptive-mean` and `./image-datasets/SHREC2017.mVOs-3d.28g-noisy(raw).960px-[allVOs].adaptive-mean` respectively.

5. **Dynamic Hand Gesture 14/28 (DHG1428) Dataset**
   - Download the [DHG1428 dataset](http://www-rech.telecom-lille.fr/DHGdataset/) and extract to the directory `./datasets/DHG1428/`.
   - Preprocess the dataset by running the notebook `./data-pipeline/parseData-DHG1428d.ipynb`. This will create a file `./datasets/DHG1428_3d_dictTVS_l250_s2800.pckl`.
   - To generate the 14G and 28G spatiotemporal datasets:
      - Modify line 23 in `./data-pipeline/configs/DHG1428.default.json` such that `"n_dataset_classes": 14,` for 14G evaluation mode or `"n_dataset_classes": 28,` for 28G evaluation mode.
      - Execute `uv run python data-pipeline/createImages_DHG1428d_mVOs.py -c "data-pipeline/configs/DHG1428.default.json"`.
   -  The 14G and 28G spatiotemporal datasets will be saved to `./image-datasets/DHG1428.mVOs-3d.14g-noisy(raw).960px-[allVOs].adaptive-mean` and `./image-datasets/DHG1428.mVOs-3d.28g-noisy(raw).960px-[allVOs].adaptive-mean` respectively.

6. **SBU Kinect Interaction Dataset (SBUKID)**
   - Download the clean version of the [SBUKID dataset](https://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/README.txt) and extract the zipped files to the directory `./datasets/SBUKId/`.
   - Preprocess the dataset by running the notebook `./data-pipeline/parseData-SBUKId.ipynb`. This will create a new folder `./datasets/SBUKId.txts/` and five files `./datasets/SBUKId_3D_dictCVS_f0{1,2,3,4,5}_s282.pckl`.
   - To generate the five cross-validation spatiotemporal datasets:
      - Modify lines 43 and 44 in `./data-pipeline/configs/SBUKId.default.json` to specify the cross-validation fold i.e. `f01` and `F01` for the first cross-validation fold, `f02` and `F02` for the second cross-validation fold, and so on.
      - Execute `uv run python data-pipeline/createImages_SBUKId_mVOs.py` five times, updating the configuration for each cross-validation fold.
   - There should be five cross-validation spatiotemporal datasets saved to `./image-datasets/SBUKId-3D-CVS.F0{1,2,3,4,5}.8G-norm.960px-[allVOs.adaptiveMean]`.
   - To verify that the cross-validation spatiotemporal datasets have been generated correctly, you can run `./data-pipeline/verifyImages-SBUKId.ipynb`.

*NOTE: The parameters required to generate the spatiotemporal datasets are set in the `*.default.json` files. See `./data-pipeline/configs/defaultConfigs.schema.json` for details about the parameters.*

> **Alternatively, the preprocessed .pckl files and generated spatiotemporal datasets can be downloaded from this [drive folder](https://drive.google.com/drive/u/0/folders/1BvoxkRDBK86A3_oNdQrnC8TLvp4l0W9x) and extracted to the corresponding `./datasets` and `./image-datasets` directories.**

<hr>
