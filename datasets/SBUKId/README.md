# e2eET Skeleton Based HGR Using Data-Level Fusion

## Data-Level Fusion: Processing Benchmark Datasets

### **SBU Kinect Interaction Dataset (SBUKID)**
   - Download the clean versions of the [SBUKID dataset](https://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/README.txt) and extract the zipped files to the directory `./datasets/SBUKId/`.
   - Preprocess the dataset by running the notebook `./modules/parse-data-SBUKId.ipynb`. This will create a new folder `./datasets/SBUKId.txts/` and five files `./datasets/SBUKId_3D_dictCVS_f0{1,2,3,4,5}_s282.pckl`.
   - To generate the five cross-validation spatiotemporal datasets:
      - Modify lines 43 and 44 in `./modules/.configs/sbukid-v5-default.hgr-config` to specify the cross-validation fold i.e. `f01` and `F01` for the first cross-validation fold, `f02` and `F02` for the second cross-validation fold, and so on.
      - Execute `python modules/create_imgs_v5_SBUKId_mVOs.py"` five times with the above modification for the cross-validation fold..
   - There should be five cross-validation spatiotemporal datasets saved to `./images_d/SBUKId-3D-CVS.F0{1,2,3,4,5}.8G-norm.960px-[allVOs.adaptiveMean]`.
   - To verify that the cross-validation spatiotemporal datasets have been generated correctly, you can run `./modules/verify-images-SBUKId.ipynb`.

*NOTE: The parameters required to generate the spatiotemporal datasets are set in the `*.hgr-config` files. See `./modules/.configs/all-HGR-ds-schemas.json` for details about the parameters.*

> **Alternatively, the preprocessed .pckl files and generated spatiotemporal datasets can be downloaded from this [drive folder](https://drive.google.com/drive/u/0/folders/1BvoxkRDBK86A3_oNdQrnC8TLvp4l0W9x) and extracted to the corresponding `./datasets` and `./images_d` directories.**

<hr>
