# e2eET Skeleton Based HGR Using Data-Level Fusion

## Data-Level Fusion: Processing Benchmark Datasets

### **Consiglio Nazionale delle Ricerche (CNR) Hand Gestures Dataset**
   - Download the [CNR dataset](https://github.com/aviogit/dynamic-hand-gesture-classification-datasets/tree/master/dynamic-hand-gestures-new-CNR-dataset-2k-images) [[alternate link](https://imaticloud.ge.imati.cnr.it/index.php/s/YNRymAvZkndzpU1/download?path=%2F&files=)] and extract to the directory `./images_d/CNR-3d-original-1920px.1080px-[topdown]/`.
   - Generate the spatiotemporal dataset by running the notebook `./modules/parse-data-CNRd.ipynb`. This will create training and validation subsets from the original dataset.

> **Alternatively, the generated spatiotemporal dataset can be downloaded from this [drive folder](https://drive.google.com/drive/folders/1LSzM9pTo6FHxqxH8Bt_YTf4Ky2lSf-gQ?usp=sharing) and extracted to the corresponding `./images_d` directory.**

<hr>
