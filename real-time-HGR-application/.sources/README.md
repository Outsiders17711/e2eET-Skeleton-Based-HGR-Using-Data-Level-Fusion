# e2eET Skeleton Based HGR Using Data-Level Fusion

## Live Real-Time HGR Application

> **The trained (.pkl) models required for the real-time application can be downloaded from this [drive folder](https://drive.google.com/drive/u/0/folders/1BvoxkRDBK86A3_oNdQrnC8TLvp4l0W9x) and extracted to the `./real-time-HGR-application/.sources` directory.**

The trained model required for the real-time application is OS-specific. You can choose to download only one of `[bf75]-7G-[cm_td_fa]-Windows.pkl` or `[bf75]-7G-[cm_td_fa]-Linux.pkl` from the drive folder for your OS. The module `./real-time-HGR-application/gestureClassInference.py:45` automatically checks the OS and loads the corresponding model.

<hr>
