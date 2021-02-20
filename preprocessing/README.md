
## Preprocessing

Raw images could be used by a deep learning model, however we can benefit from preprocessing. On this project the main three preprocess transformation are:

* **KneeLocalizer**: Center image only on region of interest (ROI) which normalize dataset.
* **BackgroundPreprocess**: Cleaning background and noisy parts which helps avoid model memorization due to characteristic background/noise.
* **CLAHE Transform**: Grayscale/Color histogram normalization which helps CNN model training.

See below some examples these transformations:

!["Preprocess steps 1"](https://github.com/lluissalord/radiology_ai/tree/master/docs/images/preprocess_steps.svg "Preprocess steps 1")
!["Preprocess steps 2"](https://github.com/lluissalord/radiology_ai/tree/master/docs/images/preprocess_steps_2.svg "Preprocess steps 2")