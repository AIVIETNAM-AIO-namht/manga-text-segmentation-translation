# UNet++ with EfficientNetV2

## Model

- Architecture: UNet++
- Encoder: EfficientNetV2 (`tu-efficientnetv2_rw_m`)
- Decoder attention: SCSE
- Framework: PyTorch
- Segmentation library: segmentation-models-pytorch
- Dataset: Manga109s
- Task: Manga text segmentation

## Dataset

The model was evaluated on 390 matched Manga109s image/ground-truth pairs, across 39 manga titles.

Ground-truth masks were aligned with the input images before evaluation.

## Preprocessing

The preprocessing follows the official ContemporaryCat inference pipeline:

1. Convert image to RGB.
2. Normalize using ImageNet mean and standard deviation.
3. Convert to tensor.
4. Pad tensor to dimensions divisible by 32.
5. Run model inference.
6. Apply sigmoid.
7. Crop prediction back to original image size.
8. Apply threshold = 0.5.

## Evaluation Metrics

The following metrics are reported:

- IoU
- Precision
- Recall
- F1-score
- Average inference time

Pixel Accuracy is not used as the main metric.

## Results

Mean IoU: 0.451551

Mean Precision: 0.493874

Mean Recall: 0.837330

Mean F1-score: 0.579298

Mean inference time: 0.3263 sec/page

## Folder Structure

```text
UNetPP_EfficientNetV2/
├── masks/
├── metrics_per_page.csv
├── metrics_summary.json
├── errors.json
├── README.md
└── requirements.txt
```

## Model Source

ContemporaryCat - Manga Text Segmentation

https://github.com/ContemporaryCat/Manga-Text-Segmentation
