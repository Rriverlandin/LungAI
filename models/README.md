# Model files

The final runnable model is `notebooks/models/LungAI_V4_final.keras`.

It accepts batches of `224 × 224 × 1` grayscale images and returns one sigmoid
score per image. Scores at or above `0.60` are classified as `PNEUMONIA`.

`mobilenet_v2.h5` is the downloaded ImageNet base used during training. It is
ignored because the final `.keras` artifact already contains inference weights.
