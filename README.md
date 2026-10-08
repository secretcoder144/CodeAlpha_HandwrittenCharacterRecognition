# CodeAlpha_HandwrittenCharacterRecognition

CNN that recognizes handwritten digits (MNIST) and digits + letters (EMNIST balanced, 47 classes).

## Setup
```
pip install tensorflow tensorflow-datasets scikit-learn matplotlib pillow numpy
```

## Run
```
python train.py --dataset mnist
python train.py --dataset emnist --epochs 20
python train.py --dataset emnist --predict my_char.png
```

## Outputs
- `<dataset>_cnn.keras` - trained model
- `<dataset>_training_curves.png`
- `<dataset>_confusion_matrix.png`
- `<dataset>_sample_predictions.png`

## Model
- 4 conv layers (32, 32, 64, 64) with batch norm, max pooling, dropout
- Light data augmentation (rotation, zoom)
- Dense 256 -> softmax
- Early stopping + learning-rate reduction

## Results
Fill in after running: test accuracy = h: 92.0%
                                       n: 7.5%
                                       M: 0.4%
