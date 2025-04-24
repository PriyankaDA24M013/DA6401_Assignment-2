# DA6401_Assignment-2

This project explores two approaches to image classification:

1. **Custom CNN Model**: A 5-layer convolutional neural network with configurable architecture
2. **Transfer Learning**: Fine-tuning a pre-trained ResNet50 model

Both approaches aim to accurately classify images from the iNaturalist dataset into their respective biological categories.

## Dataset

The project uses the iNaturalist_12K dataset, which contains:

- Training images organized in class-specific folders
- Validation set for model evaluation
- 10 biological classes including Amphibia and others

All images are resized to 224x224 pixels for consistent input dimensions.

## Part A: Custom CNN Implementation

### Model Architecture

- 5 convolutional layers, each followed by activation and max-pooling
- Configurable parameters include:
  - Number and size of filters
  - Filter scheme (same, double, or halving)
  - Activation function (ReLU, GELU, SiLU, Mish)
  - Batch normalization
  - Dropout rate
  - Number of neurons in the dense layer

### Training Process

- Data augmentation options (horizontal flips, rotations)
- Stratified train/validation split
- Hyperparameter tuning via Weights & Biases sweeps
- Adam optimizer

### Best Configuration

```python
best_config = {
    "epochs": 25,
    "lr": 0.0001,
    "batch_size": 32,
    "num_filters": 64,
    "filter_scheme": "same",
    "activation": "silu",
    "dropout": 0.3,
    "dense_neurons": 256,
    "data_aug": True,
    "batch_norm": True
}
```

## Part B: Transfer Learning with ResNet50

### Implementation Details

- Pre-trained ResNet50 model with frozen feature extraction layers
- Only the final fully-connected layer is trained
- Images normalized using ImageNet statistics
- Adam optimizer with learning rate of 0.0001
- 20 training epochs

## Results and Visualization

Both models log their performance metrics to Weights & Biases, including:

- Training and validation accuracy
- Test set predictions with ground truth comparisons
- Visual prediction grids showing model performance

## Dependencies

- PyTorch
- torchvision
- scikit-learn
- matplotlib
- Weights & Biases (wandb)
- NumPy
- OpenCV (cv2)

## Usage

1. Install the required dependencies:

```bash
pip install torch torchvision scikit-learn matplotlib wandb numpy opencv-python
```

2. Configure your Weights & Biases API key:

```python
wandb.login(key="your_api_key")
```

3. Run the sweep for best config of Custom CNN model:

```python
train_model(best_config, sweep=True)
```

4. Run the Custom CNN model:

```python
train_model(best_config, sweep=False)
```

## Monitoring and Evaluation

Both models use Weights & Biases for experiment tracking. View real-time training metrics, compare model configurations, and analyze prediction quality through the wandb dashboard.
