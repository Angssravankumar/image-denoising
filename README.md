# image-denoising
Comprehensive Report: From Dataset Preparation to RIDNet Implementation
1. Dataset Preparation
1.1 Dataset Structure

High-quality images directory: train_high/
Low-quality images directory: train_low/
Test directories: test_high/ and test_low/
1.2 Data Loading and Preprocessing

DataGenerator Class: A custom data generator to efficiently load and preprocess image batches during training and testing.
2. Autoencoder Model for Denoising
2.1 Model Definition

An Autoencoder model with an Encoder and Decoder architecture.
Encoder: Consists of convolutional layers with ReLU activations and max-pooling.
Decoder: Consists of transposed convolutional layers to upsample the feature maps.
2.2 Model Compilation

Optimizer: Adam with a learning rate of 1e-3.
Loss function: Mean Squared Error (MSE).
2.3 Training the Autoencoder

Batch size: 32
Number of epochs: 20 (initially 10, then increased to 20 for better performance)
Model Checkpoint: To save the best model based on validation loss.
3. Evaluation Metrics
3.1 PSNR (Peak Signal-to-Noise Ratio)

Measures the ratio between the maximum possible power of a signal and the power of corrupting noise.
Higher PSNR indicates better image quality.
3.2 SSIM (Structural Similarity Index)

Measures the similarity between two images.
Values range from -1 to 1, where 1 indicates perfect similarity.
3.3 Evaluation Function

A function to denoise images using the trained model and compute average PSNR and SSIM values.
4. RIDNet Model
4.1 RIDNet Architecture

Feature Extraction Module: Initial convolutional layer to extract basic features.
Residual on Residual Learning Module (EAMs): Series of Enhancement Attention Modules (EAMs) that focus on learning residuals.
Reconstruction Module: Final convolutional layer to reconstruct the denoised image with a residual connection to the input.
4.2 Enhancement Attention Module (EAM)

Consists of convolutional layers with attention mechanisms to enhance important features.
4.3 Model Compilation

Optimizer: Adam with a learning rate of 1e-3.
Loss function: Mean Squared Error (MSE).
5. Training RIDNet
5.1 Training Process

Batch size: 32
Number of epochs: 20
Model Checkpoint: To save the best model based on validation loss.
5.2 Saving the Model

Saving the final trained RIDNet model as RIDNet_final_model.h5.
Saving the best model during training as RIDNet_best_model.h5.
6. Evaluating RIDNet
6.1 Denoising and Evaluation Function

The function denoise_and_evaluate is used to compute PSNR and SSIM values for the test dataset using the trained RIDNet model.
6.2 Results

Average PSNR and SSIM: Calculated and printed for the test set.
Saving Predictions: The denoised images are saved in the ./test/predicted/ directory.
7. Final Script
The final script consolidates all steps, including data loading, model definition, training, evaluation, and saving predictions.

Key Points about RIDNet
RIDNet (Residual Image Denoising Network): Designed to remove noise from images while preserving important details.
Attention Mechanisms: EAMs help the model to focus on significant features and enhance them.
Residual Learning: Helps in better convergence by learning the difference between the input and the desired output.
Performance Metrics: PSNR and SSIM values are used to evaluate the quality of denoised images, ensuring that the model output is as close to the high-quality images as possible.
This report summarizes our entire process and provides a comprehensive overview of the steps involved in implementing and evaluating the RIDNet model for image denoising.







