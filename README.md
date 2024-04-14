# DL-finalProject
# Medical Image Classification and Anomaly Detection

This project focuses on solving medical image classification and anomaly detection tasks using deep neural networks. The following tasks were implemented with specific methodologies:

## Tasks Implemented

1. **Classification of Healthy/Sick X-ray Dataset**:
   - Experimented with pretrained models (ResNet, VGG, AlexNet) for binary classification (healthy/sick).
   - Adapted the final layer of each model to the specific classification task.
   - Achieved the highest accuracy with ResNet on the test set.

2. **Classification of Healthy/Bacterial Pneumonia/Viral Pneumonia**:
   - Applied a similar comparison approach as Task 1a but for multi-class classification.
   - VGG demonstrated the highest accuracy among the pretrained models for this task.

3. **Embedding Vector and KNN Classification**:
   - Modified the selected model from Task 1 to remove the final layer.
   - Used extracted features (embedding vectors) as input data for training a K-Nearest Neighbors (KNN) classifier.
   - Visualized the embedding vectors using t-SNE for dimensional reduction and class visualization.

4. **Anomaly Detection Using Autoencoder (AE)**:
   - Trained an autoencoder (AE) model on a dataset containing only "healthy" images.
   - Utilized AE for image reconstruction and anomaly detection based on reconstruction error.
   - Employed a threshold-based approach derived from Receiver Operating Characteristic (ROC) curve analysis for anomaly detection.

5. **Explainability Analysis**:
   - Implemented heatmap explainability to interpret the model's decision-making process.
   - Generated heatmaps with occlusion to analyze and visualize the regions of interest for both normal and pneumonia images.

## Instructions

1. **Setup and Dependencies**:
   - Ensure Python and required libraries (e.g., PyTorch, scikit-learn, matplotlib) are installed.
   - Use Jupyter Notebook or another development environment to execute the provided scripts and notebooks.

2. **Data Preparation**:
   - Organize the dataset into appropriate directories for training, validation, and testing.
   - Ensure the dataset includes categories for binary (healthy/sick) and multi-class (healthy/bacterial pneumonia/viral pneumonia) classification.

3. **Model Training and Evaluation**:
   - Train the models using appropriate hyperparameters and architectures specified for each task.
   - Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.

4. **Anomaly Detection and Explainability**:
   - Implement the anomaly detection method (autoencoder) and explainability technique (heatmap) following the described approach.
   - Document and visualize the results to understand model behavior and performance comprehensively.

Explore different models and methods to optimize performance and gain valuable insights into medical image analysis and anomaly detection.

