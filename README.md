# DL-finalProject
Project Overview and Instructions
This project aims to solve several problems related to medical image classification and anomaly detection using deep neural networks. Here's a summary of the tasks implemented and their methodologies:

Tasks Implemented
Classification of Healthy/Sick X-ray Dataset:

Experimented with pretrained models (ResNet, VGG, AlexNet) for binary classification (healthy/sick).
Adapted the final layer of each model to the specific classification task.
Trained and tested the models on the dataset, achieving the highest accuracy with ResNet on the test set.
Classification of Healthy/Bacterial Pneumonia/Viral Pneumonia:

Applied a similar comparison approach as Task 1a but for multi-class classification.
Found that VGG achieved the highest accuracy among the pretrained models for this classification task.
Embedding Vector and KNN Classification:

Modified the selected model from Task 1 to remove the final layer.
Utilized the extracted features (embedding vectors) from the modified model as input data for training a K-Nearest Neighbors (KNN) classifier.
Visualized the embedding vectors using t-SNE (t-Distributed Stochastic Neighbor Embedding) for dimensional reduction and class visualization.
Anomaly Detection Using Autoencoder (AE):

Trained an autoencoder (AE) model on a dataset consisting only of "healthy" images.
Employed the AE model to reconstruct images and calculate the Mean Squared Error (MSE) between input and output images to identify anomalies.
Utilized a threshold-based approach derived from Receiver Operating Characteristic (ROC) curve analysis to detect anomalies effectively.
Explainability Analysis:

Implemented heatmap explainability to interpret the model's decision-making process.
Generated heatmaps with occlusion to analyze and visualize the regions of interest for both normal and pneumonia images.
Observed differences in heatmap patterns between normal and pneumonia images, providing insights into how the model learned to distinguish between classes.
