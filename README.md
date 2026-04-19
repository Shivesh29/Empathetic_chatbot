 Emotion-Aware Chatbot using TACO + BlenderBot
 Overview

This project is an emotion-aware conversational AI system that detects user emotion from text and generates emotionally-aware responses using a pretrained dialogue model.

It combines:

A DeBERTa-based emotion embedding model (TACO-style pipeline)
A contrastive + classification training objective
A BlenderBot generative model for response generation
 
 
 Key Features
Emotion classification from raw text
Embedding-based emotion space learning
Cluster + label similarity constraints
Emotion-conditioned chatbot responses
27-class emotion detection (GoEmotions dataset)
 
 
 Dataset Used
GoEmotions Dataset (Google Research)
Source: HuggingFace Datasets
Labels: 27 emotion categories
Link:
https://huggingface.co/datasets/go_emotions
 
 
 Model Architecture
1. Emotion Encoder (TACO Model)
Base: microsoft/deberta-v3-small
Projection head → 128-d emotion embeddings
L2 normalization for cosine similarity
2. Loss Functions
Cross Entropy Loss (classification)
Contrastive Cluster Loss (KMeans-based)
Label Similarity Regularization
3. Generator Model
facebook/blenderbot-400M-distill
Conditioned on detected emotion
 
 
 How to Run
1. Install dependencies
pip install -r requirements.txt
2. Train the emotion model
python taco_pipeline.py

This will:

Load dataset
Train DeBERTa-based encoder
Save model as:
taco_final.pth
3. Run chatbot
python generator.py
 Example Interaction
User: I failed my exam today
>> Detected Emotion: SADNESS → SAD
Bot: I'm really sorry to hear that. It must be tough for you.

User: I got selected for internship!
>> Detected Emotion: JOY → HAPPY
Bot: That's amazing news! Congratulations!

⚙️ System Requirements
Python 3.8+
PyTorch (CUDA recommended)
8GB+ RAM recommended
GPU optional but faster training
 Future Improvements
Replace KMeans with learned clustering loss
Use RoBERTa-large for better embeddings
Fine-tune BlenderBot on emotion-conditioned dialogue datasets
Add real-time web UI (Streamlit/React)
 Author
Emotion-aware NLP project for academic/research demonstration
Built using HuggingFace Transformers + PyTorch
