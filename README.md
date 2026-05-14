# 🎵  Music Genre Classification Using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?style=for-the-badge&logo=streamlit)
![Librosa](https://img.shields.io/badge/Librosa-AudioProcessing-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Completed-success?style=for-the-badge)

### 🚀 AI-Powered Music Genre Prediction System

Predict music genres from uploaded audio files using Deep Learning, Mel Spectrograms, and CNNs.

</div>

---

# 📌 Overview

This project is an intelligent Music Genre Classification System developed using Deep Learning techniques.

The application analyzes uploaded audio files by converting them into **Mel Spectrograms** and then uses a **Convolutional Neural Network (CNN)** to predict the music genre.

The system provides:

✅ Genre Prediction  
✅ Confidence Scores  
✅ Audio Waveform Visualization  
✅ Mel Spectrogram Visualization  
✅ Interactive Probability Charts  
✅ Chunk-wise Genre Analysis  
✅ Downloadable Prediction Results  

---

# 🌟 Project Demo

## 🎧 Prediction Dashboard

- Upload `.mp3` or `.wav` files
- Visualize waveform and spectrogram
- Predict genre instantly
- Analyze top predictions

---

# 🎼 Supported Genres

| Genre | Genre | Genre |
|---|---|---|
| 🎸 Rock | 🎹 Classical | 🎤 Pop |
| 🥁 Jazz | 🔥 Hiphop | 🤘 Metal |
| 🎶 Blues | 💃 Disco | 🌴 Reggae |
| 🤠 Country |  |  |

---

# 🧠 Deep Learning Workflow

```text
Audio File
     ↓
Audio Preprocessing
     ↓
Mel Spectrogram Generation
     ↓
CNN Feature Extraction
     ↓
Genre Prediction
     ↓
Confidence Analysis
     ↓
Interactive Visualization
```

---

# ⚡ Features

## 🎵 Smart Audio Upload
- Supports MP3 and WAV files
- Built using Streamlit file uploader

## 🔥 Mel Spectrogram Visualization
- Converts audio frequencies into image representations
- Helps CNN detect music patterns

## 📊 AI Genre Prediction
- CNN predicts the most probable music genre
- Displays confidence score

## 📈 Interactive Analytics
- Top 3 genre probabilities
- Probability-over-time graph
- Chunk-wise predictions

## 📥 Download Results
- Export prediction results as CSV

---

# 🛠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core Programming Language |
| TensorFlow / Keras | Deep Learning Framework |
| Streamlit | Interactive Web Application |
| Librosa | Audio Processing |
| Plotly | Interactive Graphs |
| NumPy | Numerical Computation |
| Pandas | Data Handling |
| Matplotlib | Visualization |

---

# 📂 Project Structure

```text
AI-Music-Genre-Classifier/
│
├── genre_app.py
├── Trained_model.keras
├── music_image.png
├── training_hist.json
├── requirements.txt
├── README.md
├── .gitignore
├── Train_Music_Genre.ipynb
└── Test_music_genre.ipynb
```

---

# ⚙️ Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Music-Genre-Classifier.git
```

## 2️⃣ Open Project Folder

```bash
cd AI-Music-Genre-Classifier
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run genre_app.py
```

After running, Streamlit will automatically open the application in your browser.

---

# 📊 Dataset Information

This project uses the **GTZAN Music Genre Dataset**.

### Dataset Details

- 10 Genres
- 100 Audio Files Per Genre
- 30-Second Audio Clips
- WAV Format

---

# 🎧 Audio Processing Pipeline

The uploaded audio file goes through several preprocessing stages:

## 🔹 Resampling
Audio is standardized to a fixed sample rate.

## 🔹 Chunk Splitting
Audio is divided into smaller chunks for better prediction.

## 🔹 Mel Spectrogram Conversion
Transforms audio into visual frequency representation.

## 🔹 CNN Analysis
Deep Learning model extracts audio features.

---

# 📈 Visualizations Included

## 🎧 Audio Waveform
Displays amplitude variations over time.

## 🔥 Mel Spectrogram
Shows frequency distribution of audio.

## 📊 Genre Probability Distribution
Displays prediction confidence across genres.

## 📈 Probability Over Time
Shows chunk-wise prediction changes.

---

# 🎯 Model Information

## Architecture

- Convolutional Neural Network (CNN)
- Mel Spectrogram Input
- Multi-class Classification

## Model Capabilities

✅ Learns audio patterns  
✅ Identifies genre characteristics  
✅ Handles multiple genres  
✅ Provides probability-based predictions  

---

# 🚀 Performance Optimizations

The application includes:

- Faster preprocessing
- Optimized spectrogram generation
- Reduced chunk overlap
- Cached model loading
- Interactive UI enhancements

---

# 🌐 Future Enhancements

🚀 Real-time microphone prediction  
🚀 Spotify API integration  
🚀 More genre categories  
🚀 Transformer-based models  
🚀 Mobile app deployment  
🚀 Cloud deployment  

---

# 💡 Real-World Applications

🎵 Music Streaming Platforms  
🎵 Music Recommendation Systems  
🎵 Automatic Audio Tagging  
🎵 AI Music Assistants  
🎵 Music Analytics Platforms  

---

# 👨‍💻 Author

Developed as a Deep Learning project focused on:

- Audio Classification
- Music Information Retrieval
- CNN-based Prediction Systems
- Interactive AI Applications

---

# 📜 License

This project is intended for:

- Educational Purposes
- Research Projects
- Learning Deep Learning Concepts

---

# ⭐ Support

If you found this project useful:

✅ Star the repository  
✅ Share with others  
✅ Fork the project  
✅ Give feedback  

---

<div align="center">

# 🎶 Thank You for Visiting the Project 🎶

### Made with ❤️ using Python, TensorFlow, and Streamlit

</div>
