import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import datetime
import plotly.graph_objects as go
import pandas as pd
import os
from tensorflow.image import resize

# Genre Labels
LABELS = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("Trained_model.keras")

def load_and_preprocess_file(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        if len(chunk) < chunk_samples:
            break
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data), audio_data, sample_rate

def model_prediction(model, X_test):
    y_pred = model.predict(X_test)
    chunk_wise_best = [np.argmax(chunk) for chunk in y_pred]
    most_common_chunk_label = max(set(chunk_wise_best), key=chunk_wise_best.count)
    chunkwise_confidences = [chunk[most_common_chunk_label] for chunk in y_pred]
    avg_confidence = np.mean(chunkwise_confidences)
    return most_common_chunk_label, avg_confidence, y_pred

# Sidebar Navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Home Page
if app_mode == "Home":
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(to right, #141e30, #243b55);
            color: white;
        }
        .big-title {
            font-size: 3em;
            font-weight: bold;
            background: -webkit-linear-gradient(#ff6e7f, #bfe9ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .center {
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="big-title center">🎶 Welcome to the Music Genre Classifier 🎧</h1>', unsafe_allow_html=True)
    st.image("music_image.png", use_container_width=True)
    st.markdown("""
    <div class='center'>
        <h3>Discover what genre your music belongs to!</h3>
        <p>🎼 Upload your track, and our AI will analyze its style using deep learning.</p>
    </div>

    ### 💡 How It Works
    - *Upload Audio:* Go to the *Prediction* tab and upload a .mp3 file.
    - *AI Analysis:* We'll split and process the audio using Mel Spectrograms.
    - *Genre Reveal:* View the predicted genre with a visual graph!

    ### 🚀 Why You'll Love This
    - 🎯 *Accurate Predictions* with a trained deep learning model
    - ⚡ *Fast & Fun* results
    - 🧠 *Educational* for exploring audio and ML

    ### 👨‍💻 About the Project
    Head over to the *About Project* tab to learn about the dataset and model used!
    """, unsafe_allow_html=True)

# About Project Page
elif app_mode == "About Project":
    st.header("🎵 About the Revolutionizing Music Genre Classification Using Deep Learning Project")
    st.markdown("""
This project is built to automatically classify music genres from audio files using deep learning techniques.

Music comes in many genres like pop, rock, jazz, classical, and more. Each genre has unique patterns in rhythm, instruments, and sound. In this project, we trained a machine learning model to understand these patterns and recognize the genre of a song.
                 
This project is a deep learning application that classifies music into genres such as pop, rock, jazz, etc., using a convolutional neural network (CNN) trained on Mel Spectrograms generated from audio files. The model is deployed in a user-friendly Streamlit web app, allowing users to upload .mp3 files and instantly view the predicted genre along with probability scores.

### Objective
The main goal is to create a simple and user-friendly web app where anyone can upload a song, and the app will predict its genre accurately.

### How It Works
- You upload a music file (.mp3 format).
- The app processes the audio and extracts features using a technique called *Mel Spectrogram*, which turns sound into a visual format.
- A pre-trained deep learning model then analyzes the patterns in the spectrogram.
- It predicts which genre the music most likely belongs to.

### Dataset Used
We used the *GTZAN Music Genre Dataset*, which contains 1000 audio files across 10 different genres:
- blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock

### Technologies Used
- *TensorFlow / Keras* for building the model
- *Librosa* for audio feature extraction
- *Streamlit* for creating the web app
- *NumPy & Pandas* for data handling

### Why This Project is Important
Music genre classification is helpful in many real-world applications:
- Music streaming apps use it for recommendations
- It helps organize and label large music libraries
- It allows for better understanding of musical patterns through AI
    """)

# Prediction Page
elif app_mode == "Prediction":
    st.header("🎤 Upload and Predict Genre")

    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])

    if test_mp3 is not None:
        st.session_state.uploaded_file = test_mp3

    if st.session_state.uploaded_file is not None:
        test_mp3 = st.session_state.uploaded_file
        filepath = 'Test_Music/' + test_mp3.name
        os.makedirs("Test_Music", exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(test_mp3.read())

        st.audio(test_mp3)

        # === Audio Visualizations ===
        st.markdown("### 📊 Audio Visualizations")

        audio_data, sr = librosa.load(filepath)

        # Waveform
        fig_wave, ax_wave = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(audio_data, sr=sr, ax=ax_wave)
        ax_wave.set_title("Waveform")
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude")
        st.pyplot(fig_wave)

        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        fig_mel, ax_mel = plt.subplots(figsize=(10, 3))
        img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax_mel, cmap='viridis')
        ax_mel.set_title("Mel Spectrogram (dB)")
        fig_mel.colorbar(img, ax=ax_mel, format="%+2.0f dB")
        st.pyplot(fig_mel)

        # Predict Button
        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                model = load_model()
                X_test, _, _ = load_and_preprocess_file(filepath)
                best_index, confidence, y_pred = model_prediction(model, X_test)
                chunk_start_times = [round(i * 2.0, 2) for i in range(len(y_pred))]

                st.markdown(f"**🎯 Prediction:** :blue[{LABELS[best_index]}] genre with :green[{confidence*100:.2f}%] confidence")

                # 🎼 Genre Confidence Timeline
                st.markdown("## 🎼 Genre Confidence Timeline")

                fig = go.Figure()
                for i, label in enumerate(LABELS):
                    fig.add_trace(go.Scatter(
                        x=chunk_start_times,
                        y=[chunk[i] for chunk in y_pred],
                        mode='lines+markers',
                        name=label,
                        line=dict(width=2)
                    ))

                fig.update_layout(
                    title="Genre Confidence Over Time",
                    xaxis_title="Time (s)",
                    yaxis_title="Probability",
                    yaxis=dict(range=[0, 1]),
                    legend_title="Genres",
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                # 📋 Chunk-wise Most Probable Genre Table
                table_df = pd.DataFrame({
                    "Chunk Start Time (s)": chunk_start_times,
                    "Predicted Genre": [LABELS[np.argmax(chunk)] for chunk in y_pred],
                    "Confidence": [f"{np.max(chunk)*100:.2f}%" for chunk in y_pred]
                })
                st.markdown("### 📋 Chunk-wise Most Probable Genre Table")
                st.dataframe(table_df, use_container_width=True)

                # 📄 Download Report
                report = f"""Music Genre Classification Report
File: {test_mp3.name}
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Prediction:
{LABELS[best_index]} - {confidence*100:.2f}% confidence
"""
                st.download_button("📄 Download Report", report, file_name="genre_prediction_report.txt")
