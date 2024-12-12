# Translation and Summarisation of Korean Lectures

This project automates the transcription, translation, and summarisation of Korean lecture content, designed to aid international students in overcoming language barriers. The system integrates models, including Whisper for transcription, Helsinki-NLP for translation, and Llama-3.2 for summarisation, to provide well-structured study notes.

## Features

- **Transcription**: Converts Korean audio into text using the Whisper model.
- **Translation**: Translates Korean text into English using the Helsinki-NLP model.
- **Summarisation**: Generates concise and meaningful study notes with the Llama-3.2 model.
- **PDF Integration**: Extracts and integrates text from uploaded lecture slides.
- **Chunk-Based Processing**: Efficiently handles long audio files by splitting them into manageable 15-second chunks.

## System Workflow

1. **Input**:
   - Upload a lecture audio file (WAV/MP3) and optional lecture slides (PDF).
2. **Processing**:
   - Audio is transcribed using Whisper.
   - Transcripts are translated into English.
   - Text from lecture slides is preprocessed and integrated.
3. **Output**:
   - The system produces study notes combining transcripts and slides.
   - Outputs are presented in a user-friendly format.

## Technologies Used

- **Transcription**: Whisper (fine-tuned on KsponSpeech dataset)
- **Translation**: Helsinki-NLP (Opus-MT for Korean-English)
- **Summarisation**: Llama-3.2-3B-Instruct
- **Frontend**: Streamlit
- **Backend**: Flask
- **Text Processing**: PyPDF2, TextBlob, KeyBERT

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/khairulakmal/ds-capstone.git
   cd ds-capstone
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   python test_backend.py
   ```

4. Start the frontend:
   ```bash
   streamlit run test_frontend.py
   ```

## How to Use

1. Access the interface through your web browser.
2. Upload the lecture audio file and optional PDF slides.
3. Click "Process" to generate transcription, translation, and summary.
4. Download the generated study notes.
