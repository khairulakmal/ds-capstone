import os
import torch
import torchaudio
import re
from flask import Flask, request, jsonify
from transformers import WhisperProcessor, WhisperForConditionalGeneration, MarianMTModel, MarianTokenizer
import noisereduce as nr

# Flask setup
app = Flask(__name__)

# Load models
processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="ko", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("spow12/whisper-medium-zeroth_korean").to("cuda" if torch.cuda.is_available() else "cpu")
translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en").to("cuda" if torch.cuda.is_available() else "cpu")

# Helper functions
def resample_audio(file_path, target_sr=16000):
    waveform, original_sr = torchaudio.load(file_path)
    if original_sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)(waveform)
    return waveform.squeeze(0).numpy(), target_sr

def reduce_noise(audio_data, sr=16000):
    return nr.reduce_noise(y=audio_data, sr=sr)

def deduplicate_text(text):
    phrase_pattern = re.compile(r'(\b\w+\b(?:\s+\b\w+\b){0,5})\s+\1\b')
    text = phrase_pattern.sub(r'\1', text)
    word_pattern = re.compile(r'(\b\w+\b)(\s+\1)+')
    text = word_pattern.sub(r'\1', text)
    korean_redundancy_pattern = re.compile(r'(\b[가-힣]+\b)(\s+\1)+')
    text = korean_redundancy_pattern.sub(r'\1', text)
    return text.strip()

def insert_punctuation(text):
    sentence_endings = re.compile(r'([가-힣]+(다|요|죠|니|합니다|입니다|했어요|인가요|있나요|하나요|하니|래요|려나요|군요|던가요|였나요))(\s|$)')
    punctuated_text = sentence_endings.sub(r'\1. ', text)
    punctuated_text = re.sub(r'\s+', ' ', punctuated_text).strip()
    return punctuated_text

def translate_to_english(text):
    sentences = text.split('. ')
    translated_sentences = []
    for sentence in sentences:
        translated = translation_model.generate(
            **translation_tokenizer(sentence, return_tensors="pt", padding=True).to(translation_model.device)
        )
        translated_sentences.append(translation_tokenizer.decode(translated[0], skip_special_tokens=True))
    return translated_sentences

@app.route('/process', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    file_path = "./uploads/uploaded_audio.wav"
    audio_file.save(file_path)

    try:
        # Preprocess audio
        audio_data, frame_rate = resample_audio(file_path)
        audio_data = reduce_noise(audio_data, sr=frame_rate)

        # Transcription
        input_features = processor(audio=audio_data, sampling_rate=frame_rate, return_tensors="pt").input_features.to(model.device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        # Post-processing
        deduplicated_text = deduplicate_text(text)
        punctuated_text = insert_punctuation(deduplicated_text)

        # Translation
        translated_sentences = translate_to_english(punctuated_text)

        # Create a list of transcription-translation pairs
        transcription_sentences = punctuated_text.split('. ')
        result = [
            {"transcription": trans, "translation": translt}
            for trans, translt in zip(transcription_sentences, translated_sentences)
        ]

        return jsonify({"pairs": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided for translation"}), 400

    translated_text = translate_to_english(text)
    return jsonify({"translation": translated_text})


if __name__ == '__main__':
    app.run(debug=True)
