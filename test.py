import torch
import numpy as np
import speech_recognition as sr
from transformers import WhisperProcessor, WhisperForConditionalGeneration, MarianMTModel, MarianTokenizer
from queue import Queue
from threading import Thread
import re
import noisereduce as nr

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="ko", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("spow12/whisper-medium-zeroth_korean").to("cuda" if torch.cuda.is_available() else "cpu")

# Load MarianMT model and tokenizer for translation
translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en").to("cuda" if torch.cuda.is_available() else "cpu")

# Set up recognizer and microphone
recognizer = sr.Recognizer()
mic = sr.Microphone(sample_rate=16000)

# Queue to hold audio data
MAX_QUEUE_LENGTH = 5  # Limit the number of chunks in the queue
audio_queue = Queue()
full_transcription = []  # Store the entire transcription
stop_flag = False  # Flag to signal when to stop transcription

# Adjust for ambient noise to set energy threshold
with mic as source:
    print("Calibrating microphone for ambient noise...")
    recognizer.adjust_for_ambient_noise(source)
    print("Calibration complete. Start speaking. Type 'q' to quit.")

def reduce_noise(audio_data, sr=16000):
    # Use noisereduce to apply spectral gating noise reduction
    # This method does not rely on assuming a fixed noise sample duration
    noise_reduced_audio = nr.reduce_noise(y=audio_data, sr=sr)
    return noise_reduced_audio

# Function to detect pauses and split based on silence
def detect_pauses_and_split(audio_data, threshold=0.01, min_pause_duration=0.5):
    audio_energy = np.sqrt(np.mean(audio_data ** 2))
    return audio_energy < threshold

# Enhanced deduplication function to remove repetitive words and phrases
def deduplicate_text(text):
    # Remove consecutive duplicate phrases of up to six words
    phrase_pattern = re.compile(r'(\b\w+\b(?:\s+\b\w+\b){0,5})\s+\1\b')
    text = phrase_pattern.sub(r'\1', text)

    # Remove consecutive duplicate words
    word_pattern = re.compile(r'(\b\w+\b)(\s+\1)+')
    text = word_pattern.sub(r'\1', text)

    # Handle specific Korean redundancy
    korean_redundancy_pattern = re.compile(r'(\b[가-힣]+\b)(\s+\1)+')
    text = korean_redundancy_pattern.sub(r'\1', text)

    return text.strip()

# Heuristic function to insert punctuation based on linguistic cues
def insert_punctuation(text):
    sentence_endings = re.compile(
        r'([가-힣]+(다|요|죠|니|나|네|합니다|입니다|했어요|인가요|있나요|하나요|하니|래요|려나요|군요|던가요|였나요))(\s|$)'
    )
    punctuated_text = sentence_endings.sub(r'\1. ', text)
    punctuated_text = re.sub(r'\s+', ' ', punctuated_text).strip()
    return punctuated_text

# Function to translate text from Korean to English
def translate_to_english(text):
    # Segment the text into sentences or logical chunks
    sentences = text.split('. ')
    translated_sentences = []
    
    for sentence in sentences:
        translated = translation_model.generate(
            **translation_tokenizer(sentence, return_tensors="pt", padding=True).to(translation_model.device)
        )
        translated_sentences.append(translation_tokenizer.decode(translated[0], skip_special_tokens=True))
    
    # Join translated sentences for smoother output
    return ' '.join(translated_sentences).strip()

# Function to listen for 'q' input to quit
def listen_for_exit():
    global stop_flag
    while True:
        if input() == 'q':  # Wait for user to type 'q' to quit
            stop_flag = True
            break

# Function to listen and add audio chunks to the queue
def listen():
    global stop_flag
    with mic as source:
        while not stop_flag:
            try:
                print("\nListening for audio segment...")
                audio = recognizer.listen(source, phrase_time_limit=5)

                if audio_queue.qsize() >= MAX_QUEUE_LENGTH:
                    print("Queue is full. Discarding oldest audio chunk to keep up with real-time.")
                    audio_queue.get()

                audio_queue.put(audio)
            except Exception as e:
                print(f"Error capturing audio chunk: {e}")
                break

        audio_queue.put(None)

# Function to transcribe audio chunks from the queue
def transcribe():
    while True:
        audio = audio_queue.get()
        if audio is None:
            break

        try:
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
            audio_data = reduce_noise(audio_data)

            if detect_pauses_and_split(audio_data):
                print("Detected a pause in speech. Splitting transcription.")

            input_features = processor(audio=audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(model.device)

            print("Transcribing audio segment...")
            with torch.no_grad():
                predicted_ids = model.generate(input_features)
                text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            deduplicated_text = deduplicate_text(text)
            punctuated_text = insert_punctuation(deduplicated_text)

            full_transcription.append(punctuated_text)

            print("Current Chunk Transcription:", punctuated_text)

        except Exception as e:
            print(f"Error processing audio chunk: {e}")

listener_thread = Thread(target=listen)
transcriber_thread = Thread(target=transcribe)
exit_listener_thread = Thread(target=listen_for_exit)

listener_thread.start()
transcriber_thread.start()
exit_listener_thread.start()

listener_thread.join()
transcriber_thread.join()
exit_listener_thread.join()

final_transcription = deduplicate_text(" ".join(full_transcription))

print("\n\nFinal Transcription (Korean):")
print(final_transcription)

translated_text = translate_to_english(final_transcription)
print("\n\nTranslated Text (English):")
print(translated_text)
