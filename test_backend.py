import os
import tempfile
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
import librosa
import torch
import re
from transformers import WhisperProcessor, WhisperForConditionalGeneration, MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForCausalLM
from summarization import initialize_summarization_model, generate_summary
from keywords import initialize_kw_model, get_keywords
from textblob import TextBlob
from langdetect import detect
from nltk.tokenize import sent_tokenize

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

def clean_pdf_text(text):
    """Clean up PDF text to remove common artifacts."""
    text = re.sub(r'Page\s\d+|\d+\s+of\s+\d+', '', text)  # Remove page numbers
    text = re.sub(r'\f', ' ', text)  # Remove form feed characters
    text = re.sub(r'\n+', '\n', text)  # Collapse multiple newlines
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    text = re.sub(r'^\s*[\d\*\-\•]+\s+', '', text, flags=re.MULTILINE)  # Remove bullets/numbers
    text = re.sub(r'Copyright.*?\n', '', text, flags=re.IGNORECASE)  # Remove copyright lines
    text = re.sub(r'(All rights reserved|Confidential)', '', text, flags=re.IGNORECASE)
    return text.strip()

def extract_text_from_pdf(pdf_path):
    """
    Extract and optionally translate text from PDF slides.
    :param pdf_path: Path to the PDF file.
    :return: Cleaned and translated text from the PDF.
    """
    try:
        reader = PdfReader(pdf_path)
        texts = []  # Initialize as a list to store strings from each page

        for page in reader.pages:
            # Extract text from the PDF page
            extracted_text = page.extract_text()
            
            if not extracted_text:  # Skip empty or non-extractable pages
                continue
            
            # Detect language
            detected_lang = detect(extracted_text)

            if detected_lang != 'en':
                # Translate to English if not already English
                text_blob = TextBlob(extracted_text)
                translated_text = translate_to_english(str(text_blob))  # Ensure translation returns a string
                if isinstance(translated_text, list):
                    translated_text = " ".join(translated_text)  # Convert list to string if needed
                texts.append(translated_text)  # Append translated text to the list
            else:
                texts.append(extracted_text)  # Append the original text if already in English

        # Concatenate all page texts and clean them
        final_text = "\n".join(texts)  # Join all text snippets into a single string
        return clean_pdf_text(final_text.strip())  # Clean the concatenated text

    except Exception as e:
        return {"error": f"Failed to process PDF: {e}"}

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

def chunk_audio(file_path, chunk_duration=15, sample_rate=16000):
    """
    Split an audio file into smaller chunks.
    :param file_path: Path to the audio file.
    :param chunk_duration: Duration of each chunk in seconds.
    :param sample_rate: Sampling rate for audio processing.
    :return: List of tuples (start_time, end_time, audio_chunk).
    """
    try:
        audio_duration = librosa.get_duration(filename=file_path)
        chunks = []

        for start_time in range(0, int(audio_duration), chunk_duration):
            end_time = min(start_time + chunk_duration, audio_duration)
            audio_chunk, _ = librosa.load(file_path, sr=sample_rate, offset=start_time, duration=chunk_duration)
            chunks.append((start_time, end_time, audio_chunk))

        return chunks
    except Exception as e:
        return {"error": str(e)}


def transcribe_chunk(audio_chunk, sample_rate=16000):
    """
    Transcribe a single audio chunk using Whisper.
    :param audio_chunk: The audio data array.
    :param sample_rate: Sampling rate of the audio.
    :return: Transcription text.
    """
    try:
        # Prepare input for Whisper
        input_features = processor(audio_chunk, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            deduplicated_text = deduplicate_text(transcription)
            punctuated_text = insert_punctuation(deduplicated_text)

        return punctuated_text
    except Exception as e:
        return {"error": str(e)}

def process_audio_chunks(file_path, chunk_duration=15):
    """
    Process an audio file by chunking, transcribing, and ensuring sentence alignment across chunks.
    :param file_path: Path to the audio file.
    :param chunk_duration: Duration of each chunk in seconds.
    :return: List of transcriptions for each chunk with aligned sentences.
    """
    try:
        chunks = chunk_audio(file_path, chunk_duration)

        if isinstance(chunks, dict) and "error" in chunks:
            return chunks

        transcriptions = []
        carryover_text = ""  # To hold incomplete sentences for the next chunk

        for start_time, end_time, audio_chunk in chunks:
            # Transcribe the current chunk
            transcription = transcribe_chunk(audio_chunk)
            if isinstance(transcription, dict) and "error" in transcription:
                return transcription

            # Append carryover text from the previous chunk
            complete_text = carryover_text + " " + transcription

            # Split into sentences and handle carryover
            sentences = re.split(r'(?<=[.!?])\s+', complete_text)
            carryover_text = sentences.pop() if sentences and not complete_text.endswith(('.', '!', '?')) else ""

            # Store finalized sentences for this chunk
            finalized_transcription = " ".join(sentences)

            transcriptions.append({
                "start_time": start_time,
                "end_time": end_time,
                "transcription": finalized_transcription
            })

        # Handle any remaining carryover text after processing all chunks
        if carryover_text:
            transcriptions[-1]["transcription"] += " " + carryover_text

        return transcriptions
    except Exception as e:
        return {"error": str(e)}

def translate_to_english(text):
    sentences = sent_tokenize(text)
    translated_sentences = []
    for sentence in sentences:
        translated = translation_model.generate(
            **translation_tokenizer(sentence, return_tensors="pt", padding=True).to(translation_model.device)
        )
        translated_sentences.append(translation_tokenizer.decode(translated[0], skip_special_tokens=True))
    return translated_sentences

def translate_to_korean(text, tokenizer, model):
    # Prepare the input in the mT5 format
    prompt = f"""
    ### Instruction:
    주어진 텍스트를 한국어로 번역하세요.
    ### Input:
    {text}
    ### Output: """

    # Tokenize the input
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate the translation
    outputs = model.generate(inputs, max_length=50, temperature=0.1, top_p=1, do_sample=True, repetition_penalty=1.3, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translated_text = translated_text.split("Output: ")[1].strip().split('\n')[0].strip()
    return translated_text

@app.route('/process', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    pdf_file = request.files.get('pdf')  # PDF file is optional
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    audio_file.save(file_path)

    pdf_path = None
    if pdf_file:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
        pdf_file.save(pdf_path)

    try:
        # Process audio chunks
        results = process_audio_chunks(file_path)

        if "error" in results:
            return jsonify({"error": results["error"]}), 500

        # Combine transcriptions for full text
        full_transcription = " ".join([chunk["transcription"] for chunk in results])

        # Translate each transcription into English
        translated_chunks = []
        for chunk in results:
            translated_text = translate_to_english(chunk["transcription"])
            chunk["translation"] = " ".join(translated_text)  # Add translation to the chunk
            translated_chunks.append(chunk)

        # Combine translations for full text
        full_translation = " ".join([chunk["translation"] for chunk in translated_chunks])

        # Extract text from PDF if uploaded
        lecture_material = ""
        if pdf_path:
            lecture_material = extract_text_from_pdf(pdf_path)
            if "error" in lecture_material:
                return jsonify({"error": lecture_material["error"]}), 500

        # Generate study notes
        study_notes = generate_summary(full_translation, lecture_material, tokenizer, summarization_model)
        
        extract_keywords = get_keywords(full_translation, kw_model)
        en_keys = [kw[0] for kw in extract_keywords]
      #  ko_keys = [translate_to_korean(key, en_ko_tokenizer, en_ko_model) for key in en_keys]

        """key_dict = {}
        for i in range(len(en_keys)):
            dic = {f'{ko_keys[i]}': f'{en_keys[i]}'}
            key_dict.update(dic)"""

        return jsonify({
        "chunks": results,
        "full_transcription": full_transcription,
        "full_translation": full_translation,
        "lecture_material": lecture_material,
        "study_notes": study_notes,
        "keywords": en_keys,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":

    # Load Whisper model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="ko", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("spow12/whisper-medium-zeroth_korean")
    model.to(device)
    model.eval()

    translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en").to("cuda" if torch.cuda.is_available() else "cpu")
    en_ko_tokenizer = AutoTokenizer.from_pretrained("kwoncho/Llama-3.2-1B-KO-EN-Translation")
    en_ko_model = AutoModelForCausalLM.from_pretrained("kwoncho/Llama-3.2-1B-KO-EN-Translation").to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, summarization_model = initialize_summarization_model()
    kw_model = initialize_kw_model()
    app.run(host="0.0.0.0", port=5000)
