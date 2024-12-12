import streamlit as st
import requests
from datetime import timedelta
import re

def sanitize_markdown(text):
    """
    Clean up and format the generated Markdown for correct rendering in Streamlit.
    """
    # Remove unintended code block markers (e.g., triple backticks)
    text = text.replace("```", "").replace("`", "")
    
    # Ensure proper Markdown for bullet points
    text = re.sub(r"(?<!\n)\*\s*", r"\n* ", text)  # Ensure each bullet starts on a new line

    # Remove blank bullet points (e.g., lines starting with "*" but no content)
    text = re.sub(r"^\*\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"--\s*(.*?)\s*--|#+\s*(.*)", r"#### \1\2", text, flags=re.MULTILINE)

    # Remove unintended leading spaces that could cause code block formatting
    lines = text.splitlines()
    sanitized_lines = [line.lstrip() for line in lines if line.strip()]  # Remove leading spaces and blank lines
    return "\n".join(sanitized_lines)



st.title("Lecture Buddy")

# File uploaders
audio_file = st.file_uploader("Upload Lecture Audio", type=["wav", "mp3", "m4a"])
pdf_file = st.file_uploader("Upload Lecture Slides (Optional)", type=["pdf"])

# Submit button
if st.button("Process"):
    if not audio_file:
        st.error("Please upload an audio file.")
    else:
        with st.spinner("Processing..."):
            files = {
                "audio": audio_file,
            }
            if pdf_file:
                files["pdf"] = pdf_file

            response = requests.post("http://163.180.160.88:5000/process", files=files)
            if response.status_code == 200:
                data = response.json()
                chunks = data.get("chunks", [])
                transcript = ""
                # Iterate over chunks to display transcriptions and translations
                for idx, chunk in enumerate(chunks):
                    start_time = chunk.get("start_time", "N/A")
                    timestamp = str(timedelta(seconds=int(start_time)))
                    st.subheader(timestamp)
                    transcription = chunk.get("transcription", "N/A")
                    translation = chunk.get("translation", "N/A")
                    st.text(transcription)
                    st.text(translation)
                    st.text("")  # Empty line for spacing
                    transcript += f"{timestamp}\n{transcription}\n{translation}\n\n"

                #st.subheader("Full Transcription") # redundant
                #st.text(data["full_transcription"])

                trans_btn = st.download_button("Download the full transcript with translation", transcript, file_name='transcript.txt')

                st.subheader("Study Notes")
                summary = sanitize_markdown(data["study_notes"])
                st.markdown(summary)

                summary_btn = st.download_button("Download the summary", summary, file_name='summary.md')

                if data["keywords"]:
                    st.subheader("Keywords")
                    for item in data['keywords']:
                        st.text(item)
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")