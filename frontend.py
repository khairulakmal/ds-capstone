import streamlit as st
import requests

st.title("Audio Transcription and Translation")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Upload the audio to the backend
    with st.spinner("Processing audio..."):
        response = requests.post(
            "http://127.0.0.1:5000/process", files={"audio": uploaded_file}
        )

    # Handle response
    if response.status_code == 200:
        result = response.json()
        pairs = result.get("pairs", [])

        # Collect full transcription
        full_transcription = ""

        # Display transcription and translation pairs
        st.header("Transcription and Translation:")
        for pair in pairs:
            transcription = pair.get("transcription", "N/A")
            translation = pair.get("translation", "N/A")
            st.text(f"{transcription}")
            st.text(f"{translation}")
            st.text("")  # Empty line between pairs

            # Add to full transcription
            full_transcription += f"{transcription}. "

        # Display the full transcription at the end
        st.header("Full Transcription:")
        st.write(full_transcription.strip())

    else:
        st.error("Failed to process the audio.")
        st.error(response.json().get("error", "Unknown error."))
