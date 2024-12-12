import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed

def initialize_summarization_model():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct").to(device)

    return tokenizer, model


def truncate_text(text, tokenizer, max_length=2048):
    inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    return tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

def generate_summary(transcript, material, tokenizer, model):
    if material:
        prompt = f"""<s> [INST]
        <<SYS>>
        You are a diligent student creating detailed and comprehensive study notes based on the following lecture and accompanying material. Your goal is to extract key information and explain it in detail. Use the lecture material as a supplement to the lecture transcript (transcript takes priority).
        Your notes must include the following sections, formatted in proper Markdown syntax:
        - Use `-` for main bullet points.
        - Use `  -` (two spaces before `-`) for sub-bullets.
        - Use `####` for section headers (e.g. #### Key Topics)
        - DO NOT USE CODE BLOCKS.

        <Sections>
        1. #### Key Topics: Write a few comprehensive sentences describing the main topics covered in the lecture.
        2. #### Important Concepts: Explain each concept thoroughly, using examples where appropriate. Avoid simple lists; instead, write in a narrative style.
        3. #### Actionable Insights: Provide practical advice or insights shared during the lecture.
        4. #### To-Do: Clearly explain assignments, tutorials, or actionable instructions provided by the instructor.
        5. #### Brief Summary: Conclude with a concise summary of the lecture, covering the most important takeaways.
        </Sections>
        Keep the notes concise, structured, and free of unnecessary symbols or tokens.
        <</SYS>>

        <Lecture>
        <Lecture Transcript>
        {truncate_text(transcript, tokenizer)}
        </Lecture Transcript>

        <Lecture Material>
        {truncate_text(material, tokenizer)}
        </Lecture Material>
        </Lecture>

        Study notes: [/INST]
        """
    else:
        prompt = f"""<s> [INST]
        <<SYS>>
        You are a diligent student creating detailed and comprehensive study notes based on the following lecture. Your goal is to extract key information and explain it in detail.
        Your notes must include the following sections, formatted in proper Markdown syntax:
        - Use `-` for main bullet points.
        - Use `  -` (two spaces before `-`) for sub-bullets.
        - Use `####` for section headers (e.g., #### Key Topics).

        ### Sections:
        1. **Key Topics**: Write a few comprehensive sentences describing the main topics covered in the lecture.
        2. **Important Concepts**: Explain each concept thoroughly, using examples where appropriate. Avoid simple lists; instead, write in a narrative style.
        3. **Actionable Insights**: Provide practical advice or insights shared during the lecture.
        4. **To-Do**: Clearly explain assignments, tutorials, or actionable instructions provided by the instructor.
        5. **Brief Summary**: Conclude with a concise summary of the lecture, covering the most important takeaways.

        Keep the notes concise, structured, and free of unnecessary symbols or tokens.

        <Lecture>
        <Lecture Transcript>
        {truncate_text(transcript, tokenizer)}
        </Lecture Transcript>
        </Lecture>

        Study notes: [/INST]
        """


    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    output = pipe(
        prompt,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.2,
        top_p=0.9,
    )

    if not output:
        raise ValueError("Model did not generate any output.")

    generated_summary = output[0]["generated_text"]
    if "[/INST]" in generated_summary:
        generated_summary = generated_summary.split("[/INST]")[1].strip()

    return generated_summary

