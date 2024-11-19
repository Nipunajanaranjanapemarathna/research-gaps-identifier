import os
from flask import Flask, request, render_template
import fitz  # PyMuPDF
from transformers import BartForConditionalGeneration, BartTokenizer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)

# Load the summarization model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the ChatGroq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.1-70b-versatile"
)

def extract_text_by_keywords(pdf_path, keywords):
    extracted_text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text = page.get_text("text")
            if any(keyword.lower() in text.lower() for keyword in keywords):
                extracted_text += text + "\n"
    return extracted_text.strip()

def summarize_text(text, max_length=130, min_length=30):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    summaries = ["", "", "", ""]
    llm_response = ""

    if request.method == 'POST':
        files = request.files.getlist('pdf_files')
        
        if len(files) != 3:
            return render_template('index.html', error="Please upload exactly 3 PDF files.")

        keywords = ["Conclusions", "DISCUSSION AND CONCLUSION", "LIMITATION AND RECOMMENDATION", "CONCLUSION AND DISCUSSION""Conclusions",
                "Conclusion",
                "DISCUSSION AND CONCLUSION",
                "LIMITATION AND RECOMMENDATION",
                "CONCLUSION AND DISCUSSION",
                "Discussion",
                "DISCUSSION",
                "Findings",
                "Limitations",
                "LIMITATION",
                "Recommendations",
                "RECOMMENDATIONS",
                "Future Work",
                "FUTURE WORK",
                "Future Directions",
                "FUTURE DIRECTIONS",
                "Implications",
                "IMPLICATIONS",
                "Summary",
                "SUMMARY",
                "Concluding Remarks",
                "CONCLUDING REMARKS",
                "Final Thoughts",
                "FINAL THOUGHTS"]
        
        for i, file in enumerate(files):
            if file:
                pdf_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(pdf_path)
                
                extracted_text = extract_text_by_keywords(pdf_path, keywords)
                if extracted_text:
                    summaries[i] = summarize_text(extracted_text)

        # Prepare prompt for the LLM
        description_1 = summaries[0]
        description_2 = summaries[1]
        description_3 = summaries[2]

        prompt_extract = PromptTemplate(
            input_variables=["description_1", "description_2", "description_3"],
            template="Give me the best description about the research gaps using given details and give me a only one paragraph: \"{description_1}, {description_2}, {description_3}\""
        )

        formatted_prompt = prompt_extract.format(
            description_1=description_1,
            description_2=description_2,
            description_3=description_3,
        )

        # Invoke the LLM
        response = llm.invoke(formatted_prompt)
        llm_response = response.content

    return render_template('index.html', summary1=summaries[0], summary2=summaries[1], summary3=summaries[2], llm_response=llm_response)

if __name__ == '__main__':
    app.run(debug=True)



