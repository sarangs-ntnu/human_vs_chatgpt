import streamlit as st

# Load model directly
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TextClassificationPipeline)

tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

def score_and_visualize(text):
    prediction = pipe([text])
    f_score = 0
    f_label = ""
    label_0 = prediction[0][0]['label']
    score_0 = prediction[0][0]['score']
    label_1 = prediction[0][1]['label']
    score_1 = prediction[0][1]['score']
    if score_0 > score_1:
        f_score = (round(score_0))*100
        f_label = label_0
    else:
        f_score = (round(score_1))*100
        f_label = label_1
    return f_score, f_label

def main():
    st.title("Human vs ChatGPT Classification Model")

    # Create an input text box
    input_text = st.text_area("Enter your text", "")

    # Create a button to trigger model inference
    if st.button("Analyze"):
        # Perform inference using the loaded model
        score, label = score_and_visualize(input_text)
        st.write("The input text is ", str(score), " ", label , " based.")

if __name__ == "__main__":
    main()