import streamlit as st
import torch

# Load and prepare the model
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streamlit interface
st.title("Arabic Claim Fact-Checking")
input_claim = st.text_input("Enter an Arabic claim:")
st.write("Example claims:")
st.write("1. الشمس تدور حول الأرض.")
st.write("2. الأرض كروية الشكل.")

if st.button("Check Fact"):
    if input_claim:
        try:
            # Process the input text
            inputs = tokenizer(input_claim, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            label = {0: "True", 1: "False", 2: "Misleading"}[prediction]
            st.write(f"The claim is **{label}**")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a claim to check.")