from transformers import pipeline

# Load the pre-trained sentiment-analysis model
classifier = pipeline('sentiment-analysis')

# Test the model with sample texts
text = ["I love Hugging Face!", "This is the worst experience ever."]

# Get predictions
results = classifier(text)

# Display the results
for idx, result in enumerate(results):
    print(f"Text: {text[idx]}")
    print(f"Label: {result['label']}, Confidence: {result['score']:.4f}")
