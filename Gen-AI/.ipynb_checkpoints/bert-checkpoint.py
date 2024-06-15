import torch
from transformers import BertForQuestionAnswering, BertTokenizer

# Load the pre-trained model and tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the path to the file containing the context
context_file_path = 'clamities.txt'

# Read the context from the file
with open(context_file_path, 'r', encoding='utf-8') as file:
    context = file.read()

# Prompt the user to input a question
question = input("Please enter your question: ")

# Tokenize the input
inputs = tokenizer(question, context, return_tensors='pt')

# Perform the forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Get the start and end indices for the answer
start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits)

# Get the tokens for the answer
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
answer_tokens = tokens[start_index:end_index+1]

# Convert tokens to string
answer = tokenizer.convert_tokens_to_string(answer_tokens)

# Display the answer
print("Answer:", answer)
