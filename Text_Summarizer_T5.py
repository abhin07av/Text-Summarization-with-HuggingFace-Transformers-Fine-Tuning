### Import Tools
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
tokenizer = T5Tokenizer.from_pretrained("t5-small")

### Load Datasets
# Load dataset (example, adjust path as needed)
train_data = pd.read_csv("/kaggle/input/samsum-dataset-text-summarization/samsum-train.csv")
validation_data = pd.read_csv("/kaggle/input/samsum-dataset-text-summarization/samsum-validation.csv")

# Display a sample
train_data.head()

### Data Preprocessing
# Clean the text by removing unwanted characters
import re

def clean_text(text):
    text = re.sub(r'\r\n', ' ', text)  # Remove carriage returns and line breaks
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'<.*?>', '', text)  # Remove any XML tags
    text = text.strip().lower()  # Strip and convert to lower case
    return text

# Apply cleaning to dialogue and summary columns
train_data['dialogue'] = train_data['dialogue'].apply(clean_text)
train_data['summary'] = train_data['summary'].apply(clean_text)

validation_data['dialogue'] = validation_data['dialogue'].apply(clean_text)
validation_data['summary'] = validation_data['summary'].apply(clean_text)


# Display a sample after cleaning
train_data


input_max_len = max(len(tokenizer.encode(text)) for text in train_data['dialogue'])
output_max_len = max(len(tokenizer.encode(text)) for text in train_data['summary'])

input_max_len, output_max_len

# Preprocessing function for tokenization
def preprocess_function(examples):
    # Tokenize the dialogue and summary
    inputs = tokenizer(examples["dialogue"], padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(examples["summary"], padding="max_length", truncation=True, max_length=150)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Apply the preprocessing
train_dataset = train_data.apply(preprocess_function, axis=1)
val_dataset = validation_data.apply(preprocess_function, axis=1)

### Fine Tuning Model
# Model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory for checkpoints
    num_train_epochs=6,              # number of training epochs
    per_device_train_batch_size=8,   # batch size per device during training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir="./logs",            # directory for storing logs
    logging_steps=50,                # how often to log training info
    save_steps=500,                  # how often to save a model checkpoint
    eval_steps=50,                   # how often to run evaluation
    evaluation_strategy="epoch",     # Ensure evaluation happens every `epoch`
)

# Trainer setup
trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

### Save and load model
# Save the fine-tuned model
model.save_pretrained("./saved_summary_model")
tokenizer.save_pretrained("./saved_summary_model")

# Load the saved model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

### Summarization System
# Ensure the model is on the correct device (GPU if available)
device = model.device  # Get the device the model is on

def summarize_dialogue(dialogue):
    dialogue = clean_text(dialogue)  # Assuming clean_text is defined
    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    # Move input tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate summary
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=150,  
        num_beams=4, 
        early_stopping=True
    )
    
    # Decode the generated summary
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

#### Test with a sample input
sample_dialogue = """
Violet: Hey Claire! I was reading an article about Austin and thought you might find it interesting! 
Violet: It's about the current trends in urban development and how cities are planning for the future.
Violet: Here, let me share the link: <file_other>
Claire: Oh wow, that sounds like an insightful read. But I've actually already read that one last week. 
Claire: It was really interesting though, especially the part about sustainable architecture in cities. 
Claire: You know, I've been following these urban planning discussions for a while now.
Violet: Oh, I didn’t know that! Well, I’ll look for something else then, maybe something about eco-friendly cities or tech innovations.
Claire: That would be awesome! Let me know if you find something cool.
Violet: Sure, I’ll keep you posted. Thanks for the feedback!
"""

summary = summarize_dialogue(sample_dialogue)
print("Summary:", summary)

# Test with a dialogue on a different topic
sample_dialogue = """
John: Hey Sarah, have you seen the latest tech gadget reviews? I found this new smartwatch that's supposed to have amazing health tracking features.
John: It tracks heart rate, blood oxygen levels, sleep patterns, and even stress levels! It sounds like something right up your alley. 
Sarah: That sounds really interesting! But I’ve been trying to cut down on tech distractions. I’ve heard these devices can be really overwhelming sometimes.
Sarah: I do think it’s cool that they can track so many health metrics though. I’m curious how accurate they really are.
John: Yeah, me too! There are also some new smartphones coming out with even better cameras and longer battery life. The new flagship model from XYZ brand has some insane specs.
Sarah: Ooh, I haven’t kept up with phones recently, but I’ve heard the camera quality is getting ridiculously good. It’s almost like a professional camera in your pocket now!
Sarah: Still, I feel like I’m fine with my current phone for now. I don’t really feel the need to upgrade unless something really groundbreaking comes out.
John: Totally understand that. It’s the same with me. But I think the battery life improvements are enough to make me consider it. I hate running out of battery when I’m out and about.
Sarah: That’s fair! I’m always worried about battery life too. Honestly, I think phones should last at least two full days on a single charge by now.
John: I agree! It’s so annoying when your phone dies in the middle of the day. I wonder if we’ll ever get to a point where we don’t have to charge our phones every day.
Sarah: That would be amazing! I think as tech improves, battery tech might also catch up. Let’s hope the next generation of phones can last longer!
"""

summary = summarize_dialogue(sample_dialogue)
print("Summary:", summary)

# Test with a dialogue on a current news topic
sample_dialogue = """
Reporter: In today's news, the latest climate change report reveals alarming global temperature rises. According to the Intergovernmental Panel on Climate Change (IPCC), the Earth’s temperature is on track to rise by 1.5°C within the next two decades.
Reporter: This is expected to lead to more frequent and severe heatwaves, flooding, and extreme weather events. Coastal cities are at particular risk due to rising sea levels.
Expert: The report emphasizes that immediate action is needed to prevent catastrophic consequences. We need to significantly reduce carbon emissions and transition to renewable energy sources.
Expert: If global temperatures increase by more than 1.5°C, we could face irreversible damage to ecosystems, agriculture, and water supply. It will have a devastating impact on biodiversity as well.
Reporter: The IPCC also stresses the importance of individual action. Governments must set stronger policies, but individuals can help by reducing waste, conserving water, and supporting green initiatives.
Expert: It's not just about the big changes; small actions like using public transportation, reducing meat consumption, and recycling can collectively make a significant difference.
Reporter: With the next UN Climate Summit coming up next month, world leaders will need to prioritize climate action. The stakes have never been higher for our planet’s future.
"""

summary = summarize_dialogue(sample_dialogue)
print("Summary:", summary)
