import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTableQuestionAnswering
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
#Imports for flask and connection to javascirpt
from flask import Flask, request, jsonify
from flask_cors import CORS


# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load the table embedding model from huggingface models hub
retriever = SentenceTransformer("deepset/all-mpnet-base-v2-table", device=device)


# Read the CSV file
df = pd.read_csv("RawData.csv", encoding = "unicode_escape")
df = df.astype(str)

tables = []

for city, group in df.groupby('Provider City'):
    tables.append(group)

#Converting the table to string
def _preprocess_tables(tables: list):
    processed = []
    # loop through all tables
    for table in tables:
        # convert the table to csv and
        processed_table = "\n".join([table.to_csv(index=False)])
        # add the processed table to processed list
        processed.append(processed_table)
    return processed

# format all the dataframes in the tables list
processed_tables = _preprocess_tables(tables)

pc = Pinecone(api_key='6315210d-75de-4bdd-b3c1-ffcb5ca3b35c')

index_name = "table-qa"

# Check if the table-qa index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# tokenizer = AutoTokenizer.from_pretrained("Zarcend/AFFIRM-FineTune-25")
# model = AutoModelForTableQuestionAnswering.from_pretrained("Zarcend/AFFIRM-FineTune-25")

tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-sqa")
model = AutoModelForTableQuestionAnswering.from_pretrained("google/tapas-base-finetuned-sqa")

def retrieveTable(input):
    query = input
    # generate embedding for the query
    xq = retriever.encode([query]).tolist()
    # query pinecone index to find the table containing answer to the query
    result = index.query(vector=xq, top_k=3, include_metadata=True)
    # Get the ids for indices 0, 1, and 2
    ids = [int(result["matches"][i]["id"]) for i in range(3)]

    # Extract the whole table corresponding to ids 0, 1, and 2
    full_tables = [tables[id] for id in ids]

    # Concatenate the full tables into a single DataFrame
    batch = pd.concat(full_tables, ignore_index=True)
    batch.fillna('', inplace=True)
    batch = batch.sample(frac=1.0, random_state=42, replace=False).reset_index(drop=True).astype(str)
    return batch

def runQuery(input, batch):
    # Encode the table and query
    table = batch
    query = input
    inputs = tokenizer(table=table, queries=query, padding="max_length", return_tensors="pt")

    # Pass the inputs to the model
    outputs = model(**inputs)

    # Interpret the logits to get the answer
    logits = outputs.logits
    logits_agg = outputs.logits_aggregation

    # Get the answer from the model's output
    predicted_answer_coordinates, = tokenizer.convert_logits_to_predictions(inputs, logits.cpu().detach())

    answers = []
    for coordinates in predicted_answer_coordinates:
        cell_values = []
        for coord in coordinates:
            try:
                cell_values.append(batch.iat[coord])
            except IndexError:
                print(f"Index {coord} is out of bounds for the DataFrame with shape {batch.shape}")
                continue
        if cell_values:
            answers.append(", ".join(cell_values))
    return answers
#Connecting to Front end
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.json  # Extract JSON data sent from JS
    input = data['message']
    batch = retrieveTable(input)
    answer = runQuery(input, batch)
    print("2. Answer: ", answer)
    result = {"response": answer}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app
