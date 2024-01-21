import os
import random

import openai
import tiktoken

from pathlib import Path
from getpass import getpass

import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, 
)  
import wandb
from wandb.integration.openai import autolog
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document


MODEL_NAME = "gpt-4-1106-preview"
tokenizer = tiktoken.encoding_for_model(MODEL_NAME)

delimiter = "\nNew question:\n" 
with open("question_examples.txt", "r", encoding="utf-8") as file:
    data = file.read()
    real_queries = data.split(delimiter)

with open("system_template.txt", "r") as file:
    system_prompt = file.read()

with open("prompt_template.txt", "r") as file:
    prompt_template = file.read()

def load_documents(data_dir):
    pdf_files = list(map(str, Path(data_dir).glob("*.pdf")))
    documents = []
    for file_path in pdf_files:
        try:
            documents.extend(PyPDFLoader(file_path=file_path, extract_images=True).load())
        except Exception as e:
            print(f"Failed to load {file_path} due to {e}")
    return documents

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def extract_random_chunk(document, max_tokens=1024):
    if isinstance(document, Document):
        tokens = tokenizer.encode(document.page_content)
        if len(tokens) <= max_tokens:
            return document.page_content
    else:
        tokens = tokenizer.encode(document)
        if len(tokens) <= max_tokens:
            return document
    start = random.randint(0, len(tokens) - max_tokens)
    end = start + max_tokens
    return tokenizer.decode(tokens[start:end])

def generate_context_prompt(chunk, n_questions=3):
    questions = '\n'.join(random.sample(real_queries, n_questions))
    user_prompt = prompt_template.format(QUESTIONS=questions, CHUNK=chunk)
    return user_prompt

def generate_questions_and_answers(documents, n_questions=3, n_generations=5):
    questions = []
    for document in documents:
        chunk = extract_random_chunk(document)
        user_prompt = generate_context_prompt(chunk, n_questions)
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        print("Calling OpenAI API...")
        response = completion_with_backoff(
            model=MODEL_NAME,
            messages=messages,
            n = n_generations,
            )
        print("OpenAI API call completed")
        questions.extend([response.choices[i].message.content for i in range(n_generations)])
    return questions

def parse_generation(generation):
    lines = generation.split("\n")
    context = []
    question = []
    answer = []
    flag = None
    
    for line in lines:
        if "CONTEXT:" in line:
            flag = "context"
            line = line.replace("CONTEXT:", "").strip()
        elif "QUESTION:" in line:
            flag = "question"
            line = line.replace("QUESTION:", "").strip()
        elif "ANSWER:" in line:
            flag = "answer"
            line = line.replace("ANSWER:", "").strip()

        if flag == "context":
            context.append(line)
        elif flag == "question":
            question.append(line)
        elif flag == "answer":
            answer.append(line)

    context = "\n".join(context)
    question = "\n".join(question)
    answer = "\n".join(answer)
    return context, question, answer

def get_best_documents(documents):
    non_relevant_documents = [document for document in documents if document.page_content.count('.') < 200]
    top_documents = sorted(non_relevant_documents, key=lambda x: len(tokenizer.encode(x.page_content)),
                           reverse=True)[:int(0.03 * len(non_relevant_documents))]
    return top_documents

if __name__ == "__main__":
    autolog({"project":"amae", "job_type": "generation", "entity": "chamera"})

    if os.getenv("OPENAI_API_KEY") is None:
        os.environ["OPENAI_API_KEY"] = getpass(
            "Paste your OpenAI key from: https://platform.openai.com/account/api-keys\n")
    assert os.getenv("OPENAI_API_KEY", "").startswith(
        "sk-"), "This doesn't look like a valid OpenAI API key"
    print("OpenAI API key configured")

    documents = load_documents("docs")
    parsed_generations = []
    generations = generate_questions_and_answers(get_best_documents(documents), n_questions=3, n_generations=5)
    for generation in generations:
        context, question, answer = parse_generation(generation)
        parsed_generations.append({"context": context, "question": question, "answer": answer})

    df = pd.DataFrame(parsed_generations)
    df.to_csv('generated_examples.csv', index=False)

    wandb.log({"generated_examples": wandb.Table(dataframe=df)})
    wandb.log({"openai_gen_model": MODEL_NAME})

    artifact = wandb.Artifact("generated_examples", type="dataset")
    artifact.add_file("generated_examples.csv")
    wandb.log_artifact(artifact)
    wandb.finish()