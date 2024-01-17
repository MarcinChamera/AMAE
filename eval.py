"""Evaluate a ConversationalRetrievalChain on a dataset of questions and answers."""
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import wandb
from chain import load_chain, load_vector_store
from config import default_config
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from prompts import load_eval_prompt
from tqdm import tqdm
from getpass import getpass


def load_eval_dataset(config: SimpleNamespace) -> pd.DataFrame:
    """Load a dataset of questions and answers from a Weights & Biases artifact
    Args:
        config (SimpleNamespace): A config object
    Returns:
        pd.DataFrame: A dataframe of questions and answers
    """
    # we will load data from a wandb Table  artifact
    artifact = wandb.use_artifact(config.eval_artifact)
    # download artifact
    artifact_dir = Path(artifact.download())
    # load data
    eval_dataset = pd.read_csv(artifact_dir / "generated_examples.csv")
    return eval_dataset


def generate_answers(
    eval_dataset: pd.DataFrame, qa_chain: ConversationalRetrievalChain
) -> pd.DataFrame:
    """Generate answers for a dataset of questions and answers
    Args:
        eval_dataset (pd.DataFrame): A dataframe of questions and answers
        qa_chain (ConversationalRetrievalChain): A ConversationalRetrievalChain object
    Returns:
        pd.DataFrame: A dataframe of questions, answers, and model answers
    """
    answers = []
    for query in tqdm(eval_dataset["question"], total=len(eval_dataset)):
        result = qa_chain({"question": query, "chat_history": []})
        answers.append(result['answer'])

    eval_dataset["model_answer"] = answers
    eval_dataset.to_csv("eval_with_answers.csv", index=False)
    return eval_dataset


def evaluate_answers(
    eval_dataset: pd.DataFrame, config: SimpleNamespace
) -> pd.DataFrame:
    """Evaluate a dataset of questions, answers, and model answers
    Args:
        eval_dataset (pd.DataFrame): A dataframe of questions, answers, and model answers
        config (SimpleNamespace): A config object
    Returns:
        pd.DataFrame: A dataframe of questions, answers, model answers, and model scores
    """
    eval_prompt = load_eval_prompt()
    llm = ChatOpenAI(
        model_name=config.eval_model,
        temperature=0,
    )
    eval_chain = QAEvalChain.from_llm(llm, prompt=eval_prompt)

    examples = []
    predictions = []
    for i in range(len(eval_dataset)):
        examples.append(
            {
                "query": eval_dataset["question"].iloc[i],
                "answer": eval_dataset["answer"].iloc[i],
            }
        )
        predictions.append(
            {
                "query": eval_dataset["question"].iloc[i],
                "answer": eval_dataset["answer"].iloc[i],
                "result": eval_dataset["model_answer"].iloc[i],
            }
        )
    graded_outputs = eval_chain.evaluate(examples, predictions)
    eval_dataset["model_score"] = [x.get("results", "None") for x in graded_outputs]
    return eval_dataset


def log_results(eval_dataset: pd.DataFrame, run_name: str) -> None:
    """Log evaluation results to a Weights & Biases Artifact
    Args:
        eval_dataset (pd.DataFrame): A dataframe of questions, answers, model answers, and model scores
    """
    model_accuracy = (len(eval_dataset[eval_dataset["model_score"] == "CORRECT"]) +
                      len(eval_dataset[eval_dataset["model_score"] == "GRADE: CORRECT"])) / len(eval_dataset)
    wandb.log({"model_accuracy": model_accuracy})
    eval_dataset.to_csv("eval_results.csv", index=False)
    artifact = wandb.Artifact("eval_results", type="eval_results")
    artifact.add_file("eval_results.csv")
    wandb.log_artifact(artifact)
    eval_dataset["run_name"] = run_name
    wandb.log({"eval_results": wandb.Table(dataframe=eval_dataset)})

# Cost of generating answers:
    # Round 1:
        # Input:
        # - pages of Honda Accord manual
        # - each page had to be a non-content table page
        # - top 5% pages with the most tokens available
        # - resulted in 14 pages
        # - 3 questions per page
        # - 5 generations per page
        # - set of 70 questions total
        # - model generating answers based on synthetic questions: gpt-3.5-turbo
        # - evaluation model: gpt-3.5-turbo model

        # Output:
        # - API calls cost: 0.17$
    # Round 2:
        # Input:
        # - pages of Honda Accord, washing machine, dishwasher, oven, refrigerator, induction hob manuals
        # - each page had to be a non-content table page
        # - top 3% pages with the most tokens available
        # - resulted in 32 pages
        # - 3 questions per page
        # - 5 generations per page
        # - gpt-4-1106-preview model

        # Output:
        # - API calls cost: 0.38$
    
if __name__ == "__main__":
    with wandb.init(project=default_config.project, config=default_config, job_type="eval", entity="chamera") as run:
        if os.getenv("OPENAI_API_KEY") is None:
            os.environ["OPENAI_API_KEY"] = getpass(
                "Paste your OpenAI key from: https://platform.openai.com/account/api-keys\n")
        assert os.getenv("OPENAI_API_KEY", "").startswith(
            "sk-"), "This doesn't look like a valid OpenAI API key"
        eval_dataset = load_eval_dataset(default_config)
        vector_store = load_vector_store(run, os.environ["OPENAI_API_KEY"])
        qa_chain = load_chain(run, vector_store, os.environ["OPENAI_API_KEY"])
        if input("Generate new answers? (y/n) ") == "y":
            eval_dataset = generate_answers(eval_dataset, qa_chain)
        else:
            eval_dataset = pd.read_csv("eval_with_answers.csv")
        eval_dataset = evaluate_answers(eval_dataset, default_config)
        log_results(eval_dataset, run.name)
    