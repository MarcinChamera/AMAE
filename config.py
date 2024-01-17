from types import SimpleNamespace

PROJECT = "amae"
JOB_TYPE = "production"

default_config = SimpleNamespace(
    project=PROJECT,
    entity="chamera",
    job_type=JOB_TYPE,
    vector_store_artifact="chamera/amae/vector_store:latest",
    chat_prompt_artifact="chamera/amae/chat_prompt:latest",
    chat_temperature=0.3,
    max_fallback_retries=1,
    model_name="gpt-3.5-turbo", # model generating answers based on synthetic questions
    eval_model="gpt-3.5-turbo", # model comparing generated answers (based on synthetic questions) with "ideal" answers (synthetic answers from gpt-4)
    eval_artifact="chamera/amae/generated_examples:latest",
)