from .rag.rag_engine import ask_rag
from .llm import ask_llm


def route_question(question):

    question = question.lower()

    # prediction related
    if "risk" in question or "predict" in question:
        return "prediction"

    # report analysis
    elif "report" in question or "summary" in question:
        return "summarize"

    # knowledge questions
    else:
        return "rag"


def run_agent(question):

    task = route_question(question)

    if task == "rag":
        return ask_rag(question)

    if task == "summarize":
        return "Please paste the medical report for summarization."

    if task == "prediction":
        return "Please go to prediction page and enter patient data."