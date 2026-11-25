"""Minimal document question-answering CLI using the fine-tuned Lab 10 model."""
import argparse
from pathlib import Path
import torch
from qa_model import MODEL_DIR, MAX_LENGTH, STRIDE, get_tokenizer, predict_answers
from transformers import AutoModelForQuestionAnswering

DEMO_CONTEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. "
    "It was completed in 1889 and was designed by Gustave Eiffel's company. "
    "The tower is 330 metres tall."
)


def answer_question(question, context, model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Fine-tuned QA model was not found: {model_dir}. Run qa_model.py --run-all first.")
    tokenizer = get_tokenizer()
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir, local_files_only=True)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    item = {"id": "document", "question": question, "context": context, "answers": []}
    result = predict_answers(model, [item], tokenizer, MAX_LENGTH, STRIDE)[0]
    return result["text"]


def main():
    parser = argparse.ArgumentParser(description="Ask an extractive question about a document")
    parser.add_argument("--question", default="When was the Eiffel Tower completed?")
    parser.add_argument("--document", help="UTF-8 text file containing the document")
    parser.add_argument("--context", help="Pass a short context directly")
    args = parser.parse_args()
    if args.document:
        context = Path(args.document).read_text(encoding="utf-8")
    else:
        context = args.context or DEMO_CONTEXT
    print("Question:", args.question)
    print("Answer:", answer_question(args.question, context))


if __name__ == "__main__":
    main()
