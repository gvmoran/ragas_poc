import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness,answer_relevancy, context_precision, context_recall
import google.generativeai as genai
from ragas.llms import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from langchain.schema import Generation, LLMResult
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)


class GeminiLLM(BaseRagasLLM):
    def __init__(self, model_name="models/gemini-2.5-flash-lite"):
        self.model_name = model_name
        self.client = genai.GenerativeModel(model_name=model_name)
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            if not isinstance(prompt, str):
                prompt = getattr(prompt, 'to_string', lambda: str(prompt))()

            response = self.client.generate_content(prompt)
            text = getattr(response, 'text', None)

            if not text and hasattr(response, 'candidates') and response.candidates:
                text = response.candidates[0].content.parts[0].text.strip()

            text = text.strip() if text else ""

            generation = Generation(text=text)
            return LLMResult(generations=[[generation]])
        except Exception as e:
            print(f"[ERROR GeminiLLM.generate_text] {e}")
            generation = Generation(text="")
            return LLMResult(generations=[[generation]])

    async def agenerate_text(self, prompt: str, **kwargs) -> str:
        return self.generate_text(prompt, **kwargs)

    def is_finished(self, output: str) -> bool:
        return True
    
class GeminiEmbeddings(BaseRagasEmbeddings):
    def __init__(self, model_name="text-embedding-004"):
        self.model_name = model_name
    
    def embed_documents(self, texts):
        try:

            return [
                genai.embed_content(model=self.model_name,content=text)["embedding"]
                for text in texts
            ]
        except Exception as e:
            print(f"[ERROR GeminiEmbeddings.embed_documents] {e}")
            return [[0.0] * 768 for _ in texts]
    
    def embed_query(self, text):
        try:
            return genai.embed_content(model=self.model_name,content=text)["embedding"]
        except Exception as e:
            print(f"[ERROR GeminiEmbeddings.embed_query] {e}")
            return [0.0] * 768

    
    async def aembed_documents(self, texts):
        return self.embed_documents(texts)
    
    async def aembed_query(self, text):
        return self.embed_query(text)

def main():

    gemini_llm = GeminiLLM(model_name="models/gemini-2.5-flash-lite")
    embeddings = GeminiEmbeddings()

    data = {
        "question": [
            "What is object-oriented programming?"
        ],
        "contexts": [
            [
                "Object-oriented programming (OOP) is a paradigm based on classes and objects that seeks to model the real world.",
                "It allows encapsulation of data and behaviors into objects to better represent real-world entities."
            ]
        ],
        "response": [
            #"It is a way of programming based on objects and classes that represent real-world entities."
            #"It is programming secuentially organized with sentences like goto and if to control the flow."
            #"Object-oriented programming (OOP) is a computer programming model that organizes software design around data, or objects, rather than functions and logic. An object can be defined as a data field that has unique attributes and behavior."
            "Programmin a meeting in an activity that every one can do. People use meetings to organize their work and share ideas."
        ],
        "ground_truth": [
            "Object-oriented programming (OOP) is a programming paradigm based on objects and classes that encapsulate data and behaviors."
        ]
    }


    dataset = Dataset.from_dict(data)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=gemini_llm,
        embeddings=embeddings
    )

    print(results)

if __name__ == "__main__":
    main()