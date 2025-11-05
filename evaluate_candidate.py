import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness,answer_relevancy, context_precision, context_recall
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics import SemanticSimilarity, BleuScore, RougeScore
from ragas.metrics import AspectCritic
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
            "What is object-oriented programming?",
            "Explain the concept of polymorphism.",
            "Explain the difference between stack and heap memory.",
            "Define the term 'DRY' in software development."
        ],
        "response": [
            "It is a way of programming based on objects and classes that represent real-world entities.",
            #ai generated response
            "Polymorphism in object-oriented programming (OOP) allows objects of different classes to be treated as objects of a common superclass, enabling the same interface to behave differently depending on the underlying object.", 
            "Stack memory is the main space of the machine memory where primitives are stored. Heap memory is the secondary memory space where everything else goes, complex objects, large objects and so on and they are subject of garbage collection.",
            #ai generated response
            "DRY stands for 'Don't Repeat Yourself' — a core principle in software development that aims to reduce code duplication by ensuring that every piece of knowledge or logic exists in a single, unambiguous place."

            
        ],
        "ground_truth": [
            "Object-oriented programming (OOP) is a programming paradigm based on objects and classes that encapsulate data and behaviors.",
            "Polymorphism allows objects of different classes to be treated as objects of a common superclass, enabling method overriding.",
            "Stack memory stores local variables and function calls; heap memory is for dynamic allocation. Stack operates in LIFO, heap managed manually or by garbage collection.",
            "DRY (Don't Repeat Yourself) advocates for avoiding code duplication by reusing existing code."
        ]
    }


    dataset = Dataset.from_dict(data)

    #metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    metrics = [
        SemanticSimilarity(), 
        BleuScore(),
        RougeScore(),
        # FactualCorrectness(),  
        # FactualCorrectness(mode='precision'),  
        # FactualCorrectness(mode='recall'),  
        # AspectCritic(name="accuracy", definition="Measures how accurate the response is with respect to the ground truth."),
        # AspectCritic(name="completeness", definition="Measures how complete the response is in covering all aspects of the ground truth."),
        # AspectCritic(name="clarity", definition="Measures how clear and understandable the response is."),
        # AspectCritic(name="conciseness", definition="Measures how concise and to the point the response is."),
        # AspectCritic(name="relevance", definition="Measures how relevant the response is to the question asked."),
        # AspectCritic(name="coherence", definition="Measures how logically structured and coherent the response is."),
        # AspectCritic(name="ai_likeness", definition="Measures how much the response resembles text generated by an AI model.")
    ]
    
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=gemini_llm,
        embeddings=embeddings
    )

    print("Evaluation Results:")
    df = results.to_pandas().T
    df.to_csv("gemini_evaluation_results.csv")
    print(df)

    

if __name__ == "__main__":
    main()