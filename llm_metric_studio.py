import os

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_google_genai import HarmCategory, HarmBlockThreshold

from ragas import SingleTurnSample
from ragas.metrics import AspectCritic

from config import GEMINI_API_KEY

os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY

# Configuration for the LLM
config = {
    'model': 'models/gemini-2.5-flash-lite',
    'temperature': 0.4,
    'max_tokens': None,
    'top_p': 0.8
}

# Define safety settings for the LLM
safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH
}

#Initialize the LLM and wrap it for use with ragas
evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
    model = config['model'],
    temperature = config['temperature'],
    max_tokens = config['max_tokens'],
    top_p = config['top_p'],
    safety_settings = safety_settings))

#Initialize the embeddings model and wrap it for use with ragas
evaluator_embeddings = LangchainEmbeddingsWrapper(
    GoogleGenerativeAIEmbeddings(
    model = "models/embedding-001",
    task_type= "retrieval_document"))

print("LLM and Embeddings initialized successfully.")

test_data = {
    "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
    #"response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
    "response": "The company grew a lot with no reason.",
}

metric = AspectCritic(name="summary_accuracy", llm=evaluator_llm, definition="Verify if the summary is accurate")
test_data = SingleTurnSample(**test_data)
#print(metric.single_turn_score(test_data, reference="The company reported an 8% growth in Q3 2024, primarily driven by strong sales in the Asian market, attributed to strategic marketing and localized products, with continued growth anticipated in the next quarter."))
print(metric.single_turn_score(test_data))


