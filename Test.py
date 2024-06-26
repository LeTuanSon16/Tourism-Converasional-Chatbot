from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from dotenv import load_dotenv,find_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

import os
vector_db_path = "vectorbase/db_faiss"
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
def creat_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt


# Tao simple chain
def create_qa_chain(prompt, docsearch):
    llm_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_type="similarity_score_threshold",
                                        search_kwargs = {"k":4,"score_threshold": 0.5,},
                                        max_tokens_limit=2048
                                        ),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain

# Read tu VectorDB
def read_vectors_db():
    # Embeding
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    docsearch = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    return docsearch


# Bat dau thu nghiem
def qabot(messages):
    db = read_vectors_db()

    question = messages[-1]["content"]

#Tao Prompt
    template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi bằng tiếng Việt.
     Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n

    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
    prompt = creat_prompt(template)

    llm_chain =create_qa_chain(prompt, db)

# Chay cai chain

    response = llm_chain.invoke({"query": question})
    return response['result']


