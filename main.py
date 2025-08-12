from decouple import config
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.schema import Document
import warnings
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_react_agent,AgentExecutor
from langchain import hub
import time
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

warnings.filterwarnings("ignore", category=FutureWarning)

# === API Key ===
load_dotenv()
SECRET_KEY = os.getenv("OPEN_AI_API_KEY")
print(SECRET_KEY)

#get the api key from the server 

# === Model ===
llm = GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=SECRET_KEY)

@tool(return_direct=True)
def process_question(question: str) -> str:

    """This Function Takes the User's Question which is related to the information about products and business information and Returns the AI's Answer."""
    try:
        # === Check if vector store exists ===
        if os.path.exists("vector_store"):
            retriever_store = FAISS.load_local(
                folder_path="vector_store",
                embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                allow_dangerous_deserialization=True
            )
        else:
            # === Load PDF for Business Information ===
            file_path = "Shah_Bags_Business_Info.pdf"
            loader = PyPDFLoader(file_path)
            all_docs = loader.load()

            # === Load CSV for product information ===
            loaders = CSVLoader(file_path='products.csv', encoding="utf-8")
            result_of_products = loaders.load()

            products_data_list = []
            for row in result_of_products:
                content = row.page_content
                product_link = row.metadata.get("product_link", "")
                products_data_list.append(Document(
                    page_content=content,
                    metadata={"product_link": product_link}
                ))

            # === Split documents ===
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=100
            )
            Business_Data = splitter.split_documents(all_docs)
            chunks = Business_Data + products_data_list

            retriever_store = FAISS.from_documents(
                documents=chunks,
                embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )
            retriever_store.save_local("vector_store")

        # === Retrieval ===
        retriever = retriever_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        retriev = retriever.invoke(question)

        # === Prepare text for summarization ===
        full_text = "\n\n".join(doc.page_content for doc in retriev)

        chat_prompting_summary = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful kind assistant! You are a text compressor. "
                "Summarize the following text {text} based on the question {question} "
                "means keep only the most relevant parts. "
                "If the question is related to product or products information, return the answer with the product_link "
                "while preserving meaning. Keep it short but complete."
            ),
            ("human", "{question}")
        ])

        summary = chat_prompting_summary.invoke({
            "text": full_text,
            "question": question
        })

        summaries = llm.invoke(summary)
        compressed_docs = [Document(page_content=summaries, metadata={"compressed": True})]

        # === Prepare context for final answer ===
        context = " ".join(d.page_content for d in compressed_docs)

        chat_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful kind business assistant of Shah Bags! With the following context: {context}, "
                "create a well-structured, well-organized, well-formatted answer for the question. "
                "If context contains product_link, return it with the answer. "
                "If the context is insufficient, say 'I don't know'. "
                "If someone greets you, say you are a Shah AI Chatbot developed by shah bags."
            ),
            ("human", "{question}")
        ])

        final_prompt = chat_prompt.invoke({
            "context": context,
            "question": question
        })

        # === LLM Response ===
        answer = llm.invoke(final_prompt)

        return str(answer)

    except Exception as e:
        return f"An error occurred: {str(e)}"
    
  #Tool for Calling the customer support   
@tool(return_direct=True)
def customer_service(question: str) -> str:
    """This Function takes the query that is related to the customer service and issues/problems they are facing related to products, orders, payments, refunds in the website ."""
    import time
    try:
        time.sleep(5)
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful kind assistant! you are the decision maker. Always answer in one word only "
                "based on the above question {question} classify the question based on the problem. "
                "If the question is related to payment issue classify it as the high as word. "
                "If the question is not related to payment information, return with low as word. "
                "If the question is insufficient to classify then return with the other as word."
            ),
            ("human", "{question}")
        ])

        answer = prompt | llm
        result = answer.invoke({"question": question})

        if result == "high":
            return """I Deeply Understand Your Issue. Don't worry, we are here for you! Please contact on the button below \n\n
<a href="tel:+919834504652">
  <button style="padding:10px; background:green; color:white; border:none; border-radius:5px;">
    📞 Call Support
  </button>
</a>"""
        elif result == "low":
            return """We understand your issue and will definitely try to solve it. You can reach out to us via WhatsApp for assistance:
<a href="https://wa.me/+919834504652" target="_blank" rel="noopener noreferrer">
  <button style="padding:10px; background:#25D366; color:white; border:none; border-radius:5px;">
    💬 Chat on WhatsApp
  </button>
</a>"""
        else:
            return "Please enter your issue clearly before directly calling customer support with 'customer support request' at the end so that we can arrange a call and message service for you."
    
    except Exception as e:
        # Optional: log the exception here
        return f"Sorry, something went wrong while processing your request: {str(e)}"

#Tool for the customer order tracking    
@tool(return_direct=True)
def order_tracking(question: str) -> str:
    """
    This function is called if question is related to the order status or knowing the order details 
    or if the issue is related to late delivery or delay in the delivery.
    """
    try:
        return """We apologize if you are facing a delay in your order. You can get all the order details below:\n\n
<a href="https://grabaggs.com/order-tracking/" target="_blank" rel="noopener noreferrer">
  <button style="padding:10px; background:#25D366; color:white; border:none; border-radius:5px;">
    Order Details
  </button>
</a>"""
    except Exception as e:
        # Log the error or handle it as needed
        return f"Sorry, something went wrong while processing your request: {str(e)}"
@tool(return_direct=True)
def other_question(question: str) -> str:
    """This function handles questions that don't fit into the predefined categories.It takes a type of question that is npt related to the shah bags and delivery and customer service as well"""
    try:
        return "Hello, I am the chat assistant of Shah Bags, developed by Shah Bags as a company.I am unable to understand your question or the question is not related to this business .Sorry for that but you can reach out to our customer support for further assistance.Just Specify your issue with request for customer support at the end !"
    except Exception as e:
        return f"Sorry, something went wrong while processing your request: {str(e)}"

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question

    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(
        llm=llm,
        prompt=prompt,
        tools=[
            process_question,
            customer_service,
            order_tracking,
            other_question
        ]
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=[process_question, customer_service, order_tracking,other_question],
        verbose=True
    )

    time.sleep(5)

    response = agent_executor.invoke({"input": question})

    return {"answer": response["output"]}

@app.get("/")
async def root():
    return {"message": "Welcome to my chatbot API!"}

@app.head("/")
async def root_head():
    return Response(status_code=200)



