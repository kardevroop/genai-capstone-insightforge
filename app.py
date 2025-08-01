import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from datetime import datetime
from langchain.evaluation.qa import QAEvalChain
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import SerpAPIWrapper
from langchain.document_loaders import TextLoader
from langchain.agents.agent import AgentExecutor  
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory

# from dotenv import load_dotenv
import streamlit as st

# load_dotenv()

OPENAI_API_KEY = st.secrets["API_KEYS"]["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

SERPAPI_API_KEY = st.secrets["API_KEYS"]["SERPAPI_API_KEY"]
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@st.cache_data
def load_data():
    return pd.read_csv("sales_data.csv")

data = load_data()

#@st.cache_data
#def load_advance_summary():
#    with open("advanced_data_summary.txt", "r") as f:
#        return f.read()

@st.cache_data
def load_advance_summary():
    loader = TextLoader("advanced_data_summary.txt")
    return loader.load()

@st.cache_data
def load_capstone_pdfs():
    pdf_folder = 'Datasets/Capstone_pdfs'
    documents = []
    if os.path.exists(pdf_folder):
       for file in os.listdir(pdf_folder):
           if file.endswith('.pdf'):
               loader = PyPDFLoader(os.path.join(pdf_folder, file))
               documents.extend(loader.load())
    else:
       print(f"The folder '{pdf_folder}' does not exist.")
    return documents

def plot_product_category_sales():
    product_cat_sales = data.groupby('Product')['Sales'].sum().sort_values(ascending=False)
    # plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))
    product_cat_sales.plot(kind='bar', ax=ax)
    ax.set_title('Sales Distribution by Product')
    ax.set_xlabel('Product')
    ax.set_ylabel('Total Sales')
    # ax.set_xticks(rotation=45)
    ax.tick_params("x", rotation=45)
    plt.tight_layout()

    return fig

def plot_sales_trend():
    fig, ax = plt.subplots(figsize=(10, 6))
    data.groupby('Date')['Sales'].sum().plot(ax=ax)
    ax.set_title('Daily Sales Trend')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Sales')

    return fig

def plot_total_sales_by_region():
    region_sales = data.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    # plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))
    region_sales.plot(kind='bar', color='red', ax=ax)
    ax.set_title('Total Sales by Region')
    ax.set_xlabel('Region')
    ax.set_ylabel('Total Sales')
    # plt.xticks(rotation=45)
    plt.tight_layout()

    return fig
    
def plot_mean_satisfaction_by_region():
    region_satisfaction = data.groupby('Region')['Customer_Satisfaction'].mean().sort_values()
    # plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))
    region_satisfaction.plot(kind='bar', color='orange', ax=ax)
    ax.set_title('Average Customer Satisfaction by Region')
    ax.set_xlabel('Region')
    ax.set_ylabel('Avg. Satisfaction')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
    
def plot_mean_sales_by_region_and_product():
    product_sales = data.groupby('Product')['Sales'].mean()
    fig, ax = plt.subplots()
    product_sales.plot(kind='bar', color='mediumseagreen', ax=ax)
    ax.set_title('Average Sales per Product')
    ax.set_xlabel('Product')
    ax.set_ylabel('Total Sales')
    # ax.set_xticks(rotation=0)
    plt.tight_layout()
    return fig
    
def plot_mean_satisfaction_by_gender():
    gender_satisfaction = data.groupby('Customer_Gender')['Customer_Satisfaction'].mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    gender_satisfaction.plot(kind='bar', color='purple', ax=ax)
    ax.set_title('Avg. Customer Satisfaction by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Avg. Satisfaction')
    plt.tight_layout()
    return fig
    
def plot_mean_satisfaction_by_group():
    df = data.copy(deep=True)
    
    bins = [18, 25, 35, 45, 55, 65]
    labels = ['18–24', '25–34', '35–44', '45–54', '55–64']
    df['Age_Group'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels)
    age_group_satisfaction = df.groupby('Age_Group')['Customer_Satisfaction'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    age_group_satisfaction.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['pink', 'lightblue', 'yellow', 'lightgreen', 'orange', 'teal'], ax=ax)
    
    ax.set_title('Avg. Customer Satisfaction by Age Group')
    # plt.ylabel('')
    plt.tight_layout()
    return fig

def plot_total_sales_by_age_group():
    df = data.copy(deep=True)
    
    bins = [18, 25, 35, 45, 55, 65]
    labels = ['18–24', '25–34', '35–44', '45–54', '55–64']
    df['Age_Group'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels)

    sales_by_group = df.groupby('Age_Group')['Sales'].sum()
    
    fig, ax = plt.subplots()
    sales_by_group.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['red', 'blue', 'yellow', 'lightgreen', 'orange', 'violet'], ax=ax)
    ax.set_title('Sales Distribution by Age Group')
    # plt.ylabel('')
    plt.tight_layout()
    return fig

def plot_total_sales_by_region_and_gender():
    sales_by_region_gender = data.groupby(['Region', 'Customer_Gender'])['Sales'].sum().unstack(fill_value=0)
    fig, ax = plt.subplots()
    sales_by_region_gender.plot(kind='bar', color=['skyblue', 'lightpink'], ax=ax)
    ax.set_title('Total Sales by Region and Gender')
    ax.set_xlabel('Region')
    ax.set_ylabel('Total Sales')
    # ax.set_xticks(rotation=0)
    ax.legend(title='Gender')
    plt.tight_layout()
    return fig

class ModelMonitor:
    def __init__(self):
        self.logs = []

    def log_interaction(self, query, execution_time):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "execution_time": execution_time
        })

    def get_average_execution_time(self):
        if not self.logs:
            return 0
        return sum(log["execution_time"] for log in self.logs) / len(self.logs)

def evaluate_model(qa_pairs):
    llm = ChatOpenAI(model_name="gpt-4o-mini")
    evaluator = QAEvalChain.from_llm(llm=llm)

    examples = [{"query": qa["question"], "answer": qa["answer"]} for qa in qa_pairs]
    predictions = [{"result": agent_chain.run(input=qa["question"])} for qa in qa_pairs]

    eval_results = []

    graded = evaluator.evaluate(examples, predictions)

    for i, result in enumerate(graded):
        eval_results.append({
            "question": examples[i]["query"],
            "predicted": predictions[i]["result"],
            "actual": examples[i]["answer"],
            "correct": result.get("results", "INCORRECT").find("INCORRECT") == -1 # results vary between: "results":"CORRECT" OR "results":"GRADE: CORRECT" so we just look for the fail condition
        })
    return eval_results

def setup_qa_chain():
    summary_documents = load_advance_summary()
    pdf_documents = load_capstone_pdfs()
    rag_documents = summary_documents + pdf_documents

    db = FAISS.from_documents(rag_documents, OpenAIEmbeddings())

    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True, output_key="answer")

    # return RetrievalQA.from_chain_type(
    #     llm=llm, 
    #     retriever=db.as_retriever(),
    #     return_source_documents=True
    # )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key="answer"  # 👈 this tells memory what to track
    )

qa_chain = setup_qa_chain()

def setup_agent_chain(qa_chain):

    def rag_tool_with_sources():
        def run_with_sources(question: str) -> dict:
            result = qa_chain.invoke({"question": question})
            answer = result.get("answer", "")
            source_docs = result.get("source_documents", [])
            sources = [doc.metadata.get("source", "Unknown") for doc in source_docs]

            return {"answer": answer, "sources": sources}

        return Tool(
            name="AdvancedSalesRAG",
            func=run_with_sources,
            description="Useful for answering advanced sales questions based on internal company data. Returns answer and sources."
        )

    def wikipedia_tool_with_sources():
        wiki_api = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
        wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)
        
        def run_with_sources(question: str) -> dict:
            content = wiki_tool.run(question) 
            
            return {"answer": content, "sources": ["Wikipedia query with: " + question]}

        return Tool(
            name="WikipediaSearch",
            func=run_with_sources,
            description="Useful for answering general knowledge questions using Wikipedia. Returns answer and sources."
        )

    def serp_tool_with_sources():
        serp_tool = SerpAPIWrapper()
        
        def run_with_sources(question: str) -> dict:
            content = serp_tool.run(question) 
            
            return {"answer": content, "sources": ["Google query with: " + question]}

        return Tool(
            name = "SerpAPI",
            func = run_with_sources,
            description = "Use this to get the recent or Google web results"
        )

    rag_tool = rag_tool_with_sources() 
    wiki_search_tool = wikipedia_tool_with_sources()
    serp_search_tool = serp_tool_with_sources()

    tools = [rag_tool, wiki_search_tool, serp_search_tool]

    return initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

agent_chain = setup_agent_chain(qa_chain)

def query_with_qa_chain(question: str) -> str:
    result = qa_chain.invoke({"question": question})
    
    answer = result['answer']
    source_docs = result.get('source_documents', [])
    
    sources = [
        doc.metadata.get('source', 'Unknown').split("/")[-1]
        for doc in source_docs
    ]
    
    return f"**Answer:**\n{answer}\n\n**Sources:**\n" + " | ".join(set(sources))

def query_with_agent(question: str) -> str:
    result = agent_chain.invoke(input = question)
    answer = result.get("output", "")
    
    return f"**Answer:**\n{answer}"

@st.cache_resource
def load_and_cache_model_monitor():
    return ModelMonitor()



advanced_summary = load_advance_summary()
model_monitor = load_and_cache_model_monitor()



st.title("InsightForge: Business Intelligence Assistant 🤖 :bulb:")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "AI Assistant", "Model Performance"])


if page == "Home":
    st.write("Welcome to InsightForge, your AI-powered Business Intelligence Assistant.")
    st.write("Use the sidebar to navigate through different sections of the application.")

elif page == "Data Analysis":
    st.header("Data Analysis")

    st.subheader("Sales Summary")
    st.write(advanced_summary[0].page_content.strip())

    st.subheader("Sales Distribution by Product Category")
    fig_category = plot_product_category_sales()
    st.pyplot(fig_category)

    st.subheader("Total Sales by Region")
    fig_trend = plot_total_sales_by_region()
    st.pyplot(fig_trend)

    st.subheader("Customer Satisfaction by Region")
    fig_trend = plot_mean_satisfaction_by_region()
    st.pyplot(fig_trend)

    st.subheader("Avg. Sales per Product")
    fig_trend = plot_mean_sales_by_region_and_product()
    st.pyplot(fig_trend)

    st.subheader("Customer Satisfaction by Gender")
    fig_trend = plot_mean_satisfaction_by_gender()
    st.pyplot(fig_trend)

    st.subheader("Customer Satisfaction by Age Group")
    fig_trend = plot_mean_satisfaction_by_group()
    st.pyplot(fig_trend)

    st.subheader("Sales Distribution by Age Group")
    fig_trend = plot_total_sales_by_age_group()
    st.pyplot(fig_trend)

    st.subheader("Total Sales by Region and Gender")
    fig_trend = plot_total_sales_by_region_and_gender()
    st.pyplot(fig_trend)


elif page == "AI Assistant":
    st.header("AI Sales Analyst")

    ai_mode = st.radio("Choose AI Mode:", ["Standard RAG", "RAG with Tools"])

    user_input = st.text_input("Ask a question about the sales data:")
    if user_input:
        start_time = datetime.now()

        if ai_mode == "Standard RAG":
            response = query_with_qa_chain(user_input)
        else:
            response = query_with_agent(user_input)

        st.write(response)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        model_monitor.log_interaction(user_input, execution_time)
        st.write(f"Execution time: {execution_time:.2f} seconds")

elif page == "Model Performance":
    st.header("Model Performance")

    st.subheader("Model Evaluation")
    if st.button("Run Model Evaluation"):
        qa_pairs = [
            {
                "question": "What is our total sales amount?",
                "answer": f"The total sales amount is ${data['Sales'].sum():,.2f}."
            },
            {
                "question": "Which product has the highest sales?",
                "answer": f"The product with the highest sales is {data.groupby('Product')['Sales'].sum().idxmax()}."
            },
            {
                "question": "What is our average customer satisfaction score?",
                "answer": f"The average customer satisfaction score is {data['Customer_Satisfaction'].mean():.2f}."
            }
        ]

        eval_results = evaluate_model(qa_pairs)
        
        for result in eval_results:
            st.write(f"Question: {result['question']}")
            st.write(f"Predicted: {result['predicted']}")
            st.write(f"Actual: {result['actual']}")
            st.write(f"Correct: {result['correct']}")
            st.write("---")

        accuracy = sum([1 for r in eval_results if r['correct']]) / len(eval_results)

        st.write(f"Model Accuracy: {accuracy:.2%}")

    st.subheader("Execution Time Monitoring")

    # Extract execution times
    execution_times = [log['execution_time'] for log in model_monitor.logs]

    # Bar positions (e.g. 0, 1, 2, ...)
    positions = list(range(len(execution_times)))

    # Create the bar chart
    fig, ax = plt.subplots()
    ax.bar(positions, execution_times, color='skyblue')

    # Set x-ticks to a single category
    ax.set_xticks([sum(positions) / len(positions)])  # center label
    ax.set_xticklabels([''])

    timestamps = [log['timestamp'] for log in model_monitor.logs]
    for i, (pos, time) in enumerate(zip(positions, execution_times)):
        ax.text(pos, time + 0.1, f"{time:.2f}s", ha='center', fontsize=8)

    # Labels
    ax.set_title('Model Execution Times')
    ax.set_ylabel('Execution Time (seconds)')

    st.pyplot(fig)

    avg_execution_time = model_monitor.get_average_execution_time()
    st.write(f"Average Execution Time: {avg_execution_time:.2f} seconds")


# To run this code use the following command in TERMINAL
# streamlit run stramlit_app.py