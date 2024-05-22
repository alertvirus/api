import os
import io
import sqlite3
import uuid
from flask import Flask, request, jsonify, send_file, redirect
from flask_cors import CORS, cross_origin
import tempfile
from werkzeug.utils import secure_filename

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from repository import repository

from typing_extensions import TypedDict
from typing import List


from langchain.schema import Document
from langgraph.graph import END, StateGraph



#### Retrieval Grader

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

local_llm = "llama3"
os.environ["TAVILY_API_KEY"] = ""
# LLM
retrieval_grader_llm = ChatOllama(model=local_llm, format="json", temperature=0)

retrieval_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

retrieval_grader = retrieval_grader_prompt | retrieval_grader_llm | JsonOutputParser()

### Generate

from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

llm = ChatOllama(model=local_llm, temperature=0)

# Chain
rag_chain = prompt | llm | StrOutputParser()


### Hallucination Grader
hallucination_grade_llm = ChatOllama(model=local_llm, format="json", temperature=0)
# Prompt
hallucination_grader_prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

hallucination_grader = hallucination_grader_prompt | hallucination_grade_llm | JsonOutputParser()


# Prompt
answer_grader_llm = ChatOllama(model=local_llm, format="json", temperature=0)
answer_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)

answer_grader = answer_grader_prompt | answer_grader_llm | JsonOutputParser()


### Router

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

question_router_llm = ChatOllama(model=local_llm, format="json", temperature=0)
question_router_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
    prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
    in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

question_router = question_router_prompt | question_router_llm | JsonOutputParser()



### Search

from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3)



# Test
from pprint import pprint

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

vectorDb_Directory = ""

@app.route("/")
def home():
    return "Hello World"

@cross_origin()
@app.route("/api/upload", methods=["POST"])
def upload_file():
    # Check if file is present
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Get the file object
    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    filename = uploaded_file.filename
    print(filename)
    # Secure filename
    filename = secure_filename(uploaded_file.filename)

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    # Save the uploaded file content to the temporary file
        uploaded_file.save(temp_file.name)
    file_path = temp_file.name

    vectorDbname = add_file_to_db(filename)
    print(vectorDbname)
    
    process_pdf(file_path, vectorDbname)

    return "Files uploaded to Azure Blob Storage"

@cross_origin()
@app.route('/query_file', methods=['POST'])
def query_file():
    try:
        query = request.json.get('query')
        documentId = request.json.get('documentId')
        query_persist_directory = request.json.get('vectordbName')
        #proompt = request.json.get('proompt')
        
        persist_directory = f"./vectordb/{query_persist_directory}"
        #save query to db
        add_chats(query, 'user', query, documentId)

        result  = generate_answer(query)
        add_chats(query, 'assistant', result, documentId)
        
        # Return docsearch results as JSON response
        return jsonify({'content': result, 'role':'assistant', 'title':query})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


#TODO REPLACE SQL LITE STUFF START
@app.route('/get_data', methods=['GET'])
def get_data():
    conn = sqlite3.connect('documentchat.db')
    cursor = conn.cursor()
    args = request.args
    documentId = args.get('documentId')
    cursor.execute("SELECT * FROM Chats where documentId =" + documentId)
    rows = cursor.fetchall()
    data = []
    for row in rows:
        item = {
            'title': row[1],
            'role': row[2],
            'content': row[3]
        }
        data.append(item)
    conn.close()
    return jsonify(data)

@app.route('/get_documents', methods=['GET'])
def get_documents():
    conn = sqlite3.connect('documentchat.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Documents")
    rows = cursor.fetchall()
    data = []
    for row in rows:
        item = {
            'id': row[0],
            'name': row[1],
            'vectordbName': row[2]
        }
        data.append(item)
    conn.close()
    return jsonify(data)

@app.route('/delete', methods=['POST'])
def delete():
    data = request.get_json()
    if 'table_Name' in data and 'condition' in data:
        table_name = data['table_Name']
        condition = data['condition']

        # Call the delete_rows function with the provided data
        delete_rows(table_name, condition)

        return jsonify({'message': f"Rows deleted from {table_name} where {condition}"})
    else:
        return jsonify({'error': 'Invalid data format or missing data'}, 400)


@app.route('/delete_file')
def delete_file():
    try:
        args = request.args
        documentId = args.get('documentId')

        conn = sqlite3.connect('documentchat.db')
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM Documents where id =" + documentId)
        rows = cursor.fetchall()
        docName = ''
        vecDbName = ''
        for row in rows:
            docName = row[1]
            vecDbName = row[2]
        conn.close()

        #DELETE file
        #delete_blob(docName)
        #Delete vector db
        #delete_folder(vecDbName)
        #delete sql lite
        condition = 'id = ' + documentId
        delete_rows('Documents', condition)
        return redirect('/get_documents')

        
    except Exception as e:
        return str(e)

# Function to delete rows based on a condition
def delete_rows(table_name, condition):
    conn = sqlite3.connect('documentchat.db')
    cursor = conn.cursor()

    # Execute the DELETE statement
    delete_query = f"DELETE FROM {table_name} WHERE {condition}"
    print(delete_query)
    cursor.execute(delete_query)

    # Commit the changes to the database
    conn.commit()

    cursor.close()
    conn.close()


def add_chats(title, role, content, documentId):
    conn = sqlite3.connect('documentchat.db')
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS Chats (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    role TEXT, 
                    content TEXT, 
                    documentId INTEGER
                )''')
    
    cursor.execute("INSERT INTO Chats (title, role, content, documentId) VALUES (?, ?, ?, ?)", (title, role, content, documentId))
    conn.commit()
    conn.close()
    return 'Data added successfully', 201
#TODO REPLACE SQL LITE STUFF END 


def add_file_to_db(fileName):
    conn = sqlite3.connect('documentchat.db')
    cursor = conn.cursor()
    guid = uuid.uuid4()
    uuid_string = str(guid)
    cursor.execute('''CREATE TABLE IF NOT EXISTS Documents (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    vectordbName TEXT
                )''')
    
    cursor.execute("INSERT INTO Documents (name, vectordbName) VALUES (?, ?)", (fileName,uuid_string))
    conn.commit()
    conn.close()
    return uuid_string

def process_pdf(file_path,vectorDbname):
    # Load and process the text files
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
    )

    doc_splits = text_splitter.split_documents(pages)

   # Embed and store the texts
    persist_directory = f"./vectordb/{vectorDbname}"

    vectordb = Chroma.from_documents(documents=doc_splits,
                                 embedding=GPT4AllEmbeddings(),
                                 persist_directory=persist_directory,
                                 collection_name="rag-chroma")
    vectordb.persist()
    vectordb = None

def getWorkflow():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("websearch", web_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae

    # Build graph
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges("generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )
    return workflow

def generate_answer(query):
    workflow = getWorkflow()
    app = workflow.compile()
    inputs = {"question": query}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    #pprint(value["generation"])
    return value["generation"]


### Nodes

def retrieve(state):  #added retreiver
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    vectordb = Chroma(persist_directory=vectorDb_Directory,
                  embedding_function=GPT4AllEmbeddings())
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):    #added rag_chain
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):        ##added retrieval_grader
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state):    ##added web_search_tool
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


### Conditional edge

def route_question(state):         ##added question_router
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source["datasource"])
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

### Conditional edge

def grade_generation_v_documents_and_question(state):    #added hallucination_grader
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"








if __name__ == "__main__":
    app.run(debug=True)