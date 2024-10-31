import streamlit as st
import getpass
import os
import time
from dotenv import load_dotenv
from streamlit_chat import message
from sqlalchemy import create_engine, text, inspect
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
import ast
import re
import requests
from PIL import Image
from io import BytesIO
import json
import openai

# Initialize Streamlit app
st.set_page_config(page_title="SQL Agent Interface", layout="wide")
st.title("SQL Database Agent Interface")
chat_state = False
conversation_histories = []

# Load environment variables
load_dotenv()

# Retrieve OpenAI API key from environment
# openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = st.secrets["OPENAI_API_KEY"]

if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in your `.env` file.")
    st.stop()

# Define a helper function to query as a list
def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

# Function to initialize the agent for "Multi Table Architecture"
@st.cache_resource
def initialize_agent_different_table(db_uri):
    # Initialize the database
    db = SQLDatabase.from_uri(db_uri)
    
    # Get variables from session state
    product_name_col = st.session_state['product_name_col']
    products_table = st.session_state['products_table']
    thumb_pic_col = st.session_state['thumb_pic_col']
    products_images_table = st.session_state.get('products_images_table', None)
    supplier_product_id_col = st.session_state.get('supplier_product_id_col', None)

    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)

    # Define examples for semantic similarity
    examples = [
        {"input": "List all artists.", "query": "SELECT * FROM Artist;"},
        {
            "input": "Find all albums for the artist 'AC/DC'.",
            "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
        },
    ]

    # Initialize example selector
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(
            model="text-embedding-3-small"
        ),
        FAISS,
        k=10,
        input_keys=["input"],
    )

     # **Modified System Prompt**
    system_prefix = """You are an agent designed to find and output proper names from a vector database.
    When given an input, you must return only the proper name found in the vector database.
    Do not perform any SQL operations or generate any SQL queries.
    Do not provide any additional analysis, commentary, or other information.
    Only return the name from the vector database."""

    # Create a simple retriever-only prompt
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nRetrieved name: {query}"
        ),
        input_variables=["input"],
        prefix=system_prefix,
        suffix="",
    )

    # Create the full chat prompt
    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Define the retriever tool with vector DB
    query_str = f"SELECT {product_name_col} FROM {products_table}"
    product_name_list = query_as_list(db, query_str)
    vector_db = FAISS.from_texts(product_name_list, OpenAIEmbeddings(model="text-embedding-3-small"))
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_proper_nouns",
        description="Use to look up proper nouns. Input is an approximate spelling of the proper noun, output is valid proper nouns.",
    )

    # Create the SQL agent
    agent = create_sql_agent(
        llm=llm,
        db=db,
        extra_tools=[retriever_tool],
        prompt=full_prompt,
        agent_type="openai-tools",
        verbose=False
    )

    return agent, db, str(db.dialect), db.get_usable_table_names()

# Function to initialize the agent for "Single Table Architecture"
def initialize_agent_same_table(db_uri, selected_columns):
    # Initialize the database
    db = SQLDatabase.from_uri(db_uri)
    db_dialect = db.dialect
    table_names = db.get_usable_table_names()

    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)

    # Retrieve data from selected columns
    all_texts = []
    for table, columns in selected_columns.items():
        for column in columns:
            query = f"SELECT {column} FROM {table}"
            data = query_as_list(db, query)
            all_texts.extend(data)

    # Vectorize the combined data
    vector_db = FAISS.from_texts(all_texts, OpenAIEmbeddings())
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_proper_nouns",
        description="Use to look up values to filter on.",
    )

    # Create the SQL agent
    system = f"""You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {db_dialect} query to run, then look at the results of the query and return the answer.
    You have access to the following tables: {table_names}.
    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{input}"), MessagesPlaceholder("agent_scratchpad")]
    )

    agent = create_sql_agent(
        llm=llm,
        db=db,
        extra_tools=[retriever_tool],
        prompt=prompt,
        agent_type="openai-tools",
        verbose=True,
    )

    return agent, db_dialect, table_names

# Function to display image from URL
def display_image(url):
    try:
        with st.spinner(f"Loading image from {url}..."):
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise ValueError(f"URL does not point to an image (content-type: {content_type})")
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Post Image", use_column_width=True)
            return True
    except Exception as e:
        st.error(f"Failed to load image: {str(e)}")
        st.write(f"Image URL: {url}")
        return False


def response_after_executing_sql_query(product_name):
    db = st.session_state['db']
    engine = db._engine  # Get the SQLAlchemy engine from the SQLDatabase object

    # Get variables from session state
    product_name_col = st.session_state['product_name_col']
    thumb_pic_col = st.session_state['thumb_pic_col']
    products_table = st.session_state['products_table']
    products_images_table = st.session_state.get('products_images_table', None)
    supplier_product_id_col = st.session_state.get('supplier_product_id_col', None)
    image_location = st.session_state['image_location']

    # Build the SQL query dynamically depending on image_location
    if image_location == "Single Table Architecture":
        query_str = f'''
            SELECT *
            FROM {products_table}
            WHERE {product_name_col} LIKE :product_name;
        '''
    else:
        query_str = f'''
            SELECT pos.*, pis.{thumb_pic_col}
            FROM {products_table} pos
            LEFT JOIN {products_images_table} pis
            ON pos.{supplier_product_id_col} = pis.{supplier_product_id_col}
            WHERE pos.{product_name_col} LIKE :product_name;
        '''

    # Define the SQL query with parameter substitution
    query = text(query_str)

    # Prepare the parameter with wildcards
    params = {'product_name': f'%{product_name}%'}
    
    # Execute the query
    with engine.connect() as connection:
        result = connection.execute(query, params)
        rows = result.fetchall()
        columns = result.keys()

    # Return the results as a list of dictionaries
    result_list = [dict(zip(columns, row)) for row in rows]
    return result_list

def humanise_response(result_list, user_input):
    # Convert the result_list to a JSON string with indentation for readability
    response_string = json.dumps(result_list, indent=2, default=str)
    
    # Define the system prompt
    system_prompt = f"The user asked a question: {user_input} and the response was the string below. Humanize the response into a human-readable format. Also ask a question like whether the user wants to know about any other product or any other details of the same product."
    
    # Prepare the messages for ChatOpenAI
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=response_string)
    ]
    
    # Initialize the ChatOpenAI model
    chat = ChatOpenAI(
        model_name="gpt-4o-mini",  # Replace with "gpt-4" if you have access
        # model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Get the assistant's reply
    assistant_reply = chat(messages).content
    return assistant_reply

# Sidebar for selecting image location
st.sidebar.header("Any Additional table information with Main Table")
image_location = st.sidebar.selectbox("Agent Type", ["Single Table Architecture", "Multi Table Architecture"], key='image_location')

# Sidebar input fields for "Multi Table Architecture"
if image_location == "Multi Table Architecture":
    st.sidebar.header("Schema Configuration")
    product_name_col = st.sidebar.text_input("Exact Column Name for Product Name Column (Column which contains proper noun of products/users/companies)", value="wp_title", key='product_name_col')
    thumb_pic_col = st.sidebar.text_input("Exact Column Name for Additional Info Column", value="variation_name", key='thumb_pic_col')
    products_table = st.sidebar.text_input("Exact Table Name for the Main Table", value="wp_data", key='products_table')
    products_images_table = st.sidebar.text_input("Exact Table Name for Additional Info Column", value="variations_data", key='products_images_table')
    supplier_product_id_col = st.sidebar.text_input("Exact Column Name for Foreign ID (column name which is on the Additional Info Table)", value="wp_id", key='supplier_product_id_col')
    # product_name_col = st.sidebar.text_input("Product Name Column", value="product_name", key='product_name_col')
    # thumb_pic_col = st.sidebar.text_input("Thumbnail Picture Column", value="thumb_pic", key='thumb_pic_col')
    # products_table = st.sidebar.text_input("Products Table Name", value="products_of_suppliers", key='products_table')
    # products_images_table = st.sidebar.text_input("Product Images Table Name", value="products_images_of_suppliers", key='products_images_table')
    # supplier_product_id_col = st.sidebar.text_input("Product ID Mapping Column", value="supplier_product_id", key='supplier_product_id_col')

# Database connection settings
st.sidebar.header("Database Credentials")
db_type = st.sidebar.selectbox("Database Type", ["SQLite", "MySQL", "PostgreSQL"], key='db_type')

if db_type == "SQLite":
    db_file = st.sidebar.text_input("SQLite Database File", value="Chinook.db", key='db_file')
    connect_button = st.sidebar.button("Connect to SQLite Database", key='connect_button_sqlite')
elif db_type == "MySQL":
    host = st.sidebar.text_input("Host", value="198.12.241.155", key='host')
    port = st.sidebar.text_input("Port", value="3306", key='port')
    username = st.sidebar.text_input("Username", value="inventory_yobo", key='username')
    password = st.sidebar.text_input("Password", type="password", value="iz-ENVMm{+#[", key='password')
    database = st.sidebar.text_input("Database Name", value="inventory_yobo", key='database')

    # host = st.sidebar.text_input("Host", value="localhost", key='host')
    # port = st.sidebar.text_input("Port", value="3306", key='port')
    # username = st.sidebar.text_input("Username", value="root", key='username')
    # password = st.sidebar.text_input("Password", type="password", value="Admin1234", key='password')
    # database = st.sidebar.text_input("Database Name", value="circle", key='database')
    connect_button = st.sidebar.button("Connect to MySQL Database", key='connect_button_mysql')
elif db_type == "PostgreSQL":
    host = st.sidebar.text_input("Host", value="localhost", key='host')
    port = st.sidebar.text_input("Port", value="5432", key='port')
    username = st.sidebar.text_input("Username", value="postgres", key='username')
    password = st.sidebar.text_input("Password", value="admin1234", type="password", key='password')
    database = st.sidebar.text_input("Database Name", value="contactapi", key='database')
    connect_button = st.sidebar.button("Connect to PostgreSQL Database", key='connect_button_postgres')

# Handle database connection
if connect_button:
    if db_type == "SQLite":
        if not db_file:
            st.sidebar.error("Please provide the SQLite database file path.")
            st.stop()
        db_uri = f"sqlite:///{db_file}"
        chat_state = False
    elif db_type == "MySQL":
        if not all([host, port, username, password, database]):
            st.sidebar.error("Please fill in all the database credentials.")
            st.stop()
        db_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        chat_state = False
        print("Chat state:", chat_state)
    elif db_type == "PostgreSQL":
        if not all([host, port, username, password, database]):
            st.sidebar.error("Please fill in all the database credentials.")
            st.stop()
        db_uri = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        chat_state = False
    
    try:
        if image_location == "Multi Table Architecture":
            st.session_state['db_uri'] = db_uri
            agent, db, db_dialect, table_names = initialize_agent_different_table(db_uri)
            st.session_state['agent'] = agent
            st.session_state['db'] = db
            st.session_state['db_dialect'] = db_dialect
            st.session_state['table_names'] = table_names
        else:
            db = SQLDatabase.from_uri(db_uri)
            db_dialect = db.dialect
            table_names = db.get_usable_table_names()
            st.session_state['db'] = db
            st.session_state['db_uri'] = db_uri
            st.session_state['db_dialect'] = db_dialect
            st.session_state['table_names'] = table_names
            st.success(f"Connected to the {db_type} database successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to connect to the database: {e}")
        st.stop()

# Ensure the database is connected
if 'db' in st.session_state:
    db = st.session_state['db']
    db_uri = st.session_state['db_uri']
    db_dialect = st.session_state['db_dialect']
    table_names = st.session_state['table_names']

    # Function to get column names
    def get_column_names(db, table_name):
        inspector = inspect(db._engine)
        columns = inspector.get_columns(table_name)
        return [column['name'] for column in columns]

    # Retrieve columns for each table
    table_columns = {}
    for table in table_names:
        columns = get_column_names(db, table)
        table_columns[table] = columns

    # Sidebar for selecting columns (only for "Single Table Architecture")
    if image_location == "Single Table Architecture":
        st.sidebar.header("Select Column which has Proper Nouns (specifically Product Names)")
        selected_columns = {}
        for table in table_names:
            with st.sidebar.expander(f"Select columns from {table}"):
                columns = table_columns[table]
                selected = st.multiselect(f"Columns from {table}", columns, key=f"{table}_columns")
                if selected:
                    selected_columns[table] = selected

        # Button to initialize the agent
        if st.sidebar.button("Initialize Agent"):
            if not selected_columns:
                st.sidebar.error("Please select at least one column for vectorization.")
            else:
                try:
                    agent, db_dialect, table_names = initialize_agent_same_table(db_uri, selected_columns)
                    st.session_state['agent'] = agent
                    st.session_state['db_dialect'] = db_dialect
                    st.session_state['table_names'] = table_names
                    st.success("Agent initialized successfully!")
                except Exception as e:
                    st.sidebar.error(f"Failed to initialize the agent: {e}")
                    st.stop()
else:
    st.warning("Please connect to a database in the sidebar.")
    st.stop()

# Ensure the agent is initialized
if 'agent' in st.session_state:
    agent = st.session_state['agent']
    db_dialect = st.session_state['db_dialect']
    table_names = st.session_state['table_names']
else:
    st.warning("Please initialize the agent in the sidebar.")
    st.stop()

if 'chat_display' not in st.session_state:
    st.session_state['chat_display'] = []

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'final_output' not in st.session_state:
    st.session_state.final_output = None

# Sidebar Information
st.sidebar.header("Database Information")
st.sidebar.write(f"**Dialect:** {db_dialect}")
st.sidebar.write(f"**Tables:** {', '.join(table_names)}")

# User Input
st.header("Enter Your Query")
user_input = st.text_input("Type your SQL-related question here:", "")


def new_product(conversation_history, user_input):
    system_prompt = f"""
    You are an assistant that determines if the user's new question is about a different product than previously discussed.
    Conversation history: {conversation_history}
    If the user's new question is about the same product as in the conversation history, respond with 'no'.
    If it is about a different product, or if it is a general statement (e.g., greetings, thanks, bye), respond with 'yes'.
    Provide only 'yes' or 'no' as your response.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    deciding_agent = ChatOpenAI(
        model="gpt-4o-mini",
        # model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=3,
        api_key=openai_api_key
    )
    assistant_reply = deciding_agent(messages).content.strip()
    return assistant_reply


if st.button("Submit"):
    # chat_state = 2
    print("Chat state:", chat_state)
    # print("Conversation History:", conversation_histories)
    new_product_decider = new_product(st.session_state.conversation_history, user_input)
    print("New Product Decider:", new_product_decider)
    if user_input.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Processing your query..."):
            try:
                if image_location == "Single Table Architecture":
                    start_time = time.perf_counter()
                    response = agent.invoke({"input": user_input})

                    st.success("Query executed successfully!")
                    st.subheader("Response:")
                    if isinstance(response, dict) and "output" in response:
                        output = response["output"]
                    else:
                        output = str(response)

                    # Process the response line by line
                    lines = output.split('\n')
                    for line in lines:
                        # Check for image URLs
                        if line.startswith("Image URL:"):
                            url = line.split("Image URL:")[1].strip()
                            display_image(url)
                        elif "http" in line and any(ext in line.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                            url_match = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line)
                            if url_match:
                                url = url_match.group(0)
                                display_image(url)
                            st.write(line)
                        else:
                            st.write(line)
                elif image_location == "Multi Table Architecture":
                    start_time = time.perf_counter()
                    
                    # if chat_state == False:
                    if new_product_decider == "yes":
                        # Check if the response contains the expected output format
                        response = agent.invoke({"input": user_input})

                        st.success("Query executed successfully!")
                        st.subheader("Response:")

                        end_time_name = time.perf_counter()
                        elapsed_time_name_ms = (end_time_name - start_time) * 1000

                        if isinstance(response, dict) and "output" in response:
                            output = response["output"]
                        else:
                            output = str(response)

                        # Display only the name
                        final_output = response_after_executing_sql_query(output)
                        st.session_state.final_output = final_output # Save the final output in session state
                        st.write(f"Retrieved Name: {output}")
                        natural_output = humanise_response(final_output, user_input)

                        # Display the query results
                        if natural_output:
                            st.write("Query Results:")
                            # st.dataframe(final_output)
                            st.write(f"Retrieved Name: {natural_output}")
                        else:
                            st.write("No results found for the given product name.")

                        end_formatting_time = time.perf_counter()
                        elapsed_time_formatting_ms = (end_formatting_time - end_time_name) * 1000
                        
                        
                        # Debug Chat State
                        conversation_pair = {
                            "user_message": user_input,
                            "assistant_response": natural_output
                        }
                        # conversation_histories.append(conversation_pair)
                        # print("Conversation History:", conversation_histories)

                        st.session_state.conversation_history.append(conversation_pair)
                        print("Conversation History:", st.session_state.conversation_history)

                        st.session_state.chat_display.append({"message": user_input, "is_user": True})
                        st.session_state.chat_display.append({"message": natural_output, "is_user": False})

                        for i, chat in enumerate(st.session_state.chat_display):
                            message(
                                chat["message"],
                                is_user=chat["is_user"],
                                key=str(i)
                            )

                        chat_state = True
                        print("Chat state:", chat_state)
                    if new_product_decider == "no":
                        final_output = st.session_state.final_output
                        natural_output = humanise_response(final_output, user_input)

                        elapsed_time_name_ms = 0
                        end_formatting_time = time.perf_counter()
                        elapsed_time_formatting_ms = (end_formatting_time - start_time) * 1000
                        
                        conversation_pair = {
                            "user_message": user_input,
                            "assistant_response": natural_output
                        }

                        st.session_state.conversation_history.append(conversation_pair)
                        print("Conversation History:", st.session_state.conversation_history)

                        st.session_state.chat_display.append({"message": user_input, "is_user": True})
                        st.session_state.chat_display.append({"message": natural_output, "is_user": False})

                        for i, chat in enumerate(st.session_state.chat_display):
                            message(
                                chat["message"],
                                is_user=chat["is_user"],
                                key=str(i)
                            )

                        chat_state = True
                        print("Chat state:", chat_state)




                    # # Check if the response contains the expected output format
                    # if isinstance(response, dict) and "output" in response:
                    #     output = response["output"]
                    # else:
                    #     output = str(response)

                    # # Display only the name
                    # final_output = response_after_executing_sql_query(output)
                    # st.write(f"Retrieved Name: {output}")
                    # natural_output = humanise_response(final_output, user_input)

                    # # Display the query results
                    # if natural_output:
                    #     st.write("Query Results:")
                    #     # st.dataframe(final_output)
                    #     st.write(f"Retrieved Name: {natural_output}")
                    # else:
                    #     st.write("No results found for the given product name.")

                    end_time = time.perf_counter()
                    elapsed_time_ms = (end_time - start_time) * 1000
                
                

                
                st.info(f"Time taken for Finding Name: {elapsed_time_name_ms:.2f} ms")
                st.info(f"Time taken for Executing Query and Formatting Query: {elapsed_time_formatting_ms:.2f} ms")
                st.info(f"Time taken for Overall Query: {elapsed_time_ms:.2f} ms")

            except Exception as e:
                st.error(f"An error occurred: {e}")

if st.sidebar.button("Clear Chat"):
    st.session_state.chat_display = []
    st.session_state.conversation_history = []
    st.session_state.final_output = None

# Agent Logs (Optional)
st.markdown("---")
st.subheader("Agent Logs")