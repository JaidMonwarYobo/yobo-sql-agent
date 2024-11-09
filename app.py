import ast
import getpass
import json
import os
import re
import time
from io import BytesIO

import openai
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PIL import Image
from sqlalchemy import create_engine, inspect, text
from streamlit_chat import message

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


# Function to initialize the agent for "Product Query Agent"
@st.cache_resource
def initialize_agent_different_table(
    db_uri,
    primary_table_selection,
    product_name_col,
    secondary_table_selection,
    additional_info_column,
    primary_table_foreign_key_col,
    secondary_table_foreign_key_col,
):
    # Initialize the database
    db = SQLDatabase.from_uri(db_uri)

    st.session_state["table_config"] = {
        "primary_table": primary_table,
        "product_name_col": product_name_col,
        "secondary_tables": secondary_table_selection,
        "additional_info_columns": additional_info_column,
        "primary_foreign_keys": primary_table_foreign_key_col,
        "secondary_foreign_keys": secondary_table_foreign_key_col,
    }

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
        OpenAIEmbeddings(model="text-embedding-ada-002"),
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
    query_str = f"SELECT {product_name_col} FROM {primary_table_selection}"
    product_name_list = query_as_list(db, query_str)
    vector_db = FAISS.from_texts(
        product_name_list, OpenAIEmbeddings(model="text-embedding-ada-002")
    )
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
        verbose=False,
    )

    return agent, db, str(db.dialect), db.get_usable_table_names()


# Function to initialize the agent for "Universal Agent"
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
        [
            ("system", system),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
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
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise ValueError(
                    f"URL does not point to an image (content-type: {content_type})"
                )
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Post Image", use_column_width=True)
            return True
    except Exception as e:
        st.error(f"Failed to load image: {str(e)}")
        st.write(f"Image URL: {url}")
        return False


def response_after_executing_sql_query(product_name):
    db = st.session_state["db"]
    engine = db._engine  # Get the SQLAlchemy engine from the SQLDatabase object
    config = st.session_state["table_config"]

    # Get variables from session state
    product_name_col = st.session_state["product_name_col"]
    # additional_info_column = st.session_state["additional_info_column"]
    primary_table_selection = st.session_state["primary_table_selection"]
    # secondary_table_selection = st.session_state.get("secondary_table_selection", None)
    # primary_table_foreign_key_col = st.session_state.get(
    #     "primary_table_foreign_key_col", None
    # )
    # secondary_table_foreign_key_col = st.session_state.get(
    #     "secondary_table_foreign_key_col", None
    # )
    image_location = st.session_state["image_location"]

    # Build the SQL query dynamically depending on image_location
    if image_location == "Universal Agent":
        query_str = f"""
            SELECT *
            FROM {primary_table_selection}
            WHERE {product_name_col} LIKE :product_name;
        """
    else:
        # query_str = f"""
        #     SELECT pos.*, pis.{additional_info_column}
        #     FROM {primary_table_selection} pos
        #     LEFT JOIN {secondary_table_selection} pis
        #     ON pos.{primary_table_foreign_key_col} = pis.{secondary_table_foreign_key_col}
        #     WHERE pos.{product_name_col} LIKE :product_name;
        # """

        base_query = f"SELECT {config['primary_table']}.*"
        for i, additional_col in enumerate(config["additional_info_columns"]):
            base_query += f", {config['secondary_tables'][i]}.{additional_col}"

        # Add FROM and JOIN clauses
        base_query += f" FROM {config['primary_table']}"
        for i in range(len(config["secondary_tables"])):
            base_query += f" LEFT JOIN {config['secondary_tables'][i]} ON {config['primary_table']}.{config['primary_foreign_keys'][i]} = {config['secondary_tables'][i]}.{config['secondary_foreign_keys'][i]}"

        # Add WHERE clause based on user input
        where_clause = f" WHERE LOWER({config['primary_table']}.{config['product_name_col']}) LIKE LOWER('%{product_name}%')"

        query_str = base_query + where_clause

    # Define the SQL query with parameter substitution
    query = text(query_str)

    # Prepare the parameter with wildcards
    params = {"product_name": f"%{product_name}%"}

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
    system_prompt = f"The user asked a question: {user_input} and the response was the string below. Humanize the response into a conversation format. Also ask a question like whether the user wants to know about any other product or any other details of the same product."

    # Prepare the messages for ChatOpenAI
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=response_string),
    ]

    # Initialize the ChatOpenAI model
    chat = ChatOpenAI(
        model_name="gpt-4o-mini",  # Corrected model name
        # model="gpt-3.5-turbo",
        temperature=0.7,
    )

    # Get the assistant's reply
    assistant_reply = chat(messages).content
    return assistant_reply


def humanise_product_response_with_context(
    result_list, user_input, conversation_history
):
    # Convert the result_list to a JSON string with indentation for readability
    response_string = json.dumps(result_list, indent=2, default=str)

    # Define the system prompt
    system_prompt = f"The user asked a question: {user_input} and the response was the string below for the conversation context as: {conversation_history}. Humanize the response into a conversation format. Also ask a question like whether the user wants to know about any other product or any other details of the same product."

    # Prepare the messages for ChatOpenAI
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=response_string),
    ]

    # Initialize the ChatOpenAI model
    chat = ChatOpenAI(
        model_name="gpt-4o-mini",  # Corrected model name
        # model="gpt-3.5-turbo",
        temperature=0.7,
    )

    # Get the assistant's reply
    assistant_reply = chat(messages).content
    return assistant_reply


def humanise_response_with_context(result_list, user_input, conversation_history):
    # Convert the result_list to a JSON string with indentation for readability
    response_string = json.dumps(result_list, indent=2, default=str)

    # Define the system prompt
    system_prompt = f"The user asked a question: {user_input} and the response was the string below for the conversation context as: {conversation_history}. Humanize the response into a conversation format. Also ask a question whether the user wants to know about any more information that suits the conversation flow."

    # Prepare the messages for ChatOpenAI
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=response_string),
    ]

    # Initialize the ChatOpenAI model
    chat = ChatOpenAI(
        model_name="gpt-4o-mini",  # Corrected model name
        # model="gpt-3.5-turbo",
        temperature=0.7,
    )

    # Get the assistant's reply
    assistant_reply = chat(messages).content
    return assistant_reply


# Sidebar: Setup Agent (Placed After Database Credentials)
st.sidebar.header("AGENT SETUP")
image_location = st.sidebar.selectbox(
    "Agent Type", ["Universal Agent", "Product Query Agent"], key="image_location"
)

# Sidebar: Database Credentials (Placed First)
st.sidebar.header("DATABASE CREDENTIALS")
db_type = st.sidebar.selectbox(
    "Database Type", ["SQLite", "MySQL", "PostgreSQL"], key="db_type"
)

if db_type == "SQLite":
    db_file = st.sidebar.text_input(
        "SQLite Database File", value="Chinook.db", key="db_file"
    )
    connect_button = st.sidebar.button(
        "Connect to SQLite Database", key="connect_button_sqlite"
    )
elif db_type == "MySQL":
    host = st.sidebar.text_input("Host", value="198.12.241.155", key="host")
    port = st.sidebar.text_input("Port", value="3306", key="port")
    username = st.sidebar.text_input("Username", value="inventory_yobo", key="username")
    password = st.sidebar.text_input(
        "Password", type="password", value="", key="password"
    )
    database = st.sidebar.text_input(
        "Database Name", value="inventory_yobo", key="database"
    )
    connect_button = st.sidebar.button(
        "Connect to MySQL Database", key="connect_button_mysql"
    )
elif db_type == "PostgreSQL":
    host = st.sidebar.text_input("Host", value="localhost", key="host")
    port = st.sidebar.text_input("Port", value="5432", key="port")
    username = st.sidebar.text_input("Username", value="postgres", key="username")
    password = st.sidebar.text_input(
        "Password", value="", type="password", key="password"
    )
    database = st.sidebar.text_input(
        "Database Name", value="contactapi", key="database"
    )
    connect_button = st.sidebar.button(
        "Connect to PostgreSQL Database", key="connect_button_postgres"
    )

# Handle database connection
if connect_button:
    if db_type == "SQLite":
        if not db_file:
            st.sidebar.error("Please provide the SQLite database file path.")
            st.stop()
        db_uri = f"sqlite:///{db_file}"
    elif db_type == "MySQL":
        if not all([host, port, username, password, database]):
            st.sidebar.error("Please fill in all the database credentials.")
            st.stop()
        db_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    elif db_type == "PostgreSQL":
        if not all([host, port, username, password, database]):
            st.sidebar.error("Please fill in all the database credentials.")
            st.stop()
        db_uri = f"postgresql://{username}:{password}@{host}:{port}/{database}"

    try:
        db = SQLDatabase.from_uri(db_uri)
        db_dialect = db.dialect
        table_names = db.get_usable_table_names()
        st.session_state["db"] = db
        st.session_state["db_uri"] = db_uri
        st.session_state["db_dialect"] = db_dialect
        st.session_state["table_names"] = table_names

        # Retrieve columns for each table
        inspector = inspect(db._engine)
        table_columns = {}
        for table in table_names:
            columns = inspector.get_columns(table)
            table_columns[table] = [column["name"] for column in columns]
        st.session_state["table_columns"] = table_columns

        st.success(f"Connected to the {db_type} database successfully!")

        # Automatically populate the primary_table_selection if not already selected
        if (
            "image_location" in st.session_state
            and st.session_state["image_location"] == "Product Query Agent"
            and "primary_table_selection" not in st.session_state
        ):
            st.session_state["primary_table_selection"] = (
                table_names[0] if table_names else None
            )

    except Exception as e:
        st.sidebar.error(f"Failed to connect to the database: {e}")
        st.stop()


# Sidebar: Schema Configuration (Only for "Product Query Agent" and after Database Connection)
if image_location == "Product Query Agent":
    st.sidebar.header("SCHEMA CONFIGURATION")
    st.sidebar.subheader("Primary Table Configuration:")
    col1, col2 = st.sidebar.columns(2)

    # Ensure that table_names are available
    if "table_names" in st.session_state and st.session_state["table_names"]:
        table_names = st.session_state["table_names"]

        # Primary Table Selection

        primary_table = col1.selectbox(
            "Select Primary Table for product information",
            options=table_names,
            key="primary_table_selection",
        )

        # Product Name Column Selection
        if (
            "table_columns" in st.session_state
            and primary_table in st.session_state["table_columns"]
        ):
            primary_table_columns = st.session_state["table_columns"][primary_table]
            if primary_table_columns:
                product_name_col = col2.selectbox(
                    "Select Product Name Column",
                    options=primary_table_columns,
                    key="product_name_col",
                )
            else:
                st.sidebar.warning("No columns found for the selected primary table.")
                st.stop()
        else:
            st.sidebar.warning("No columns found for the selected primary table.")
            st.stop()

        # Secondary Table Selection (excluding the primary table)
        st.sidebar.subheader("Additional Attribute Configuration:")
        col1, col2 = st.sidebar.columns(2)
        # secondary_tables = [tbl for tbl in table_names if tbl != primary_table] # Exclude for multiple sec tables
        if "secondary_tables" not in st.session_state:
            st.session_state["secondary_tables"] = []
            st.session_state["additional_info_columns"] = []
            st.session_state["primary_foreign_keys"] = []
            st.session_state["secondary_foreign_keys"] = []

        def add_secondary_table():
            st.session_state["secondary_tables"].append(None)
            st.session_state["additional_info_columns"].append(None)
            st.session_state["primary_foreign_keys"].append(None)
            st.session_state["secondary_foreign_keys"].append(None)

        if st.sidebar.button("Add Another Attribute"):
            add_secondary_table()

        for i in range(len(st.session_state["secondary_tables"])):
            st.sidebar.subheader(f"Additional Attribute #{i+1}")
            col1, col2 = st.sidebar.columns(2)

            # Secondary table selection
            secondary_table = col1.selectbox(
                "Select Table for Additional Attribute",
                options=st.session_state["table_names"],
                key=f"secondary_table_selection_{i}",
            )
            st.session_state["secondary_tables"][i] = secondary_table

            # Additional info column selection
            if (
                "table_columns" in st.session_state
                and secondary_table in st.session_state["table_columns"]
            ):
                secondary_table_columns = st.session_state["table_columns"][
                    secondary_table
                ]
                additional_info_column = col2.selectbox(
                    "Select Column for Additional Attribute",
                    options=secondary_table_columns,
                    key=f"additional_info_column_{i}",
                )
                st.session_state["additional_info_columns"][i] = additional_info_column

            # Foreign Key Configuration
            st.sidebar.subheader(f"Foreign Key Configuration #{i+1}")
            col1, col2 = st.sidebar.columns(2)

            primary_table_foreign_key_col = col1.selectbox(
                "Select Foreign Key Column from Primary Table",
                options=primary_table_columns,
                key=f"primary_table_foreign_key_col_{i}",
            )
            st.session_state["primary_foreign_keys"][i] = primary_table_foreign_key_col

            secondary_table_foreign_key_col = col2.selectbox(
                "Select Foreign Key Column from Additional Attribute Table",
                options=secondary_table_columns,
                key=f"secondary_table_foreign_key_col_{i}",
            )
            st.session_state["secondary_foreign_keys"][
                i
            ] = secondary_table_foreign_key_col

        # # Add button for new secondary table
        # if st.sidebar.button("Add Another Secondary Table"):
        #     add_secondary_table()

        # Initialize button
        if st.sidebar.button("Initialize Agent"):
            try:
                # Get selections from widgets
                primary_table = st.session_state["primary_table_selection"]
                product_name_col = st.session_state["product_name_col"]

                # Pass all secondary table configurations to initialize_agent
                agent, db, db_dialect, table_names_updated = (
                    initialize_agent_different_table(
                        st.session_state["db_uri"],
                        primary_table,
                        product_name_col,
                        st.session_state["secondary_tables"],
                        st.session_state["additional_info_columns"],
                        st.session_state["primary_foreign_keys"],
                        st.session_state["secondary_foreign_keys"],
                    )
                )

                # Assign to session_state
                st.session_state["agent"] = agent
                st.session_state["db"] = db
                st.session_state["db_dialect"] = db_dialect
                st.session_state["table_names"] = table_names_updated
                st.success("Product Query Agent initialized successfully!")
            except Exception as e:
                st.sidebar.error(f"Failed to initialize the agent: {e}")
                st.stop()
    else:
        st.sidebar.warning("Please connect to a database first.")
        st.stop()

# Ensure the database is connected
if "db" in st.session_state:
    db = st.session_state["db"]
    db_uri = st.session_state["db_uri"]
    db_dialect = st.session_state["db_dialect"]
    table_names = st.session_state["table_names"]

    # Function to get column names
    def get_column_names(db, table_name):
        inspector = inspect(db._engine)
        columns = inspector.get_columns(table_name)
        return [column["name"] for column in columns]

    # Retrieve columns for each table
    table_columns = {}
    for table in table_names:
        columns = get_column_names(db, table)
        table_columns[table] = columns

    st.session_state["table_columns"] = table_columns

    # Sidebar for selecting columns (only for "Universal Agent")
    if image_location == "Universal Agent":
        st.sidebar.header(
            "Select Column with Proper Nouns (such as usernames, product names, etc.). Choose only ONE column in total."
        )
        selected_columns = {}
        for table in table_names:
            with st.sidebar.expander(f"Select columns from {table}"):
                columns = table_columns[table]
                selected = st.multiselect(
                    f"Columns from {table}", columns, key=f"{table}_columns"
                )
                if selected:
                    selected_columns[table] = selected

        # Button to initialize the agent
        if st.sidebar.button("Initialize Agent"):
            if not selected_columns:
                st.sidebar.error("Please select at least one column for vectorization.")
            else:
                try:
                    agent, db_dialect, table_names = initialize_agent_same_table(
                        db_uri, selected_columns
                    )
                    st.session_state["agent"] = agent
                    st.session_state["db_dialect"] = db_dialect
                    st.session_state["table_names"] = table_names
                    st.success("Agent initialized successfully!")
                except Exception as e:
                    st.sidebar.error(f"Failed to initialize the agent: {e}")
                    st.stop()
else:
    st.warning("Please connect to a database in the sidebar.")
    st.stop()

# Ensure the agent is initialized
if "agent" in st.session_state:
    agent = st.session_state["agent"]
    db_dialect = st.session_state["db_dialect"]
    table_names = st.session_state["table_names"]
else:
    st.warning("Please initialize the agent in the sidebar.")
    st.stop()

if "chat_display" not in st.session_state:
    st.session_state["chat_display"] = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "final_output" not in st.session_state:
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

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]

    deciding_agent = ChatOpenAI(
        model="gpt-4o-mini",  # Corrected model name
        # model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=3,
        api_key=openai_api_key,
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
                if image_location == "Universal Agent":
                    start_time = time.perf_counter()
                    response = agent.invoke(
                        {
                            "input": user_input
                            + " and the previous context of the conversation is as follows: "
                            + str(st.session_state.conversation_history)
                        }
                    )

                    st.success("Query executed successfully!")
                    st.subheader("Response:")
                    if isinstance(response, dict) and "output" in response:
                        output = response["output"]
                    else:
                        output = str(response)

                    end_time = time.perf_counter()
                    elapsed_time_ms = (end_time - start_time) * 1000

                    conversation_pair = {
                        "user_message": user_input,
                        "assistant_response": output,
                    }

                    st.session_state.conversation_history.append(conversation_pair)
                    print(
                        "Conversation History:", st.session_state.conversation_history
                    )

                    st.session_state.chat_display.append(
                        {"message": user_input, "is_user": True}
                    )
                    st.session_state.chat_display.append(
                        {"message": output, "is_user": False}
                    )

                    for i, chat in enumerate(st.session_state.chat_display):
                        message(chat["message"], is_user=chat["is_user"], key=str(i))

                    chat_state = True
                    print("Chat state:", chat_state)

                    # Process the response line by line
                    lines = output.split("\n")
                    for line in lines:
                        # Check for image URLs
                        if line.startswith("Image URL:"):
                            url = line.split("Image URL:")[1].strip()
                            display_image(url)
                        elif "http" in line and any(
                            ext in line.lower()
                            for ext in [".jpg", ".jpeg", ".png", ".gif"]
                        ):
                            url_match = re.search(
                                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                                line,
                            )
                            if url_match:
                                url = url_match.group(0)
                                display_image(url)
                            st.write(line)
                        else:
                            st.write(line)

                    st.info(f"Total Response Time: {elapsed_time_ms:.2f} ms")
                elif image_location == "Product Query Agent":
                    start_time = time.perf_counter()

                    if new_product_decider.lower() == "yes":
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
                        st.session_state.final_output = (
                            final_output  # Save the final output in session state
                        )
                        st.write(f"Retrieved Name: {output}")
                        natural_output = humanise_response(final_output, user_input)

                        # Display the query results
                        if natural_output:
                            st.write("Query Results:")
                            # st.dataframe(final_output)
                            st.write(f"Yobo's Response: {natural_output}")
                        else:
                            st.write("No results found for the given product name.")

                        end_formatting_time = time.perf_counter()
                        elapsed_time_formatting_ms = (
                            end_formatting_time - end_time_name
                        ) * 1000

                        # Debug Chat State
                        conversation_pair = {
                            "user_message": user_input,
                            "assistant_response": natural_output,
                        }
                        # conversation_histories.append(conversation_pair)
                        # print("Conversation History:", conversation_histories)

                        st.session_state.conversation_history.append(conversation_pair)
                        print(
                            "Conversation History:",
                            st.session_state.conversation_history,
                        )

                        st.session_state.chat_display.append(
                            {"message": user_input, "is_user": True}
                        )
                        st.session_state.chat_display.append(
                            {"message": natural_output, "is_user": False}
                        )

                        st.subheader("Conversation History")
                        for i, chat in enumerate(st.session_state.chat_display):
                            message(
                                chat["message"], is_user=chat["is_user"], key=str(i)
                            )

                        chat_state = True
                        print("Chat state:", chat_state)
                    elif new_product_decider.lower() == "no":
                        final_output = st.session_state.final_output
                        natural_output = humanise_product_response_with_context(
                            final_output,
                            user_input,
                            st.session_state.conversation_history,
                        )

                        st.write(f"Yobo's Response: {natural_output}")

                        elapsed_time_name_ms = 0
                        end_formatting_time = time.perf_counter()
                        elapsed_time_formatting_ms = (
                            end_formatting_time - start_time
                        ) * 1000

                        conversation_pair = {
                            "user_message": user_input,
                            "assistant_response": natural_output,
                        }

                        st.session_state.conversation_history.append(conversation_pair)
                        print(
                            "Conversation History:",
                            st.session_state.conversation_history,
                        )

                        st.session_state.chat_display.append(
                            {"message": user_input, "is_user": True}
                        )
                        st.session_state.chat_display.append(
                            {"message": natural_output, "is_user": False}
                        )

                        st.subheader("Conversation History")

                        for i, chat in enumerate(st.session_state.chat_display):
                            message(
                                chat["message"], is_user=chat["is_user"], key=str(i)
                            )

                        chat_state = True
                        print("Chat state:", chat_state)

                    end_time = time.perf_counter()
                    elapsed_time_ms = (end_time - start_time) * 1000

                    st.info(
                        f"Time taken for Finding Name (t_1): {elapsed_time_name_ms:.2f} ms"
                    )
                    st.info(
                        f"Time taken for Executing Query and Formatting Query (t_2): {elapsed_time_formatting_ms:.2f} ms"
                    )
                    st.info(
                        f"Total Response Time (t_1 + t_2): {elapsed_time_ms:.2f} ms"
                    )

            except Exception as e:
                st.error(f"An error occurred: {e}")

if st.sidebar.button("Clear Chat"):
    st.session_state.chat_display = []
    st.session_state.conversation_history = []
    st.session_state.final_output = None

# Agent Logs (Optional)
st.markdown("---")
st.subheader("Agent Logs")
