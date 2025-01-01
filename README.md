# TEXT-TO-SQL Bot Using LlamaIndex And Gemini

This project provides a natural language to SQL (NL2SQL) conversion tool using machine learning models, specifically leveraging **Gemini** LLM and **SQLAlchemy** for database interactions. The app allows users to input natural language queries, and it will convert them into corresponding SQL queries, which are then executed against a database.

The app is designed with **Streamlit** as the web interface, which makes it user-friendly and interactive. It uses advanced embeddings and a pre-defined retrieval system to intelligently map SQL schemas and generate highly accurate SQL queries.

---

## Features

- **NL2SQL Conversion**: Input a natural language query, and the app will generate and display an SQL query.
- **SQL Query Execution**: The generated SQL queries are executed against a specified database (SQLite in this case).
- **Database Schema Awareness**: The app intelligently retrieves relevant schema information for query generation.
- **Interactive UI**: Built with **Streamlit** for easy interaction and results visualization.

---

## Requirements

Ensure you have the following prerequisites before setting up the app:

- Python >= 3.7
- Streamlit
- SQLAlchemy
- Gemini LLM API
- LlamaIndex (for database interaction and indexing)
- SQLite (or another SQL database)

---

## Installation

### Step 1: Clone the Repository

Clone the project repository to your local machine:

```bash
git clone https://github.com/rohannaika7/text-to-sql.git
cd text-to-sql.git
```

### Step 2: Install Dependencies

It is recommended to use a virtual environment for this project. You can create and activate one with:

```bash
python3 -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows
```

Then, install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies, including:

- **Streamlit**
- **SQLAlchemy**
- **LlamaIndex**
- **Gemini**

---

## Configuration

### Step 1: Set up API Keys

You'll need API keys for Gemini (LLM) and OpenAI (if used):

- **Gemini API Key**: Set up your Gemini API key by following the instructions on the Gemini documentation and paste it where indicated in the code.

Update the respective code block with your API keys:

```python
llm = Gemini(model_name="models/gemini-1.5-pro", api_key="GEMINI API KEY", temperature=0, max_tokens=1048576)
embeddings = GeminiEmbedding(model="models/textembedding-gecko@004", api_key="GEMINI API KEY")
```

### Step 2: Set Up Database

This app uses SQLite as a default database, but you can modify the connection string for other databases.

- Place your SQLite database file (`bankdeposit.db`) in the project folder, or configure the connection string accordingly if using a different SQL database.

```python
engine = create_engine("sqlite:///bankdeposit.db")
```

### Step 3: Add Table-Level and Column-Level Descriptions

To enhance the understanding of your database schema and improve the quality of generated SQL queries, you can add **table-level** and **column-level descriptions** to each table and column. These descriptions provide helpful context about the purpose of each table and its individual columns, which can assist in generating more accurate SQL queries.

Here’s how you can add the descriptions:

#### 1. **Table-Level Descriptions**:
In the `table_details` dictionary, define descriptions for each table. These descriptions will be helpful for understanding the context of each table in the database.

Example:

```python
table_details = {
    "table1_name": "This table contains transaction records, including account holder information, transaction types, and amounts.",
    "table2_name": "This table stores customer details, including personal information and account balances."
}
```

#### 2. **Column-Level Descriptions**:
For each table in your database, you can add descriptions for individual columns. This is done in the `data` dictionary, where each table's columns are listed along with their descriptions.

Example:

```python
data = {
    "table1_name": {
        "column1_name": "The unique identifier for the transaction.",
        "column2_name": "The amount of the transaction.",
        "column3_name": "The date when the transaction occurred.",
    },
    "table2_name": {
        "column1_name": "The unique identifier for the customer.",
        "column2_name": "The customer's full name.",
        "column3_name": "The balance in the customer's account."
    }
}
```

#### 3. **Update Metadata with Descriptions**:
Once you've added the table and column descriptions, the next step is to reflect these descriptions into the database schema metadata.

```python
metadata = MetaData()
metadata.reflect(bind=engine)

# Add descriptions to the tables and columns
for table_name, table_desc in table_details.items():
    table = metadata.tables[table_name]
    table.comment = table_desc  # Add table-level description

    for column_name, column_desc in data.get(table_name, {}).items():
        column = table.columns[column_name]
        column.comment = column_desc  # Add column-level description
```
---

## Running the App

To run the app, simply execute the following command:

```bash
streamlit run app.py
```

This will start the Streamlit server, and you can access the app in your browser by navigating to `http://localhost:8501`.

---

## How the App Works

1. **Natural Language Input**: The user inputs a query in natural language.
   
2. **Text-to-SQL Conversion**: The input is processed by the `Gemini` LLM, which generates an SQL query based on the natural language query.

3. **SQL Query Execution**: The SQL query is executed against the connected database, and results are fetched.

4. **Display Results**: The app displays the generated SQL query and the query results (if any) in an easy-to-read format.

---

## Code Structure

Here’s a breakdown of the main components in the code:

### 1. **Imports and Setup**
The app uses the following key libraries:
- `SQLAlchemy` for interacting with the database.
- `GeminiEmbedding` and `Gemini` for text-to-SQL transformation.
- `LlamaIndex` for indexing retrieving database schemas, helping to generate more accurate SQL queries.
- `Streamlit` for the interactive user interface.

### 2. **Database Connection**
The SQLite database connection is established using `SQLAlchemy`:

```python
engine = create_engine("sqlite:///bankdeposit.db")
```

### 3. **Llama Index Setup**
The **Llama Index** is used for managing SQL table schemas:

```python
sql_database = SQLDatabase(engine, sample_rows_in_table_info=100, include_tables=["table1_name", "table2_name"], metadata=metadata)
```

### 4. **Query Pipeline**
The query pipeline (`QP`) is where the entire flow from input to SQL generation and execution takes place. It uses several components like:
- **Input Component**: Takes in user input.
- **Text-to-SQL LLM**: Translates natural language into an SQL query.
- **SQL Retriever**: Executes the generated SQL query.
- **Response Synthesis**: Forms the final response from the SQL result.

### 5. **Prompt Templates**
The system uses predefined prompt templates for the `Gemini` model:

```python
text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(dialect=engine.dialect.name)
```

---

## Running the App in a Production Environment

If you want to run this application in a production environment:

1. **Deploying on a Cloud Platform**: You can deploy this app on platforms such as **Heroku**, **AWS**, or **Google Cloud**.
2. **Database Configuration**: Make sure your cloud service supports the database type you're using (SQLite, PostgreSQL, MySQL, etc.).
3. **Environment Variables**: Store API keys securely in environment variables or secret management systems rather than hardcoding them in the script.

---

## Troubleshooting

1. **Database Connection Error**: Ensure that the database file is accessible and the connection string is correct.
2. **Gemini API Key Error**: If there’s an issue with the Gemini key, verify that you have the correct key and have set it up properly in the code.

---

## Contributing

If you'd like to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. You can also report issues or suggest improvements.

---

## License

This project is open-source and available under the MIT License.

---

## Acknowledgements

- **LlamaIndex**: For enabling SQL table indexing and retrieval.
- **Gemini LLM**: For providing state-of-the-art language models for text-to-SQL conversion.
- **Streamlit**: For the beautiful web interface.

---

This readme should help you set up, run, and customize the **NL2SQL Bot** app for generating SQL queries from natural language inputs.
