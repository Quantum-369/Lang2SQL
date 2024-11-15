import os, sys 
import pandas as pd 
import numpy as np
import streamlit as st
import sqlparse
from collections import OrderedDict, Counter
from github import Github
import mysql.connector
import streamlit_authenticator as stauth
import yaml 
from yaml.loader import SafeLoader
from dotenv import load_dotenv
load_dotenv()
import mysql.connector
import os
from typing import Union, Optional
from datetime import datetime
import streamlit.components.v1 as components
from typing import List
from typing import Dict

# LLM libraries
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains.llm import LLMChain
from langchain_anthropic import ChatAnthropic

# Function to load the user_query_history table
@st.cache_data
def load_user_query_history(user_name):
    # Getting the sample details of the selected table
    connection = mysql.connector.connect(
        host = os.getenv('MYSQL_HOST'),
        user = os.getenv('MYSQL_USER'),
        password = os.getenv('MYSQL_PASSWORD'),
        database = os.getenv('MYSQL_DATABASE'))

    query = f"SELECT * FROM lang2sql.lang2sql_user_query_history WHERE user_name = '{user_name}' AND timestamp > current_date - 20"
    df = pd.read_sql(sql=query,con=connection)
    return df


# Function to list all the catalog, schema and tables present in the database 
@st.cache_data
def list_catalog_schema_tables(database=None):
    # Establishing the connection to the MySQL database
    connection = mysql.connector.connect(
        host = os.getenv('MYSQL_HOST'),
        user = os.getenv('MYSQL_USER'),
        password = os.getenv('MYSQL_PASSWORD'),
        database = os.getenv('MYSQL_DATABASE')
    )
    with connection.cursor() as cursor:
        query = """
        SELECT 
            TABLE_SCHEMA as schema_name,
            TABLE_NAME as table_name,
            TABLE_TYPE as table_type    -- Removed trailing comma here
        FROM 
            information_schema.TABLES
        WHERE 
            TABLE_SCHEMA NOT IN ('information_schema', 'performance_schema', 
                               'mysql', 'sys')
        """
        
        # If specific database is provided, add WHERE clause
        if database:
            query += f" AND TABLE_SCHEMA = '{database}'"
            
        query += " ORDER BY TABLE_SCHEMA, TABLE_NAME"
        
        # Execute query and fetch results
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Create DataFrame
        df = pd.DataFrame(
            results,
            columns=['schema_name', 'table_name', 'table_type']  # Removed trailing comma here
        )
        
        return df
        

# Function to create enriched database schema details for the Prompt
@st.cache_data        
def get_enriched_database_schema(database: str, tables_list: List[str]) -> str:
    """
    Create enriched database schema details including table structure, sample data,
    and categorical field information for the specified tables.
    
    Parameters:
    -----------
    database : str
        Name of the database to connect to
    tables_list : List[str]
        List of table names to process
    
    Returns:
    --------
    str
        Formatted string containing enriched schema information
    """
    table_schema = ""

    try:
        # Establishing MySQL connection
        connection = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST'),
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASSWORD'),
            database=database
        )

        # Iterating through each selected table to get enriched schema information
        for table in tables_list:
            with connection.cursor(dictionary=True) as cursor:
                # Getting the Schema for the table
                query = f"SHOW CREATE TABLE `{database}`.`{table}`"
                cursor.execute(query)
                result = cursor.fetchone()
                stmt = result['Create Table'].split("ENGINE=")[0]

                # Identifying Categorical Columns (Assuming VARCHAR columns as categories)
                query = f"DESCRIBE `{database}`.`{table}`"
                df = pd.read_sql(sql=query, con=connection)
                
                # Convert bytes to string in the Type column if necessary
                if df['Type'].dtype == 'object':
                    df['Type'] = df['Type'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))
                
                # Filter for string-type columns
                string_cols = df[
                    df['Type'].str.lower().str.contains('varchar|char|enum|set', na=False)
                ]['Field'].tolist()

                # Generating SQL for distinct values in categorical columns
                if string_cols:
                    sql_parts = []
                    for col in string_cols:
                        sql_parts.append(
                            f"SELECT '{col}' AS column_name, "
                            f"COUNT(DISTINCT `{col}`) AS cnt, "
                            f"GROUP_CONCAT(DISTINCT `{col}` SEPARATOR ', ') AS `values` "
                            f"FROM `{database}`.`{table}`"
                        )
                    sql_distinct = " UNION ALL ".join(sql_parts)

                    # Getting categorical field information
                    df_categories = pd.read_sql(sql=sql_distinct, con=connection)
                    # Only keep columns with 20 or fewer distinct values
                    df_categories = df_categories[df_categories['cnt'] <= 20].drop(columns='cnt')
                    df_categories_string = (
                        df_categories.to_string(index=False) 
                        if not df_categories.empty 
                        else "No Categorical Fields"
                    )
                else:
                    df_categories_string = "No Categorical Fields"

                # Getting sample rows from the table
                query = f"SELECT * FROM `{database}`.`{table}` LIMIT 3"
                sample_df = pd.read_sql(sql=query, con=connection)
                
                # Convert any bytes objects in the sample data
                for column in sample_df.columns:
                    if sample_df[column].dtype == 'object':
                        sample_df[column] = sample_df[column].apply(
                            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
                        )
                
                sample_rows = sample_df.to_string(index=False)

                # Format the table details
                table_details = (
                    f"Table: {table}\n"
                    f"{stmt}\n\n"
                    f"Sample Rows:\n{sample_rows}\n\n"
                    f"Categorical Fields:\n{df_categories_string}\n"
                    f"{'-' * 80}\n"  # Separator between tables
                )
                
                # Append table schema, sample rows, and categorical fields
                table_schema += table_details

        return table_schema

    except mysql.connector.Error as err:
        raise Exception(f"Database error: {err}")
    except Exception as e:
        raise Exception(f"Error getting enriched schema: {e}")
        

# Function to render the mermaid diagram
def process_llm_response_for_mermaid(response: str) -> str:
    # Extract the Mermaid code block from the response
    start_idx = response.find("```mermaid") + len("```mermaid")
    end_idx = response.find("```", start_idx)
    mermaid_code = response[start_idx:end_idx].strip()

    return mermaid_code

# Function to render the sql code
def process_llm_response_for_sql(response: str) -> str:
    # Extract the Mermaid code block from the response
    start_idx = response.find("```sql") + len("```sql")
    end_idx = response.find("```", start_idx)
    sql_code = response[start_idx:end_idx].strip()

    return sql_code


def mermaid(code: str) -> None:
    # Escaping backslashes for special characters in the code
    code_escaped = code.replace("\\", "\\\\").replace("`", "\\`")
    
    # components.html(
    #     f"""
    #     <div id="mermaid-container" style="width: 100%; height: 100%; overflow: auto;">
    #         <pre class="mermaid">
    #             {code_escaped}
    #         </pre>
    #     </div>

    #     <script type="module">
    #         import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    #         mermaid.initialize({{ startOnLoad: true }});
    #     </script>
    #     """,
    #     height=800  # You can adjust the height as needed
    # )       
    components.html(
        f"""
        <div id="mermaid-container" style="width: 100%; height: 800px; overflow: auto;">
            <pre class="mermaid">
                {code_escaped}
            </pre>
        </div>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=800  # You can adjust the height as needed
    )

# Function to create the ERD diagram for the selected schema and tables 
@st.experimental_fragment      
@st.cache_data 
def create_erd_diagram(database, tables_list):
    """
    Create an ERD diagram for specified MySQL tables using Mermaid syntax.
    
    Parameters:
    -----------
    database : str
        Name of the database
    tables_list : list
        List of table names to include in the ERD
    
    Returns:
    --------
    str
        Mermaid syntax for the ERD diagram
    """
    table_schema = {}

    # Establishing MySQL connection
    connection = mysql.connector.connect(
        host=os.getenv('MYSQL_HOST'),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        database=database
    )
    # Iterating through each selected table to get column information
    for table in tables_list:
        with connection.cursor() as cursor:
            # Query to get column information
            query = f"""
            SELECT 
                COLUMN_NAME,
                COLUMN_TYPE,
                IS_NULLABLE,
                COLUMN_KEY
            FROM 
                information_schema.COLUMNS 
            WHERE 
                TABLE_SCHEMA = '{database}'
                AND TABLE_NAME = '{table}'
            ORDER BY 
                ORDINAL_POSITION
            """
            
            cursor.execute(query)
            columns = cursor.fetchall()
            
            # Format column information
            cols_dict = []
            for col in columns:
                col_name = col[0]
                col_type = col[1]
                is_nullable = "NULL" if col[2] == "YES" else "NOT NULL"
                key_type = f"[{col[3]}]" if col[3] else ""
                cols_dict.append(f"{col_name} : {col_type} {is_nullable} {key_type}".strip())
            
            table_schema[table] = cols_dict
    print(table_schema)

    # Generating the mermaid code for the ERD diagram
    ### Defining the prompt template
    template_string = """ 
    You are an expert in creating ERD diagrams (Entity Relationship Diagrams) for databases. 
    You have been given the task to create an ERD diagram for the selected tables in the database. 
    The ERD diagram should contain the tables and the columns present in the tables. 
    You need to generate the Mermaid code for the complete ERD diagram.
    Make sure the ERD diagram is clear and easy to understand with proper relationships details.

    The selected tables in the database are given below (delimited by ##) in the dictionary format: Keys being the table names and values being the list of columns and their datatype in the table.

    ##
    {table_schema}
    ##

    Before generating the mermaid code, validate it and make sure it is correct and clear.     
    Give me the final mermaid code for the ERD diagram after proper analysis.
    """

    prompt_template = PromptTemplate.from_template(template_string)

    ### Defining the LLM chain
    llm_chain = LLMChain(
        llm=ChatAnthropic(model="claude-3-5-haiku-20241022",temperature=0),
        prompt=prompt_template
    )

    response =  llm_chain.invoke({"table_schema":table_schema})
    output = response['text']    
    return output

@st.experimental_fragment
@st.cache_data
def quick_analysis(table_schema: str) -> Dict:
    """
    Generate analytical questions based on database schema using Claude.
    
    Parameters:
    -----------
    table_schema : str
        Database schema information including table structures and sample data
    
    Returns:
    --------
    Dict
        Structured output containing generated analysis questions
    """
    
    # Define the output schema
    output_schema = ResponseSchema(
        name="quick_analysis_questions",
        description="Generated analytical questions based on the database schema. "
                   "Questions should be business-focused and SQL-answerable."
    )
    
    # Create output parser
    output_parser = StructuredOutputParser.from_response_schemas([output_schema])
    format_instructions = output_parser.get_format_instructions()
    
    # Define prompt template for MySQL-specific analysis
    template_string = """You are an expert MySQL data analyst. Your task is to generate the top 5 analytical questions based on the provided schema.

    Requirements for questions:
    1. Focus on relationships between tables using proper JOINs
    2. Include questions about:
       - Business metrics and KPIs
       - Trends and patterns
       - Customer behavior
       - Product performance
       - Employee performance
    3. Questions should be answerable using MySQL queries
    4. Consider practical business insights that managers need daily
    5. Use proper table relationships based on the schema

    SCHEMA:
    ##
    {table_schema}
    ##

    The output must be in this JSON format:
    {format_instructions}

    IMPORTANT:
    - Generate exactly 5 questions
    - Questions should require SQL with proper table joins (no CROSS JOINs)
    - Focus on meaningful business insights
    - Consider the available columns and relationships
    - Questions should be clear and specific
    """
    
    # Create prompt template
    prompt = PromptTemplate.from_template(template_string)
    
    # Initialize Claude model
    llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022",
        temperature=0.2,  # Slight creativity for question generation
        max_tokens=1024
    )
    
    # Create LLM chain
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False
    )
    
    try:
        # Generate and parse response
        response = chain.invoke({
            "table_schema": table_schema,
            "format_instructions": format_instructions
        })
        
        if not isinstance(response, dict) or 'text' not in response:
            raise ValueError("Unexpected response format from LLM")
            
        # Parse the response text to extract questions
        try:
            # First try to parse as JSON using the output parser
            parsed_output = output_parser.parse(response['text'])
            if isinstance(parsed_output, dict) and 'quick_analysis_questions' in parsed_output:
                return {'quick_analysis_questions': parsed_output['quick_analysis_questions']}
        except:
            # Fallback: Extract questions directly from text
            questions = [
                q.strip() for q in response['text'].split('\n')
                if q.strip() and not q.startswith('{') and not q.endswith('}')
            ]
            return {'quick_analysis_questions': questions[:5]}
            
    except Exception as e:
        raise Exception(f"Error generating analysis questions: {str(e)}")
# Function to create SQL code for the selected question and return the data from the database
@st.experimental_fragment
@st.cache_data
def create_sql(question: str, table_schema: str) -> str:
    """
    Generate SQL queries from natural language questions using Claude LLM.
    
    Parameters:
    -----------
    question : str
        Natural language question to convert to SQL
    table_schema : str
        Database schema information including table structures and sample data
    
    Returns:
    --------
    str
        Generated SQL query
    """
    
    # Define the prompt template for SQL generation
    template_string = """You are a expert data engineer working with a MySQL environment.
    Your task is to generate a working SQL query based on the natural language question.
    
    Follow these rules when generating SQL:
    1. During joins, use table aliases for clarity (e.g., c.customer_id)
    2. For string values, ensure they are enclosed in quotes
    3. When concatenating non-string columns, cast them to string using CAST() or CONVERT()
    4. For date comparisons with strings, use proper date casting
    5. For string columns, only use values that match the categorical values provided in the schema
    6. Make column references unambiguous by using table aliases
    7. Include proper table name with database prefix (e.g., database.table_name)
    
    SCHEMA:
    {table_schema}
    
    QUESTION:
    {question}
    
    IMPORTANT: Return ONLY the SQL query without any explanations or markdown formatting.
    """
    
    # Create prompt template
    prompt = PromptTemplate.from_template(template_string)
    
    # Initialize Claude model
    llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022",  
        temperature=0,  # Use 0 for consistent, deterministic outputs
        max_tokens=1024  # Adjust based on your needs
    )
    
    # Create LLM chain
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False  # Set to True for debugging
    )
    
    try:
        # Generate SQL using the chain
        response = chain.invoke({
            "question": question,
            "table_schema": table_schema
        })
        
        # Extract SQL from response
        sql = response['text'].strip()
        
        # Remove any markdown code block formatting if present
        sql = sql.replace('```sql', '').replace('```', '').strip()
        
        return sql
        
    except Exception as e:
        raise Exception(f"Error generating SQL: {str(e)}")

# Function to create SQL code for the selected question and return the data from the database
@st.experimental_fragment
@st.cache_data

def create_advanced_sql(question: str, sql_code: str, table_schema: str) -> str:
    """
    Generate advanced SQL queries by building upon existing SQL code and schema information.
    
    Parameters:
    -----------
    question : str
        The analytical question to be answered
    sql_code : str
        Existing SQL code to be used as a base
    table_schema : str
        Database schema information including table structures and sample data
    
    Returns:
    --------
    str
        Generated SQL query incorporating the base query in a CTE
    """
    
    # Define prompt template for MySQL-specific advanced analysis
    template_string = """You are an expert MySQL data engineer. Your task is to enhance an existing SQL query to answer a more complex analytical question.

    Guidelines:
    1. Wrap the existing SQL code in a Common Table Expression (CTE) named 'MASTER'
    2. Do not modify the original SQL code within the CTE
    3. Build upon the MASTER CTE to answer the new question
    4. Only join additional tables if necessary for the new question
    5. Ensure proper table aliasing and column references
    6. Use appropriate aggregations and grouping as needed
    7. Consider performance optimization in your query design

    Base SQL Code:
    ##
    {sql_code}
    ##

    Database Schema:
    ##
    {table_schema}
    ##

    New Analysis Question:
    ##
    {question}
    ##

    IMPORTANT:
    - Return ONLY the SQL query without any explanations
    - Ensure proper table prefixes/aliases
    - Use appropriate JOIN types (INNER, LEFT, etc.)
    - Handle NULL values appropriately
    - Return clean, formatted SQL code
    """
    
    # Create prompt template
    prompt = PromptTemplate.from_template(template_string)
    
    # Initialize Claude model
    llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022",
        temperature=0,  # Use 0 for consistent SQL generation
        max_tokens=2048  # Increased for complex queries
    )
    
    # Create LLM chain
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False
    )
    
    try:
        # Generate enhanced SQL query
        response = chain.invoke({
            "sql_code": sql_code,
            "question": question,
            "table_schema": table_schema
        })
        
        if not isinstance(response, dict) or 'text' not in response:
            raise ValueError("Unexpected response format from LLM")
        
        # Extract and clean SQL code
        sql = response['text']
        
        # Remove any markdown code block formatting
        sql = sql.replace('```sql', '').replace('```', '').strip()
        
        # Validate basic SQL structure
        if not sql.lower().startswith('with master as'):
            # Add WITH clause if missing
            sql = f"WITH MASTER AS (\n{sql_code}\n)\n{sql}"
        
        # Format the SQL for readability
        formatted_sql = format_sql(sql)
        
        return formatted_sql
        
    except Exception as e:
        raise Exception(f"Error generating advanced SQL: {str(e)}")

def format_sql(sql: str) -> str:
    """
    Format SQL query for better readability.
    
    Parameters:
    -----------
    sql : str
        Raw SQL query
        
    Returns:
    --------
    str
        Formatted SQL query
    """
    try:
        # Basic SQL formatting
        formatted = sql.replace('\n\n', '\n')  # Remove extra newlines
        
        # Add proper indentation for WITH clause
        if formatted.lower().startswith('with'):
            lines = formatted.split('\n')
            indented = []
            indent_level = 0
            
            for line in lines:
                line = line.strip()
                
                # Adjust indent level based on content
                if any(line.lower().startswith(word) for word in ['with', 'select', 'from', 'where', 'group by', 'having', 'order by']):
                    indent_level = 1
                elif line.startswith('('):
                    indent_level += 1
                elif line.startswith(')'):
                    indent_level -= 1
                
                # Add indentation
                indented.append('    ' * indent_level + line)
            
            formatted = '\n'.join(indented)
            
        return formatted
        
    except Exception as e:
        # If formatting fails, return original SQL
        return sql

# Usage example
def test_advanced_sql():
    # Example inputs
    base_sql = """
    SELECT 
        c.country,
        COUNT(o.orderId) as order_count
    FROM customer c
    JOIN orders o ON c.custId = o.custId
    GROUP BY c.country
    """
    
    question = "What is the average order value by country for countries with more than 10 orders?"
    
    schema = """
    CREATE TABLE customer (
        custId INT PRIMARY KEY,
        country VARCHAR(50)
    );
    
    CREATE TABLE orders (
        orderId INT PRIMARY KEY,
        custId INT,
        total_amount DECIMAL(10,2),
        FOREIGN KEY (custId) REFERENCES customer(custId)
    );
    """
    
    try:
        # Generate advanced SQL
        result = create_advanced_sql(question, base_sql, schema)
        print("Generated SQL:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")



# Function to load data from the database given the SQL query
@st.experimental_fragment
@st.cache_data
def load_data_from_query(query: str, limit: Optional[int] = 1000) -> pd.DataFrame:
    """
    Execute SQL query and return results as a pandas DataFrame.
    
    Parameters:
    -----------
    query : str
        SQL query to execute
    limit : int, optional
        Maximum number of rows to return (default: 1000)
    
    Returns:
    --------
    pd.DataFrame
        Query results as a DataFrame
    """
    try:
        # Establish MySQL connection using environment variables
        connection = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST'),
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASSWORD'),
            database=os.getenv('MYSQL_DATABASE')
        )

        # Clean query and add limit if not present
        query = query.strip().rstrip(';')
        if limit and 'LIMIT' not in query.upper():
            query = f"{query} LIMIT {limit}"

        # Execute query and return results as DataFrame
        df = pd.read_sql(query, connection)
        
        return df
        
    except Exception as e:
        raise Exception(f"Error executing query: {str(e)}")    

# Function to validate if self-correction is needed for the generated SQL query
@st.experimental_fragment
def self_correction(query):
    error_msg = ""

    try:
        df = load_data_from_query(query)
        # print(df.shape)
        # df.head()
        error_msg += "Successful"
    except Exception as e:
        error_msg += str(e)
    
    if error_msg == "Successful":
        return error_msg
    else:
        # print("There is error")
        # print(error_msg)
        return error_msg

# Function to validate and self-correct generated SQL query    
@st.experimental_fragment
def correct_sql(
    question: str,
    sql_code: str,
    table_schema: str,
    error_msg: str
) -> str:
    """
    Correct SQL query based on error message and schema information.
    
    Parameters:
    -----------
    question : str
        Original analysis question
    sql_code : str
        SQL query to correct
    table_schema : str
        Database schema information
    error_msg : str
        Error message from failed query execution
        
    Returns:
    --------
    str
        Corrected SQL query
    """
    
    # Define prompt template for SQL correction
    template_string = """You are an expert MySQL data engineer. Your task is to fix the SQL query based on the provided error message.

    Guidelines:
    1. Focus on fixing the specific error provided
    2. Ensure proper table and column references
    3. Use correct MySQL syntax
    4. Maintain the original query intent
    5. Keep all necessary JOIN conditions
    
    Database Schema:
    ##
    {table_schema}
    ##
    
    Original Question:
    ##
    {question}
    ##
    
    Current SQL Code:
    ##
    {sql_code}
    ##
    
    Error Message:
    ##
    {error_msg}
    ##
    
    IMPORTANT: Return ONLY the corrected SQL query without any explanations.
    """
    
    # Create prompt
    prompt = PromptTemplate.from_template(template_string)
    
    # Initialize Claude model
    llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022",
        temperature=0,  # Use 0 for consistent corrections
        max_tokens=1024
    )
    
    # Create LLM chain
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False
    )
    
    try:
        # Generate corrected SQL
        response = chain.invoke({
            "question": question,
            "sql_code": sql_code,
            "table_schema": table_schema,
            "error_msg": error_msg
        })
        
        # Extract SQL from response
        if isinstance(response, dict) and 'text' in response:
            corrected_sql = response['text'].strip()
            # Remove any markdown code block formatting
            corrected_sql = corrected_sql.replace('```sql', '').replace('```', '').strip()
            return corrected_sql
        else:
            raise ValueError("Unexpected response format from LLM")
            
    except Exception as e:
        raise Exception(f"Error correcting SQL: {str(e)}")

# Final function to validate and self-correct
def validate_and_correct_sql(question,query,table_schema):
    error_msg = self_correction(query)

    if error_msg == "Successful":
        # print("Query is successful")
        return "Correct",query
    else:
        modified_query = correct_sql(question,query,table_schema,error_msg=error_msg)
        return "Incorrect",modified_query
    

# Add the selected question to the user history
@st.experimental_fragment


def add_to_user_history(
    user_name: str,
    question: str,
    query: str,
    favourite_ind: Union[bool, int]
) -> Optional[str]:
    """
    Add a query to user history in the database with improved error handling and validation.
    
    Parameters:
    -----------
    user_name : str
        Name or ID of the user
    question : str
        Original question that prompted the query
    query : str
        SQL query to save
    favourite_ind : Union[bool, int]
        Indicator if this is a favorite query
        
    Returns:
    --------
    Optional[str]
        Success message if operation completed
    """
    
    connection = None
    
    try:
        # Convert boolean to integer if necessary
        favourite_ind = 1 if favourite_ind is True else 0 if favourite_ind is False else favourite_ind
        
        # Input validation
        if not user_name or not question or not query:
            raise ValueError("User name, question, and query are required fields")
            
        # Establish database connection
        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE")
        )
        
        cursor = connection.cursor(prepared=True)
        
        # Define insert query with proper schema prefix
        insert_query = """
        INSERT INTO lang2sql.lang2sql_user_query_history (
            user_name,
            timestamp,
            question,
            query,
            favourite_ind
        ) VALUES (
            %s,
            CURRENT_TIMESTAMP(),
            %s,
            %s,
            %s
        )
        """
        
        # Execute query with parameters
        cursor.execute(
            insert_query,
            (user_name, question, query, favourite_ind)
        )
        
        # Commit transaction
        connection.commit()
        
        return "Success"
        
    except mysql.connector.Error as e:
        if connection:
            connection.rollback()
        st.error(f"Database error while saving favorite: {str(e)}")
        return None
        
    except Exception as e:
        if connection:
            connection.rollback()
        st.error(f"Error saving favorite: {str(e)}")
        return None
    