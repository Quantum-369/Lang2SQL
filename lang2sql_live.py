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
from utils import create_erd_diagram, list_catalog_schema_tables, process_llm_response_for_mermaid, mermaid,quick_analysis,create_advanced_sql,create_sql,self_correction,add_to_user_history,validate_and_correct_sql
from utils import load_data_from_query,process_llm_response_for_sql, get_enriched_database_schema,load_user_query_history
load_dotenv()

# sample streamlit
#st.write("Hi Harsha Congrats!")

# Set Page Config (Modern and Futuristic)
st.set_page_config(
    page_title="Lang2SQL",
    page_icon="📊",
    layout="wide",  # Wide layout for a more spacious, future-forward look
    initial_sidebar_state="collapsed"  # Minimalist approach for a clean, modern UI
)

# Apply sleek background color and typography
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #1c1c1c;  /* Dark futuristic background */
        color: #ffffff;  /* White text for contrast */
    }
    h1 {
        color: #00FF99;  /* Neon green for futuristic title */
        font-family: 'Roboto', sans-serif;
        font-size: 4em;  /* Larger title for more impact */
    }
    h6 {
        color: #d3d3d3;  /* Light gray for text */
        font-family: 'Arial', sans-serif;
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True
)

# The App - Futuristic Header with Soft Animations
st.markdown("<h1 style='text-align: center;'>Lang2SQL &#128640;</h1>", unsafe_allow_html=True)

st.markdown("<h6 style='text-align: center;'>Your futuristic assistant for translating natural language to SQL. Empowering business stakeholders, product managers, and intermediate coders to unlock data from traditional SQL databases effortlessly.</h6>", unsafe_allow_html=True)

# Adding the authentication
with open('authenticator.yml') as f:
    config = yaml.load(f, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, user_name = authenticator.login()
if authentication_status:
    # Logout button in the main area
    authenticator.logout('Logout', 'main')
    
    # Welcome message with proper string formatting and line breaks
    st.write(
        f"Welcome back, *{name}*! Your Lang2SQL assistant is ready to help you "
        "unlock data and turn natural language into actionable SQL queries. "
        "Let's explore the possibilities!"
    )
    
    # Sidebar with MySQL logo
    st.sidebar.image('C:\\Users\\harsh\\Downloads\\MS projects\\Lang2Sql\\artifacts\\MySQL.png')
    #selecting schema, table from Target database
    df=list_catalog_schema_tables()
    df_mysql=pd.DataFrame(df)
    df_mysql.columns=["schema_name","table_name","table_type"]
    # getting schema to table mapping for dynamically selecting only relevant tables for a given catalog and schema
    schema_table_mapping_df = df_mysql.groupby(["schema_name"]).agg({'table_name': lambda x: list(np.unique(x))}).reset_index()
    # Selecting the schema (without catalog)
    schema_candidate_list = df_mysql['schema_name'].unique().tolist()
    schema_candidate_list = [val for val in schema_candidate_list if val != "lang2sql"]
    schema = st.sidebar.selectbox("Select the schema", options=schema_candidate_list)

    # Selecting the tables within the selected schema
    table_candidate_list = schema_table_mapping_df[schema_table_mapping_df["schema_name"] == schema]["table_name"].values[0]
    tables_list = st.sidebar.multiselect("Select the table", options=["All"] + table_candidate_list)

    # Handling "All" selection for tables
    if "All" in tables_list:
        tables_list = table_candidate_list
    # Handling database
    database=schema

    if st.sidebar.checkbox(":orange[Proceed]"):   
        # Get enriched schema first before any analysis
        table_schema = get_enriched_database_schema(database, tables_list)     
        with st.expander(":red[View the ERD Diagram]"):
            if 'erd_response' not in st.session_state:
                st.session_state.erd_response = None
                st.session_state.mermaid_code = None

            if st.button("Generate ERD"):
                st.session_state.erd_response = create_erd_diagram(database, tables_list)
                st.session_state.mermaid_code = process_llm_response_for_mermaid(st.session_state.erd_response)
                mermaid(st.session_state.mermaid_code)
            
            elif st.button("Regenerate"):
                create_erd_diagram.clear()
                st.session_state.erd_response = create_erd_diagram(database, tables_list)
                st.session_state.mermaid_code = process_llm_response_for_mermaid(st.session_state.erd_response)
                mermaid(st.session_state.mermaid_code)
            
            # Display existing diagram if it exists
            elif st.session_state.mermaid_code is not None:
                mermaid(st.session_state.mermaid_code)
            else:
                st.info("Click 'Generate ERD' to create the diagram.")
# Quick Analysis
        st.markdown("<h2 style='text-align: left; color: red;'> Quick Analysis </h2>", unsafe_allow_html=True)
        with st.expander(":red[View the Section]"):
            quick_analysis_questions = quick_analysis(table_schema)
            if st.button("Need new ideas?"):
                quick_analysis.clear()
                quick_analysis_questions = quick_analysis(table_schema)
                questions = quick_analysis_questions['quick_analysis_questions']  # Updated access
                selected_question = st.selectbox("Select the question", options=questions)
                if st.checkbox("Analyze"):
                    st.write(f"##### {selected_question}")
                    # st.text(mermaid_code)
                    response_sql_qa = create_sql(selected_question,table_schema)
                    response_sql_qa = process_llm_response_for_sql(response_sql_qa)
                    
                    # Self-correction loop
                    flag, response_sql_qa = validate_and_correct_sql(selected_question,response_sql_qa,table_schema)
                    while flag != 'Correct':
                        flag, response_sql_qa = validate_and_correct_sql(selected_question,response_sql_qa,table_schema)

                    st.code(response_sql_qa)
                    col1,col2 = st.columns(2)
                    if col1.button("Query Sample Data"):
                        df_query = load_data_from_query(response_sql_qa)                        
                        col1.write(df_query)
                    
                    # Saving the Favourites    
                    # Adding session_state for favourite button
                    if 'fav_ind_qa' not in st.session_state:
                        st.session_state.fav_ind_qa = False

                    fav_ind_qa = col2.button("Save the query",key="ddd34-13")
                    if fav_ind_qa:
                        st.session_state.fav_ind_qa = True
                        add_to_user_history(name,selected_question,response_sql_qa,favourite_ind=True)
                        col2.write("Added to favourites!") 
                        

            else:
                questions = quick_analysis_questions['quick_analysis_questions']  # Updated access
                selected_question = st.selectbox("Select the question", options=questions)
                if st.checkbox("Analyze"):
                    st.write(f"##### {selected_question}")
                    # st.text(mermaid_code)
                    response_sql_qa = create_sql(selected_question,table_schema)
                    response_sql_qa = process_llm_response_for_sql(response_sql_qa)

                    # Self-correction loop
                    flag, response_sql_qa = validate_and_correct_sql(selected_question,response_sql_qa,table_schema)
                    while flag != 'Correct':
                        flag, response_sql_qa = validate_and_correct_sql(selected_question,response_sql_qa,table_schema)

                    st.code(response_sql_qa)
                    col1,col2 = st.columns(2)
                    if col1.button("Query Sample Data"):
                        df_query = load_data_from_query(response_sql_qa)
                        col1.write(df_query)

                    # Saving the Favourites    
                    # Adding session_state for favourite button
                    if 'fav_ind_qa_2' not in st.session_state:
                        st.session_state.fav_ind_qa_2 = False

                    fav_ind_qa_2 = col2.button("Save the query",key="dd34-13")
                    if fav_ind_qa_2:
                        st.session_state.fav_ind_qa_2 = True
                        add_to_user_history(name,selected_question,response_sql_qa,favourite_ind=True)
                        col2.write("Added to favourites!") 
############################################################################################################
        # Favourite Section
        st.markdown("<h2 style='text-align: left; color: red;'> Your Favourites </h2>", unsafe_allow_html=True)
        with st.expander(":red[View the Section]"):
            # Clear cache to ensure fresh data
            load_user_query_history.clear()
            
            # Load favorites
            fav_df = load_user_query_history(user_name=name)
            
            # Check if there are any favorites
            if fav_df.empty:
                st.info("You don't have any favorites yet. Save some queries to see them here!")
            else:
                # Display favorites count
                st.write(f"You have {len(fav_df)} saved favorites")
                
                # Get unique questions for dropdown
                fav_questions = fav_df['question'].unique().tolist()
                
                if fav_questions:
                    # Add unique key for selectbox
                    fav_question = st.selectbox(
                        "Select a saved query", 
                        options=fav_questions,
                        key="favorites_selectbox"
                    )
                    
                    # Get the matching query
                    matching_queries = fav_df[fav_df['question'] == fav_question]['query']
                    
                    if not matching_queries.empty:
                        fav_sql = matching_queries.values[0]
                        st.write(f"##### {fav_question}")
                        st.code(fav_sql)
                        
                        col1, col2 = st.columns(2)
                        
                        # Add unique key for button
                        if col1.button("Query Sample Data", key=f"fav_query_btn_{hash(fav_question)}"):
                            try:
                                fav_query = load_data_from_query(fav_sql)
                                col1.write(fav_query)
                            except Exception as e:
                                col1.error(f"Error executing query: {str(e)}")
############################################################################################################
        # Deep-Dive Analysis
        st.markdown("<h2 style='text-align: left; color: red;'> Deep-Dive Analysis </h2>", unsafe_allow_html=True)

        with st.expander(":red[View the Section]"):
            dd_question = st.text_area("Enter your question here..",key="dd-10",)

            generate_sql_1 = st.checkbox("Generate SQL",key="dd-11")
            if generate_sql_1:
                response_sql_1 = create_sql(dd_question,table_schema)
                response_sql_1 = process_llm_response_for_sql(response_sql_1)

                # Self-correction loop
                flag, response_sql_1 = validate_and_correct_sql(dd_question,response_sql_1,table_schema)
                while flag != 'Correct':
                    flag, response_sql_1 = validate_and_correct_sql(dd_question,response_sql_1,table_schema)

                st.code(response_sql_1)

                col1, col2 = st.columns(2)                

                query_sample_data_1 = col1.checkbox("Query Sample Data",key="dd-102")
                if query_sample_data_1:
                    df_query_1 = load_data_from_query(response_sql_1)
                    col1.write(df_query_1)

                # Saving the Favourites    
                # Adding session_state for favourite button
                if 'fav_ind_1' not in st.session_state:
                    st.session_state.fav_ind_1 = False

                fav_ind_1 = col2.button("Save the query",key="dd-13")
                if fav_ind_1:
                    st.session_state.fav_ind_1 = True
                    add_to_user_history(name,dd_question,response_sql_1,favourite_ind=True)
                    col2.write("Added to favourites!")


                build_1 = col1.checkbox("Build on top of this result?",key="dd-21")
                if build_1:
                    dd_question_2 = st.text_area("Enter your question here..",key="dd-23",)

                    generate_sql_2 = st.checkbox("Generate SQL",key="dd-24")
                    if generate_sql_2:
                        response_sql_2 = create_advanced_sql(dd_question_2,response_sql_1,table_schema)
                        response_sql_2 = process_llm_response_for_sql(response_sql_2)

                        # Self-correction loop
                        flag, response_sql_2 = validate_and_correct_sql(dd_question_2,response_sql_2,table_schema)
                        while flag != 'Correct':
                            flag, response_sql_2 = validate_and_correct_sql(dd_question_2,response_sql_2,table_schema)

                        st.code(response_sql_2)

                        col1, col2 = st.columns(2)
                        query_sample_data_2 = col1.checkbox("Query Sample Data",key="dd-25")
                        if query_sample_data_2:
                            df_query_2 = load_data_from_query(response_sql_2)
                            col1.write(df_query_2)
                                                
                        
                        # Saving the Favourites    
                        # Adding session_state for favourite button
                        if 'fav_ind_2' not in st.session_state:
                            st.session_state.fav_ind_2 = False

                        fav_ind_2 = col2.button("Save the query",key="dd3-133")
                        if fav_ind_2:
                            st.session_state.fav_ind_2 = True
                            add_to_user_history(name,dd_question_2,response_sql_2,favourite_ind=True)
                            col2.write("Added to favourites!")  
        

else:
    st.write("Please login to continue.")