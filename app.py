import openai
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Access the API keys
openai_api_key = os.getenv('OPENAI_API_KEY')
serpapi_api_key = os.getenv('SERP_API_KEY')

# Set the API key for OpenAI
openai.api_key = openai_api_key

# Initialize SerpAPI wrapper
search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

# Define tools for Langchain agent
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events or company information"
    )
]

# Function to calculate the number of tokens in a string
def count_tokens(text):
    # Using a rough estimate, as exact token count depends on the model's tokenizer
    return len(text.split())

# Define PromptTemplate for detailed company information
basic_info_template = PromptTemplate.from_template(
    "Retrieve detailed information for the company {company_name}. The details should include:\n"
    "- Company Website\n"
    "- Contact Number\n"
    "- Email\n"
    "- Headquarters Location\n"
    "- Postal Code\n"
    "- Products\n"
    "- Competitors\n"
    "- Career Portal URL\n"
    "Please ensure the information is accurate and up-to-date."
)

# Function to get detailed company information
def get_company_info(company_name):
    try:
        # Get detailed information using the prompt template
        detailed_info_prompt = basic_info_template.format(company_name=company_name)
        
        # Check the number of tokens in the prompt
        prompt_tokens = count_tokens(detailed_info_prompt)
        max_allowed_tokens = 4097 - 256  # Subtracting 256 tokens for completion
        
        # Log the prompt length for debugging
        st.write(f"Initial prompt length: {prompt_tokens} tokens.")
        
        # Truncate the prompt if it's too long
        if prompt_tokens > max_allowed_tokens:
            detailed_info_prompt = " ".join(detailed_info_prompt.split()[:max_allowed_tokens])
            st.write(f"Prompt was truncated to fit the token limit: {count_tokens(detailed_info_prompt)} tokens.")
        
        detailed_info_response = agent.run(detailed_info_prompt)
        
        # Ensure the response is not empty
        if not detailed_info_response:
            return f"No valid response received for {company_name}."
        
        # Format the response into bullet points
        response_lines = detailed_info_response.split(", ")
        formatted_response = []
        
        for line in response_lines:
            if any(keyword in line for keyword in ["Website", "Number", "Email", "Location", "Code", "Products", "Competitors", "Portal"]):
                formatted_response.append(f"- {line}")
            else:
                # Append to the last element if it doesn't start with a keyword
                if formatted_response:
                    formatted_response[-1] += f", {line}"
                else:
                    formatted_response.append(line)
        
        return formatted_response
    except Exception as e:
        return [f"Error retrieving information for {company_name}: {e}"]

# Initialize Langchain agent with max tokens
llm = OpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=256)  # Lowered max tokens to 256
agent = initialize_agent(
    tools=tools, 
    llm=llm,
    agent_type="zero-shot-react-description", 
    verbose=True
)

# Streamlit app
def main():
    st.title("Company Information Retrieval")

    company_name = st.text_input("Enter the company name:")
    if st.button("Get Information"):
        if company_name:
            company_info = get_company_info(company_name)
            
            if company_info:
                st.write(f"### Company Information for {company_name}")
                for line in company_info:
                    st.write(line)
            else:
                st.write("No information found.")
        else:
            st.write("Please enter a company name.")

if __name__ == "__main__":
    main()
