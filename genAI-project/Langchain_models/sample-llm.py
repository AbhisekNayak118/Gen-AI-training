import os 
from typing import Optional 
from pydantic import BaseModel, Field 
 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser 
from tenacity import retry, stop_after_attempt, wait_exponential 

 
# ========================= 
# 1. SETUP (ENV + MODEL) 
# ========================= 
 
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  
 
endpoint = HuggingFaceEndpoint( 
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct", 
    task="text-generation",     
    max_new_tokens=200, 
    temperature=0.7 
) 
 
model = ChatHuggingFace(llm=endpoint) 
 
# ========================= 
# 2. PYDANTIC OUTPUT PARSER 
# ========================= 
 
class AnswerSchema(BaseModel): 
    definition: str = Field(description="Definition of the concept") 
    example: Optional[str] = Field(description="Example if applicable") 
 
parser = PydanticOutputParser(pydantic_object=AnswerSchema) 
 
# ========================= 
# 3. PROMPT TEMPLATE 
# ========================= 
 
prompt = PromptTemplate( 
input_variables=["question"], 
partial_variables={"format_instructions": parser.get_format_instructions()}, 
template=""" 
You are an expert AI tutor. 
 
Answer the following question: 
{question} 
 
{format_instructions} 
""" 
) 
 
# ========================= 
# 4. LCEL CHAIN (MODERN) 
# ========================= 
 
chain = prompt | model | parser 
 
# ========================= 
# 5. RETRY MECHANISM 
# ========================= 
 
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10)) 
def safe_invoke(question: str): 
    try: 
        response = chain.invoke({ 
        "question": question 
        }) 
        return response 
    
    except Exception as e: 
        print("Error occurred:", str(e)) 
        raise e 
 
# ========================= 
# 6. MAIN EXECUTION 
# ========================= 
 
if __name__ == "__main__": 
    question = "Apple iphone 17 pro max 2024 features and specifications ?"
 
    try: 
        result = safe_invoke(question)  
        print("\n Parsed Output:") 
        print("-----------------------------------------------------------------")
        print("Definition:", result.definition) 
        print("-----------------------------------------------------------------")
        print("Example:", result.example) 
        print("-----------------------------------------------------------------")
    
    except Exception as final_error: 
        print(" Failed after retries:", str(final_error) 
)
