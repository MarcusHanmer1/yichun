# Cplt and web help used

import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages.ai import AIMessageChunk

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Google API Key not found. Please set it in your .env file.")

def get_vector_store_from_pdf(_pdf_file):
    """ Processes the PDF file and returns a FAISS vector store. """
    if _pdf_file is not None:
        try:
            pdf_reader = PdfReader(_pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            return vector_store
        except Exception as e:
            print(f"Error in get_vector_store_from_pdf: {e}")
            return None

def get_text_from_file(_example_file):
    """Reads text from an uploaded file (PDF or TXT)."""
    if _example_file is None:
        return ""
    try:
        if _example_file.type == "text/plain":
            return _example_file.read().decode("utf-8")
        elif _example_file.type == "application/pdf":
            pdf_reader = PdfReader(_example_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error in get_text_from_file: {e}")
        return ""
    return ""

def _string_to_stream(s: str):
    """Wraps a string in a generator to mimic an LLM stream."""
    yield AIMessageChunk(content=s)

def run_pdf_mode_pipeline(user_prompt, vector_store, example_text, include_answer_key):
    """
    Runs the full 3-agent (Generate, Critique, Refine) RAG pipeline.
    Returns a STREAM of the final, refined text.
    """
    try:
        retriever = vector_store.as_retriever(search_k=7)
        relevant_docs = retriever.invoke(user_prompt)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        
        if include_answer_key:
            answer_key_request = "You MUST include a detailed, step-by-step answer key. Separate the questions from the answer key with the tag '---ANSWER KEY---'."
        else:
            answer_key_request = "Do NOT include an answer key."

        generator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        
        generator_prompt_template_str = """
        You are an expert exam question generator. Your task is to create a set of questions (a "v1 draft") based on the user's request.
        You MUST use the provided context from the course material.
        You MUST match the style, tone, and difficulty of the example questions.
        {answer_key_request}

        **CONTEXT FROM COURSE MATERIAL:**
        {context}
        **EXAMPLE QUESTIONS (Follow this style):**
        {examples}
        **USER REQUEST:**
        {request}

        **V1 DRAFT (You MUST format your entire response using rich Markdown. Use lists, bolding, and LaTeX for any mathematical expressions):**
        """
        
        generator_prompt = PromptTemplate.from_template(generator_prompt_template_str)
        generator_final_prompt = generator_prompt.format(
            context=context_text,
            examples=example_text,
            request=user_prompt,
            answer_key_request=answer_key_request
        )
        
        generator_response = generator_llm.invoke(generator_final_prompt)
        v1_draft = generator_response.content

        marker_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        
        marker_prompt_template_str = """
        You are an expert 'Marker' agent, a harsh and strict university examiner.
        Your job is to write an internal critique of the 'v1 Draft' questions.
        You must be BRUTALLY HONEST. The user will NOT see this. Your critique will be used to fix the draft.
        Focus on 100% factual accuracy of the questions AND the answer key.

        **THE RUBRIC (Be harsh):**
        1.  **Factual Accuracy:** Are the questions AND the answer key 100% correct according to the CONTEXT? Point out every single error.
        2.  **Prompt Relevance:** Do the questions directly address the USER'S REQUEST?
        3.  **Style Match:** Do the questions match the style of the EXAMPLE QUESTIONS?
        4.  **Answer Key (if requested):** Was the instruction '{answer_key_request}' followed perfectly? Is the answer key detailed and correct?

        **--- INPUTS FOR YOUR REVIEW ---**
        1. CONTEXT FROM COURSE MATERIAL: {context}
        2. EXAMPLE QUESTIONS (The style to match): {examples}
        3. USER'S ORIGINAL REQUEST: {request}
        4. THE 'V1 DRAFT' (Your target for critique): {v1_draft}

        **--- YOUR TASK ---**
        Provide a concise, constructive, and harsh critique. List every single error you find.
        If there are no errors, simply write "PERFECT".
        """

        marker_prompt = PromptTemplate.from_template(marker_prompt_template_str)
        marker_final_prompt = marker_prompt.format(
            context=context_text,
            examples=example_text,
            request=user_prompt,
            v1_draft=v1_draft,
            answer_key_request=answer_key_request
        )

        critique_response = marker_llm.invoke(marker_final_prompt)
        critique_content = critique_response.content
        
        if critique_content.strip().upper() == "PERFECT":
            return _string_to_stream(v1_draft)
        else:
            refiner_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
            
            refiner_prompt_template_str = """
            You are an expert 'Refiner' agent. Your job is to rewrite a 'v1 Draft' to fix all issues from a 'Critique'.
            You must fix every point in the critique. Do not add your own opinions.
            You MUST preserve the original format, including the '---ANSWER KEY---' separator.

            **--- INPUTS ---**
            
            1. USER'S ORIGINAL REQUEST: {request}
            
            2. THE 'V1 DRAFT' (The original version):
            {v1_draft}
            
            3. THE 'HARSH CRITIQUE' (The issues you must fix):
            {critique}
            
            **--- YOUR TASK ---**
            Rewrite the 'v1 Draft' to perfectly fix all issues from the 'Critique'.
            Output *only* the final, corrected text.
            
            **REFINED V2 DRAFT (You MUST format your entire response using rich Markdown. Use lists, bolding, and LaTeX for any mathematical expressions. Preserve the '---ANSWER KEY---' separator):**
            """
            
            refiner_prompt = PromptTemplate.from_template(refiner_prompt_template_str)
            refiner_final_prompt = refiner_prompt.format(
                request=user_prompt,
                v1_draft=v1_draft,
                critique=critique_content
            )
            
            refiner_response_stream = refiner_llm.stream(refiner_final_prompt)
            return refiner_response_stream
    
    except Exception as e:
        print(f"Error in PDF pipeline: {e}")
        return _string_to_stream(f"An error occurred: {e}")

def run_general_mode_pipeline(user_prompt, example_text, include_answer_key):
    """
    Runs the 1-agent (Generator) general knowledge pipeline.
    Returns a STREAM of the final text.
    """
    try:
        if include_answer_key:
            answer_key_request = "You MUST include a detailed, step-by-step answer key. Separate the questions from the answer key with the tag '---ANSWER KEY---'."
        else:
            answer_key_request = "Do NOT include an answer key."

        generator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        
        general_prompt_template_str = """
        You are an expert exam question generator.
        Your task is to create a set of questions based on the user's request using your general knowledge.
        You MUST match the style, tone, and difficulty of the example questions.
        {answer_key_request}

        **EXAMPLE QUESTIONS (Follow this style):**
        {examples}
        **USER REQUEST:**
        {request}

        **GENERATED QUESTIONS (You MUST format your entire response using rich Markdown. Use lists, bolding, and LaTeX for any mathematical expressions):**
        """
        
        generator_prompt = PromptTemplate.from_template(general_prompt_template_str)
        
        generator_final_prompt = generator_prompt.format(
            examples=example_text,
            request=user_prompt,
            answer_key_request=answer_key_request
        )
        
        generator_response_stream = generator_llm.stream(generator_final_prompt)
        
        return generator_response_stream
    
    except Exception as e:
        print(f"Error in general pipeline: {e}")
        return _string_to_stream(f"An error occurred: {e}")
