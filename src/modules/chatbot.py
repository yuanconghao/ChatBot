import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback

#fix Error: module 'langchain' has no attribute 'verbose'
import langchain
langchain.verbose = False

class Chatbot:

    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors

    # qa_template = """
    #     You are a helpful AI assistant named ChatBot. The user gives you a file its content is represented by the following pieces of context, use them to answer the question at the end.
    #     If you don't know the answer, just say you don't know. Do not try to make up an answer.
    #     If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
    #     Use as much detail as possible when responding.
    #
    #     context: {context}
    #     =========
    #     question: {question}
    #     ======
    #     """
    qa_template = """
            You are an intelligent report generator named ChatBot.
            The user gives you a file its content is represented by the following pieces of context,and I need your help to read and summarize the following questions:
                1. Please summarize how many vocabulary words the child has speak and its mastery level.(with Chinese translation) 
                2. Please summarize the interaction and expression between the child and the foreign teacher from the listening, speaking, and reading dimensions.(with Chinese translation)                     
                3. Please summarize the areas where the child needs improvement and the corresponding improvement plan.(with Chinese translation)
                4. Finally, please give me some test questions and suggestions to enhance the effectiveness of today's lesson.(with Chinese translation)
                5. Please summarize the sentences the child said.(with Chinese translation)
                6. Based on the child's performance today, please offer some English learning suggestions.(with Chinese translation)
                7. Based on the child's situation, please suggest an English language game that involves role-playing.(with Chinese translation) 
                
                Follow the format of the output that follows:                  
                 1. Vocabulary: xxx\n\n
                 2. Interaction: xxx\n\n
                    - (1) listening: xxx\n\n
                    - (2) speaking: xxx\n\n
                    - (3) reading dimensions: xxx\n\n
                 3. Improvement: xxx\n\n                 
                 4. Lesson Suggestions: xxx\n\n
                 5. Summary: \n\n
                    - (1):xxx;\n 
                    - (2):xxx;\n 
                    - (3):xxx;\n  
                 6. Learning suggestions: \n\n
                    - (1):xxx;\n 
                    - (2):xxx;\n 
                    - (3):xxx;\n
                 7. English Language Game: xxx\n\n
                 
            
            context: {context}
            =========
            question: {question}
            ======
            """

    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])

    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        retriever = self.vectors.as_retriever()


        chain = ConversationalRetrievalChain.from_llm(llm=llm,
            retriever=retriever, verbose=True, return_source_documents=True, combine_docs_chain_kwargs={'prompt': self.QA_PROMPT})

        chain_input = {"question": query, "chat_history": st.session_state["history"]}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        #count_tokens_chain(chain, chain_input)
        return result["answer"]


def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### Tokens used in this conversation : {cb.total_tokens} tokens')
    return result 

    
    
