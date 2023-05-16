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
            The user gives you a file its content is represented by the following pieces of context,and you can help me summarize it in the following format:
                1. Please summarize how many vocabulary words the child has learned and their mastery level.
                2. Please summarize the interaction and expression between the child and the foreign teacher from the listening, speaking, and reading dimensions.
                3. Please summarize the areas where the child needs improvement and the corresponding improvement plan.
                4. Finally, please give me some test questions and suggestions to enhance the effectiveness of today's lesson.
                5. Please summarize the sentences the child said.
                6. Based on the child's performance today, please offer some English learning suggestions.
                7. Based on the child's situation, please suggest an English language game that involves role-playing. 
            
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

    
    
