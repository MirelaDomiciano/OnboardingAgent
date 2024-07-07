import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain import hub
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from calendarEvents import create_event_tool

load_dotenv()

class Agent:
    def __init__(self):
        self.llm = ChatGroq(model="llama3-70b-8192", temperature=0.5)
        self.search = GoogleSearchAPIWrapper()
        self.chat_history = {"conversations": []}
        self.vectorstore = self.initialize_vectorstore()
        self.tool_search = Tool(
            name="google_search",
            description="""Pesquise por tutoriais na web, pesquise em documentações oficiais e responda com o tutorial completo de acesso ou instalação da ferramenta solicitada, o tutorial deve ser passado em tópicos explicando passo a passo o que o usuário deve fazer.
                            Entre as ferramentas deste escopo estão: Github, Vscode, Jira e Discord.
                            Você não deve pesquisar sobre outras ferramentas. Lembre-se de consultar o histórico de conversas para entender se a pergunata e considerar se está é a ferramente""",
            func=self.search.run,
        )
        
        self.tool_rag = Tool(
            name="rag_tool",
            description="""Retrieve e generate informação do documento pdf com as informações da empresa Tech4Humans e também do TechLab Agentes. Caso o assunto seja abordado no pdf, monte uma resposta final unindo as informações disponíveis, se a informação não for encontrada, responda com uma resposta padrão.
            Lembre-se de consultar o histórico de conversas para entender se a pergunata e considerar se está é a ferramente.""",
            func=self.rag_tool,
        )

        self.tool_calendar = Tool(
            name="create_event",
            description="""Crie um evento no Google Calendar fornecendo os detalhes do evento como resumo, localização, descrição, hora de início, hora de término e fuso horário.
                            Essa função deve ser chamada quando o usuário quiser marcar um evento, como reunião, almoço, etc.
                            Exemplo a função deve ser chamada: "summary": "Reunião de Marketing", "location": "Sala de Reuniões 1", "description": "Reunião para discutir estratégias de marketing.", "start_time": "2024-07-05T10:00:00", "end_time": "2024-07-05T11:00:00", "timezone": "America/Sao_Paulo". 
                            Deve-se converter expressões como "amanhã às 7h da manhã" para o formato de data e hora amanha = 2024-06-05 e 7h da manhã = 7:00:00. E depois retornar que o evento foi criado com sucesso. Monte uma resposta final coma informação se o evento foi criado com sucesso ou não.
                            Lembre-se de consultar o histórico de conversas para entender se a pergunata e considerar se está é a ferramente""",
            func=create_event_tool,
        )

        self.prompt = self.get_react_chat_template()

        self.tools_list = [self.tool_rag, self.tool_search, self.tool_calendar]
        self.agent = create_react_agent(llm=self.llm, tools=self.tools_list, prompt=self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools_list, verbose=True, handle_parsing_errors=True)

    def get_react_chat_template(self):
        return PromptTemplate.from_template(
            """Você é um assistente de onboarding e deve auxiliar novos membros da empresa Tech4Humans utilizando as ferramentas implementadas, caso não chgue em uma resposta deve responder que não pode ajudar no assunto. 
            Entre as ferramentas tem-se de busca para tutoriais de instalação e acesso de ferramentas, RAG para pesquisa no documento com as informações da empresa e agendamento de reuniões no google Calendar. 
            Você não deve: Falar sobre outras empresas; Não deve fornecer informações pessoais; Deve inibir discurso de ódio. Não pode aceitar requisições maliciosas. 
            Responda as seguintes perguntas da melhor forma possível em português (pt-br), usando as ferramentas disponibilizadas e levando em consideração o {chat_history} como contexto. 

            TOOLS:
            ------
            Você tem acesso às seguintes ferramentas:
            {tools}
            Você deve usar as ferramentas no seguinte formato e somente uma ferramenta por input:
            ```
            Tool: search_tool
                Question:{input}
                Thought: Tem algo útil no histórico da conversa? Se sim use para pensar sobre qual das ferramentas você deve usar para responder a pergunta, se não ignore e pense qual ferramenta é mais coerente com a pergunta.
                Action: a ação a ser tomada, deve ser uma das [{tool_names}]. Pesquisar na web somente as ferramentas citadas na search_tool, caso não esteja responda que não pode auxiliar. Quando necessário marcar um evento ou reunião chamar função tool_calendar. Os demais assuntos devem ser verificados no documento pdf, como círculos da empresa, programas internos. Sempre envie informações completas. 
                Action Input: a entrada para a ação
                Observation: o resultado da ação
                Thought: I now know the final answer
                Final Answer: a resposta final à pergunta original
            ---
            Begin
            Use o histŕico para te ajudar a responder as perguntas.
            
            Histórico de conversa:
            {chat_history}

            New input: {input}
            {agent_scratchpad}"""
        )
                
    def initialize_vectorstore(self):
        print("Load and process PDF documents...")
        pdf_files = ["Base.pdf", "TechLab Tech4ai.pdf"]
        docs = []

        for pdf_file in pdf_files:
            print(f"Loading PDF file: {pdf_file}")
            loader = PyPDFLoader(pdf_file)
            docs.extend(loader.load())
            print(f"Loaded {pdf_file}")

        print("Split the combined documents..")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
        splits = text_splitter.split_documents(docs)

        print("Create Vectorstore...")
        vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())
        print("Vectorstore created")
        return vectorstore

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_tool(self, query):
        print(f"Executing rag_tool with query: {query}")

        llm = ChatGroq(model="llama3-70b-8192")
        
        retriever = self.vectorstore.as_retriever()  # Assuming vectorstore is a global variable or accessible

        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        result = rag_chain.invoke(query)
        return result


    def execute_agent(self, query):
        try:
            # Store the user input in chat_history before getting the response
            self.update_chat_history("User", query)
            self.query = query

            # Get the agent's response
            result = self.agent_executor.invoke({"input": query, "chat_history": self.chat_history, "agent_scratchpad": ""})
            
            # Extract the final answer from the result
            final_answer = result["output"]

            # Store the agent response in chat_history
            self.update_chat_history("Agent", final_answer)

            # Return the final answer to be displayed
            return final_answer
        
        except Exception as e:
            error_message = f"An error occurred: {e}"
            self.update_chat_history("Agent", error_message)
            return error_message

    def update_chat_history(self, role, message):
        self.chat_history["conversations"].append({"role": role, "message": message})
