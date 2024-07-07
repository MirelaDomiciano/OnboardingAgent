import streamlit as st
import os

# Inicializa o agente (supondo que você tenha a classe Agent implementada em um módulo separado)
from agent import Agent

# Configuração da página
st.set_page_config(page_title="Tech4Humans Onboarding Assistent", layout="centered")

# Inicializa o agente
agent = Agent()

# Armazena as mensagens geradas pelo agente
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Exibe ou limpa as mensagens do chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


# Função para gerar a resposta do agente
def generate_agent_response(prompt_input):
    # Aqui você deve implementar a lógica para chamar seu agente e obter a resposta
    response = agent.execute_agent(prompt_input)
    return response['output']

# Entrada fornecida pelo usuário
if prompt := st.chat_input(placeholder="Type a message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Gera uma nova resposta se a última mensagem não for do assistente
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_agent_response(prompt)
                placeholder = st.empty()
                full_response = response  # Supondo que 'response' seja uma string completa
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

# Botão para limpar o histórico de chat
st.button('Clear Chat History', on_click=clear_chat_history)