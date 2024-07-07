from agent import Agent
import streamlit as st

# Streamlit app
def main():
    st.title("Onboarding Assistant")

    agent = Agent()

    # Input from user
    user_input = st.text_input("Faça sua pergunta ou solicitação:")

    if st.button("Enviar"):
        if user_input:
            response = agent.execute_agent(user_input)
            st.write(response['output'])

    # Display chat history
    st.subheader("Histórico de Conversa")
    for i in range(0, len(agent.chat_history), 2):
        st.write(agent.chat_history[i])
        st.write(agent.chat_history[i + 1])

if __name__ == "__main__":
    main()
