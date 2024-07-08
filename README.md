# Onboarding Assistent

Este é um assistente virtual desenvolvido para auxiliar no processo de onboarding de novos membros na empresa Tech4Humans. O assistente utiliza uma variedade de ferramentas para fornecer suporte, incluindo busca por tutoriais específicos, pesquisa em documentos internos da empresa e agendamento de eventos no Google Calendar.

## Funcionalidades Principais

- **Busca por Tutoriais:** Permite buscar e fornecer tutoriais detalhados de acesso e instalação para softwares específicos.
- **RAG (Retrieve and Generate):** Utiliza documentos PDF da empresa para buscar informações relevantes sobre processos internos e políticas.
- **Agendamento de Eventos:** Facilita a criação de eventos no Google Calendar com base nas solicitações dos usuários.


## Prerequisitos

- Python 3.6 
- .env com as chaves de API como desmostrado do .envexamples. Os links para obtenção das chaves serão deixados abaixo.


## Instalação e API_KEYS

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/MirelaDomiciano/OnboardingAgent.git
    ```
2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```
4. **API_KEYS**:
    - ***Setup Google API Credentials***:
        Siga o tutorial [Python Quickstart](https://developers.google.com/calendar/api/quickstart/python).
    
    - ***Groq API key***:
        Acesse: [Groq Console Keys](https://console.groq.com/keys)

    - ***HunggingFace API key***:
        Acesse: [HuggingFace Keys](https://huggingface.co/settings/tokens)

4. **Criar Ambiente Virtual**:
    ```bash
        python -m venv venv
        source venv/bin/activate  # Para Linux/Mac
        .\venv\Scripts\activate  # Para Windows
    ``` 

## Uso

1. **Run**:
    
    Abra um terminal na pasta do projeto e rode:
    ```bash
    streamlit run app.py
    ```
    Isso abrirá uma paginá no seu navegador, a página demora para carregar, pois antes cria o vectorstore que será usado para a ferramenta de rag.



