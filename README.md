# 📄 Assistente de Contratos (RAG + FAISS)

Este é um aplicativo Streamlit com RAG (Retrieval-Augmented Generation) usando LangChain + FAISS para responder perguntas sobre contratos em PDF ou TXT.

## ✅ Funcionalidades

- Upload de contrato (.pdf ou .txt)
- Processamento com LangChain + FAISS
- Chat com histórico e fontes citadas
- Suporte à OpenAI API

## 🚀 Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Como hospedar no Render

1. Crie um repositório no GitHub com este conteúdo
2. Acesse https://render.com e crie um novo Web Service
3. Configure:

| Campo             | Valor |
|-------------------|-------|
| Build Command     | `pip install -r requirements.txt` |
| Start Command     | `streamlit run app.py --server.port 10000 --server.address 0.0.0.0` |

4. Adicione a variável de ambiente:

```env
OPENAI_API_KEY=sk-sua-chave-aqui
```

5. Acesse o app pelo link gerado pelo Render após o deploy