from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from vecotordb import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are a very helpful asisstant. And you will be very expert in Answering Questions, That will be asked to you

Here are some relvant reviews: {reviews}
Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n ************************************* \n\n")
    question = input("Ask your question (q to quit):")
    print("\n\n ************************************* \n\n")
    if question == "q":
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)