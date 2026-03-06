from openai import OpenAI

client = OpenAI()

question = input("Ask a question: ")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": question
        }
    ],
)

answer = response.choices[0].message.content

print("pls check the output file for answer")

# save

with open("output.txt", "a") as file:
    file.write("\n New Question: \n")
    file.write("Question: " + question + "\n")
    file.write("Answer: " + answer + "\n")
