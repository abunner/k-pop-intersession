import lmstudio as lms
model = lms.llm("google/gemma-3n-e4b")

with open("reviews.txt", "r") as f:
    lines = f.readlines()

positives = []
negatives = []
others = []

for line in lines:
    chat = lms.Chat("You are a data scientist. You read movie reviews and classify them as positive or negative sentiment.")
    chat.add_user_message(f"Classify this review as Positive, Negative or Other. Respond with one word only: {line}")
    result = model.respond(chat)
    rating = result.content.strip()
    if rating == "Positive":
        positives.append(line)
    elif rating == "Negative":
        negatives.append(line)
    elif rating == "Other":
        others.append(line)
    else:
        print("?????")
    print(f"{rating}: {line.strip()}")

print(f"Positive count {len(positives)}")
print(f"Negative count {len(negatives)}")
print(f"Neither count {len(others)}")
