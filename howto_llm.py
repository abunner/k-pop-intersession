import lmstudio as lms
model = lms.llm("google/gemma-3n-e4b")

# "System Instruction" aka the prompt prefix
chat = lms.Chat("You are a helpful assistant.")

# The actual prompt
chat.add_user_message(f"What does Sony Pictures Entertainment do?")
result = model.respond(chat)
print(result.content)
print(f"Prompt token count: {result.stats.prompt_tokens_count}")
print(f"Output token count: {result.stats.predicted_tokens_count}")
