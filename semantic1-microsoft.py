import asyncio

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistoryTruncationReducer
from semantic_kernel.kernel import Kernel

# Semantic kernel example from microsoft

async def main():
    kernel = Kernel()
    kernel.add_service(AzureChatCompletion(deployment_name="gpt-4o"))

    # Keep the last two messages
    truncation_reducer = ChatHistoryTruncationReducer(
        target_count=2,
    )
    truncation_reducer.add_system_message("You are a helpful chatbot.")

    is_reduced = False

    while True:
        user_input = input("User:> ")

        if user_input.lower() == "exit":
            print("\n\nExiting chat...")
            break

        is_reduced = await truncation_reducer.reduce()
        if is_reduced:
            print(f"@ History reduced to {len(truncation_reducer.messages)} messages.")

        response = await kernel.invoke_prompt(
            prompt="{{$chat_history}}{{$user_input}}", user_input=user_input, chat_history=truncation_reducer
        )

        if response:
            print(f"Assistant:> {response}")
            truncation_reducer.add_user_message(str(user_input))
            truncation_reducer.add_message(response.value[0])

    if is_reduced:
        for msg in truncation_reducer.messages:
            print(f"{msg.role} - {msg.content}\n")
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())