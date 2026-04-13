import apple_fm_sdk as fm
import asyncio
from pathlib import Path
from datetime import datetime


def save_response_to_md(user_prompt, response, base_dir="responses"):
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"{timestamp}.md"
    
    file_contents = f"User Query: {user_prompt} \n\n"
    file_contents += f"Model response: \n \n{response}"
    file_path.write_text(file_contents, encoding="utf-8")
    return file_path

async def main():
    # Get the default system foundation model
    model = fm.SystemLanguageModel()

    # Check if the model is available
    is_available, reason = model.is_available()
    if is_available:
        # Create a session
        session = fm.LanguageModelSession()

        # Generate a response
        print("\nWelcome to Foundation Models, Apple's on-device LLM.\n")
        prompt = input("What can I help with?  ")
        response = await session.respond(prompt)
        print(f"Model response: \n \n {response}")

        saved_response_file = save_response_to_md(prompt, response)
        print(f"Successfully saved to {saved_response_file}")
    else:
        print(f"Foundation Models not available: {reason}")

# Run async function
asyncio.run(main())