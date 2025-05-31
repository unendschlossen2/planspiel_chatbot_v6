import huggingface_hub
from huggingface_hub import login
from typing import Optional

def attempt_huggingface_login(token: Optional[str] = None) -> bool:
    if not huggingface_hub.whoami():
        if token:
            try:
                login(token=token, add_to_git_credential=False)
                print("Hugging Face login successful using provided token.")
                return True
            except Exception as e:
                print(f"Hugging Face login failed with provided token: {e}")
                return False
        else:
            print("No Hugging Face token provided. Skipping token-based login.")
            print("You can log in interactively if in a suitable environment or by setting the HUGGING_FACE_HUB_TOKEN environment variable.")
            return False
    else:
        print("Already logged in to Hugging Face.")
        return True