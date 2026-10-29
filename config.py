"""
SocialSense AI — IBM Granite credentials and client factory.
"""
import os
from dotenv import load_dotenv

load_dotenv()

IBM_API_KEY = os.getenv("IBM_API_KEY", "cOUFZi9BjsF5oq_cZ5Np-8esTa6z-fFhQVY3RbNCpgOX")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID", "5cc13a5b-c3b2-4126-8368-4dfcb32c3593")
IBM_ENDPOINT_URL = os.getenv("IBM_ENDPOINT_URL", "https://au-syd.ml.cloud.ibm.com")

# IBM Granite model IDs
GRANITE_MODEL_ID = "ibm/granite-3-8b-instruct"

_client = None

def get_watsonx_client():
    """Return a cached WatsonX AI client."""
    global _client
    if _client is None:
        try:
            from ibm_watsonx_ai import APIClient, Credentials
            credentials = Credentials(
                url=IBM_ENDPOINT_URL,
                api_key=IBM_API_KEY,
            )
            _client = APIClient(credentials=credentials, project_id=IBM_PROJECT_ID)
        except Exception as e:
            return None, str(e)
    return _client, None


def generate_text(prompt: str, max_tokens: int = 900, temperature: float = 0.7) -> str:
    """
    Call IBM Granite via watsonx.ai and return the generated text.
    Falls back to a labelled placeholder when credentials are not available.
    """
    try:
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params

        credentials = Credentials(
            url=IBM_ENDPOINT_URL,
            api_key=IBM_API_KEY,
        )

        model = ModelInference(
            model_id=GRANITE_MODEL_ID,
            credentials=credentials,
            project_id=IBM_PROJECT_ID,
            params={
                Params.MAX_NEW_TOKENS: max_tokens,
                Params.TEMPERATURE: temperature,
                Params.STOP_SEQUENCES: [],
            },
        )
        response = model.generate_text(prompt=prompt)
        return response.strip() if isinstance(response, str) else str(response)

    except Exception as e:
        return f"[IBM Granite unavailable — {e}]\n\nSample output for demonstration purposes only."
