print('Program launching! (This might take a bit to load)')
import argparse
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn
import logging
from typing import List, Optional, Union, Dict, Any

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
#MODEL_NAME = "google/gemma-3-12b-it" # The target model as per design
MODEL_NAME = "google/gemma-3-4b-it"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.1

# Quantization settings (as per design)
# These can be overridden by command-line arguments
USE_BFLOAT16 = False # Generally not needed if using 8-bit quantization
USE_QUANTIZATION = True
QUANTIZATION_BITS = 8 # Default to 8-bit

# Global variables for model and tokenizer
tokenizer = None
model = None
text_generation_pipeline = None

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Gemma LLM API Server (OpenAI Compatible)",
    description="An OpenAI-compatible API to interact with the Google Gemma 3 12B IT model locally.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- Pydantic Models for OpenAI API Compatibility ---

class Message(BaseModel):
    """Represents a message in a conversation, conforming to OpenAI API."""
    role: str = Field(..., description="The role of the author of this message (e.g., 'system', 'user', 'assistant').")
    content: str = Field(..., description="The content of the message.")

class ChatCompletionRequest(BaseModel):
    """Request body for the /v1/chat/completions endpoint."""
    messages: List[Message] = Field(..., description="A list of messages comprising the conversation so far.")
    model: str = Field(MODEL_NAME, description="ID of the model to use. Default is the hosted model.")
    max_tokens: Optional[int] = Field(DEFAULT_MAX_NEW_TOKENS, ge=1, description="The maximum number of tokens to generate in the chat completion.")
    temperature: Optional[float] = Field(DEFAULT_TEMPERATURE, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: Optional[float] = Field(DEFAULT_TOP_P, ge=0.0, le=1.0, description="Nucleus sampling parameter.")
    top_k: Optional[int] = Field(DEFAULT_TOP_K, ge=0, description="Limits the sampling to the k most likely next tokens (Hugging Face specific, not standard OpenAI).")
    repetition_penalty: Optional[float] = Field(DEFAULT_REPETITION_PENALTY, ge=1.0, description="Penalizes repeated tokens (Hugging Face specific, not standard OpenAI).")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Up to 4 sequences where the API will stop generating further tokens.")
    seed: Optional[int] = Field(None, description="If specified, our system will make a best effort to sample deterministically.")
    stream: Optional[bool] = Field(False, description="If true, sends partial message deltas. Not implemented for initial version.")
    # For simplicity, other OpenAI parameters (n, logit_bias, presence_penalty, frequency_penalty, etc.) are omitted.

class ChatCompletionChoiceMessage(BaseModel):
    """Message object within a choice, conforming to OpenAI API."""
    role: str = Field("assistant", description="The role of the author of this message (always 'assistant' for generated content).")
    content: str = Field(..., description="The generated content.")

class ChatCompletionChoice(BaseModel):
    """A single completion choice, conforming to OpenAI API."""
    index: int = Field(0, description="The index of the choice in the list of choices.")
    message: ChatCompletionChoiceMessage = Field(..., description="The generated message.")
    logprobs: Optional[Any] = Field(None, description="Log probability information (not implemented for simplicity).")
    finish_reason: str = Field("stop", description="The reason the model stopped generating tokens (e.g., 'stop', 'length').")

class ChatCompletionUsage(BaseModel):
    """Token usage information, conforming to OpenAI API."""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt.")
    completion_tokens: int = Field(..., description="Number of tokens in the generated completion.")
    total_tokens: int = Field(..., description="Total tokens used (prompt + completion).")

class ChatCompletionResponse(BaseModel):
    """Response body for the /v1/chat/completions endpoint."""
    id: str = Field(..., description="A unique identifier for the chat completion.")
    object: str = Field("chat.completion", description="The object type, which is always 'chat.completion'.")
    created: int = Field(..., description="The Unix timestamp (in seconds) of when the chat completion was created.")
    model: str = Field(..., description="The model used for the chat completion.")
    choices: List[ChatCompletionChoice] = Field(..., description="A list of chat completion choices.")
    usage: ChatCompletionUsage = Field(..., description="Usage statistics for the completion request.")
    # For simplicity, other OpenAI response parameters (system_fingerprint) are omitted.

class Model(BaseModel):
    """Represents a single model in the /v1/models response, conforming to OpenAI API."""
    id: str = Field(..., description="The ID of the model (e.g., 'gpt-3.5-turbo').")
    object: str = Field("model", description="The object type, which is always 'model'.")
    created: int = Field(..., description="The Unix timestamp (in seconds) when the model was created.")
    owned_by: str = Field("user", description="The organization or user who owns the model.")
    # Additional fields like 'permission' are omitted for simplicity

class ModelsListResponse(BaseModel):
    """Response body for the /v1/models endpoint."""
    object: str = Field("list", description="The object type, which is always 'list'.")
    data: List[Model] = Field(..., description="A list of model objects.")


# --- Function to initialize the model ---
def initialize_model(model_name: str, use_bfloat16: bool, use_quantization: bool, quantization_bits: Optional[int]):
    """
    Initializes the tokenizer and model.
    Downloads them if not already cached.
    """
    global tokenizer, model, text_generation_pipeline

    logging.info(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logging.critical(f"Failed to load tokenizer for {model_name}: {e}")
        raise

    logging.info(f"Loading model {model_name}...")
    # Determine torch_dtype based on CUDA availability and bfloat16 preference
    torch_dtype = torch.bfloat16 if use_bfloat16 and torch.cuda.is_available() else \
                  torch.float16 if torch.cuda.is_available() else \
                  torch.float32

    model_kwargs = {"torch_dtype": torch_dtype}

    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto" # Automatically places model parts on available GPUs
        logging.info("CUDA is available. Model will attempt to use GPU.")
    else:
        logging.warning("CUDA not available. Running on CPU will be extremely slow for large models.")
        # Ensure quantization and bfloat16 are off for CPU as they are GPU-specific optimizations
        use_quantization = False
        use_bfloat16 = False # bfloat16 is a GPU feature

    if use_quantization:
        try:
            from transformers import BitsAndBytesConfig
            if quantization_bits == 8:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization_bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4", # NormalFloat 4-bit
                    bnb_4bit_compute_dtype=torch_dtype, # Compute in selected dtype (e.g., bfloat16)
                    bnb_4bit_use_double_quant=True, # Double quantization for slightly better quality
                )
            else:
                logging.error(f"Invalid QUANTIZATION_BITS specified: {quantization_bits}. Must be 4 or 8.")
                use_quantization = False # Disable if invalid bits
            
            if use_quantization: # Only apply if still enabled after validation
                model_kwargs["quantization_config"] = quantization_config
                logging.info(f"Applying {quantization_bits}-bit quantization...")
        except ImportError:
            logging.error("bitsandbytes not found. Quantization will not be applied.")
            logging.error("Please install it with: pip install bitsandbytes")
            use_quantization = False # Disable quantization if library is missing
            model_kwargs.pop("quantization_config", None) # Remove quantization config if not used
        except Exception as e:
            logging.error(f"Error applying quantization: {e}. Quantization disabled.")
            use_quantization = False
            model_kwargs.pop("quantization_config", None)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        model.eval() # Set model to evaluation mode
        logging.info("Model loaded successfully!")

        # Initialize the text generation pipeline
        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        logging.info("Text generation pipeline initialized.")

    except Exception as e:
        logging.critical(f"Failed to load model or pipeline: {e}", exc_info=True)
        raise

# --- API Endpoints ---

@app.get("/")
async def read_root():
    """
    A simple health check endpoint.
    """
    return {"message": "Gemma LLM API is running. Check /docs for API documentation."}

@app.get("/v1/models", response_model=ModelsListResponse)
async def list_models():
    """
    Returns a list of models available on the API, conforming to OpenAI's /v1/models.
    """
    if not model:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded yet.")
    
    current_timestamp = int(time.time())
    
    # In this single-model server, we return details for the loaded model.
    # The 'owned_by' field is set to 'user' as it's a local/self-hosted model.
    gemma_model_info = Model(
        id=MODEL_NAME,
        object="model",
        created=current_timestamp, # Use current time as creation timestamp for simplicity
        owned_by="user"
    )
    
    return ModelsListResponse(object="list", data=[gemma_model_info])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Generates text based on the provided messages using the Gemma model,
    conforming to the OpenAI Chat Completions API.
    """
    if not text_generation_pipeline:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded. Please wait or check server logs.")

    if request.stream:
        # Implementing streaming is more complex with FastAPI's StreamingResponse
        # and a generator. For this initial version, we'll return an error.
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Streaming is not yet implemented for this API.")

    logging.info(f"Received chat completion request. Messages: {request.messages}")

    try:
        # Apply the chat template to the messages list
        # This converts the OpenAI-style messages into the format expected by Gemma
        formatted_prompt = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True # Important for Gemma's instruction-following
        )

        # Prepare generation arguments for the pipeline
        generation_args = {
            "max_new_tokens": request.max_tokens,
            "do_sample": True, # Always use sampling for chat models
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "repetition_penalty": request.repetition_penalty,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "return_full_text": False # Crucial: Only return the newly generated text
        }

        # Add stop sequences if provided
        if request.stop:
            # Hugging Face pipeline expects `stop_sequences` not `stop`
            # It can be a list of strings
            generation_args["stop_sequences"] = request.stop if isinstance(request.stop, list) else [request.stop]

        # Set seed for reproducibility if provided
        if request.seed is not None:
            torch.manual_seed(request.seed) # Set PyTorch seed for reproducibility

        # Generate response using the pipeline
        outputs = text_generation_pipeline(
            formatted_prompt,
            **generation_args
        )

        generated_text = outputs[0]["generated_text"].strip()
        logging.info(f"Generated response (first 50 chars): '{generated_text[:50]}...'")

        # Calculate token usage
        prompt_tokens = len(tokenizer.encode(formatted_prompt))
        completion_tokens = len(tokenizer.encode(generated_text))
        total_tokens = prompt_tokens + completion_tokens

        # Construct OpenAI Chat Completions API response
        response_data = ChatCompletionResponse(
            id=f"chatcmpl-{os.urandom(12).hex()}", # Unique ID
            object="chat.completion",
            created=int(time.time()), # Current Unix timestamp
            model=MODEL_NAME,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionChoiceMessage(
                        role="assistant",
                        content=generated_text,
                    ),
                    finish_reason="stop", # Assuming 'stop' for now; can be refined
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        )
        return response_data

    except Exception as e:
        logging.error(f"Error during text generation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma LLM API Server (OpenAI Compatible).")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME,
                        help=f"Hugging Face model ID to use (default: {MODEL_NAME}).")
    parser.add_argument("--bf16", action="store_true", default=USE_BFLOAT16,
                        help="Use bfloat16 for model weights (requires compatible GPU).")
    parser.add_argument("--quantize", type=int, choices=[4, 8], default=QUANTIZATION_BITS,
                        help="Quantize the model to 4-bit or 8-bit for reduced memory usage. Requires 'bitsandbytes'.")
    parser.add_argument("--token", type=str, default=None,
                        help="Your Hugging Face Hub access token. Will try HF_TOKEN env var if not provided.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host IP to bind the API server to.")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the API server on.")

    args = parser.parse_args()

    # Login to Hugging Face Hub if token is provided or in environment
    if args.token:
        login(token=args.token)
    elif os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
    else:
        logging.warning("Hugging Face token not found. Model download might fail if it's private or requires auth.")
        logging.warning("Please set the HF_TOKEN environment variable or use --token argument.")

    # Determine quantization settings based on arguments
    current_use_quantization = args.quantize is not None
    current_quantization_bits = args.quantize

    # Initialize model and tokenizer before starting the server
    try:
        initialize_model(
            args.model_name,
            args.bf16,
            current_use_quantization,
            current_quantization_bits
        )
    except Exception as e:
        logging.critical(f"Failed to initialize model, exiting: {e}")
        # In a real-world scenario, you might want to raise an exception
        # or implement a retry mechanism. For a simple script, exiting is fine.
        exit(1)

    logging.info(f"Starting FastAPI server on {args.host}:{args.port}")
    # uvicorn.run starts the FastAPI application
    uvicorn.run(app, host=args.host, port=args.port)
