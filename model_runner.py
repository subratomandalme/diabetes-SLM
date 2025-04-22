import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import argparse
import sys
import traceback
import time

WORKING_MODEL_NAME = "google/flan-t5-large"

def load_model(model_name):
    print(f"--- Loading model: {model_name} ---", flush=True)
    try:
        print("Loading tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded.", flush=True)

        print("Determining device and dtype...", flush=True)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        dtype = torch.float16 if use_cuda else torch.float32
        device_map_config = "auto" if use_cuda else None
        print(f"Using device: {device}, dtype: {dtype}, device_map: {device_map_config}", flush=True)

        print(f"Loading model with AutoModelForSeq2SeqLM.from_pretrained({model_name})...", flush=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map_config
        )

        model_device_map = model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'
        model_device = model.device if not model_device_map or not isinstance(model_device_map, dict) else 'See Map'

        print(f"Model loaded successfully. Device map: {model_device_map}, Device: {model_device}", flush=True)
        print("--- Model loading complete ---", flush=True)
        return model, tokenizer, device
    except Exception as e:
        print(f"!!! Error loading model {model_name}: {e} !!!", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        raise

def generate_response(model, tokenizer, device, prompt, max_len=256, verbose=False):
    input_prompt = (
        "You are a knowledgeable medical assistant specializing in diabetes. "
        f"Please provide a detailed, coherent, and accurate response to the question below:\n{prompt}"
    )
    
    print(f"[MODEL_RUNNER DEBUG] Generating response for prompt: '{prompt[:100]}...'", file=sys.stderr, flush=True)
    generation_start_time = time.time()

    try:
        inputs = tokenizer(
            input_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        print("[MODEL_RUNNER DEBUG] Starting model.generate()...", file=sys.stderr, flush=True)
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_len,
                num_beams=6,
                no_repeat_ngram_size=4,
                length_penalty=1.3,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )
        generation_end_time = time.time()
        print(f"[MODEL_RUNNER DEBUG] model.generate() finished. Duration: {generation_end_time - generation_start_time:.2f}s", file=sys.stderr, flush=True)

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print("[MODEL_RUNNER DEBUG] Response decoded successfully.", file=sys.stderr, flush=True)
        return response.strip()
    except Exception as e:
        generation_end_time = time.time()
        print(f"[MODEL_RUNNER DEBUG] model.generate() failed after {generation_end_time - generation_start_time:.2f}s", file=sys.stderr, flush=True)
        print(f"!!! Error generating response: {e} !!!", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return f"Error during text generation: {str(e)}"

def interactive_mode(model, tokenizer, device, max_len_generate):
    print(f"\n{WORKING_MODEL_NAME} Interactive mode started. Type 'exit' to quit.", flush=True)
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() == 'exit': break
            if not user_input: continue
            response = generate_response(model, tokenizer, device, user_input, max_len=max_len_generate, verbose=True)
            print(f"Model: {response}", flush=True)
        except EOFError: break
        except KeyboardInterrupt: break
    print("\nInteractive mode ended.", flush=True)

def main():
    parser = argparse.ArgumentParser(description=f"Run {WORKING_MODEL_NAME} for API/interactive access.")
    parser.add_argument("--model_path", type=str, help="[Ignored] Path argument, uses google/flan-t5-large.")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode instead of API mode")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum generation length for response")
    args = parser.parse_args()

    model_name_to_load = WORKING_MODEL_NAME
    if args.model_path:
        print(f"NOTE: Ignoring --model_path ('{args.model_path}') argument. Loading hardcoded model: {model_name_to_load}", file=sys.stderr, flush=True)

    try:
        model, tokenizer, device = load_model(model_name_to_load)
    except Exception as e:
        print(f"Failed to load required model '{model_name_to_load}'. Exiting.", file=sys.stderr, flush=True)
        sys.exit(1)

    if args.interactive:
        interactive_mode(model, tokenizer, device, args.max_len)
    else:
        print("Model loaded and ready for input.", flush=True)
        while True:
            try:
                print("[MODEL_RUNNER DEBUG] Waiting for input query...", file=sys.stderr, flush=True)
                query = input()
                query = query.strip()
                print(f"[MODEL_RUNNER DEBUG] Received query: '{query[:100]}...'", file=sys.stderr, flush=True)
                if not query: continue

                
                response = generate_response(model, tokenizer, device, query, max_len=args.max_len)

                
                print("[MODEL_RUNNER DEBUG] Printing response to stdout...", file=sys.stderr, flush=True)
                print(response, flush=True)
                
                print("[MODEL_RUNNER DEBUG] Printing END_OF_RESPONSE marker to stdout...", file=sys.stderr, flush=True)
                print("END_OF_RESPONSE", flush=True)
                print("[MODEL_RUNNER DEBUG] Finished processing query.", file=sys.stderr, flush=True)

            except EOFError:
                print("Stdin closed (EOFError). Exiting model runner.", file=sys.stderr, flush=True)
                break
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt received. Exiting.", file=sys.stderr, flush=True)
                break
            except Exception as e:
                print(f"!!! Error processing input query: {e} !!!", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                
                print(f"Error occurred: {e}", flush=True)
                print("END_OF_RESPONSE", flush=True)

if __name__ == "__main__":
    main()
