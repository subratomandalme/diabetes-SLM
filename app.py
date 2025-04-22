import subprocess
import sys
import os
import queue
import threading
import time
import traceback
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


MODEL_RUNNER_PATH = 'model_runner.py'
APP_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
STATIC_FOLDER = os.path.join(APP_DIR, "static")
TEMPLATE_FOLDER = os.path.join(APP_DIR, "templates")

HF_CACHE_DIR = "/tmp/hf_cache" 
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)


input_queue = queue.Queue()
output_queue = queue.Queue()
llm_process = None
llm_ready = False
llm_lock = threading.Lock()


def pipe_reader(pipe, processing_queue, pipe_name):
    try:
        for line in iter(pipe.readline, ''):
            processing_queue.put((pipe_name, line))
    except ValueError:
        print(f"Pipe {pipe_name} closed unexpectedly (ValueError).", flush=True)
    except Exception as e:
        print(f"Error reading pipe {pipe_name}: {e}", flush=True)
        traceback.print_exc()
    finally:
        processing_queue.put((pipe_name, None))
        try: pipe.close()
        except Exception: pass
        print(f"Pipe reader thread for {pipe_name} finished.", flush=True)


def run_llm():
    global llm_process, llm_ready
    with llm_lock:
        llm_ready = False
        if llm_process and llm_process.poll() is None:
             print("Terminating existing LLM process before restart.", flush=True)
             try:
                  llm_process.terminate()
                  llm_process.wait(timeout=5)
             except subprocess.TimeoutExpired:
                  print("Existing LLM process did not terminate quickly, killing.", flush=True)
                  llm_process.kill()
             except Exception as e:
                  print(f"Error terminating existing process: {e}", flush=True)
        llm_process = None

    pipe_processing_queue = queue.Queue()
    process = None
    stdout_thread = None
    stderr_thread = None

    try:
        absolute_model_runner_path = os.path.abspath(MODEL_RUNNER_PATH)
        if not os.path.exists(absolute_model_runner_path):
            print(f"CRITICAL ERROR: model_runner.py not found at resolved path {absolute_model_runner_path}", flush=True)
            output_queue.put(f"Error: LLM runner script not found at {absolute_model_runner_path}")
            return

        cmd = [sys.executable, absolute_model_runner_path]
        working_dir = os.path.dirname(absolute_model_runner_path)

        if not working_dir or not os.path.isdir(working_dir):
             print(f"Warning: Determined working directory '{working_dir}' is invalid.", flush=True)
             working_dir = os.getcwd()
             print(f"Setting working directory to CWD: {working_dir}.", flush=True)
             if not os.path.isdir(working_dir):
                  print(f"CRITICAL ERROR: Fallback CWD '{working_dir}' is also invalid. Cannot start subprocess.", flush=True)
                  output_queue.put(f"Error: Invalid working directory '{working_dir}'")
                  return

        
        sub_env = os.environ.copy()
        sub_env['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
        sub_env['HF_HOME'] = HF_CACHE_DIR
        print(f"Setting TRANSFORMERS_CACHE for subprocess to: {HF_CACHE_DIR}", flush=True)

        print(f"Starting LLM process with command: {' '.join(cmd)}", flush=True)
        print(f"Using working directory for subprocess: {working_dir}", flush=True)

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            cwd=working_dir,
            env=sub_env
        )

        with llm_lock:
            llm_process = process

        stdout_thread = threading.Thread(target=pipe_reader, args=(process.stdout, pipe_processing_queue, 'stdout'), daemon=True)
        stderr_thread = threading.Thread(target=pipe_reader, args=(process.stderr, pipe_processing_queue, 'stderr'), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        print("LLM process started. Pipe readers active. Waiting for initialization signal...", flush=True)

        initialization_successful = False
        current_response = ""
        readers_finished_count = 0
        query_in_progress = False
        process_died = False

        while readers_finished_count < 2:
            process_died = process.poll() is not None
            if process_died:
                print("LLM process terminated unexpectedly mid-loop.", flush=True)
                break

            if initialization_successful and not query_in_progress and not input_queue.empty():
                query = input_queue.get()
                print(f"[DEBUG] Processing query from input queue: {query[:100]}...", flush=True)
                try:
                    current_response = ""
                    process.stdin.write(query + "\n")
                    process.stdin.flush()
                    query_in_progress = True
                    print("[DEBUG] Query sent to LLM stdin.", flush=True)
                except BrokenPipeError:
                    print("LLM process stdin pipe broke while writing query.", flush=True)
                    process_died = True
                    break
                except Exception as e_write:
                    print(f"Error writing to LLM stdin: {e_write}", flush=True)
                    process_died = True
                    break

            try:
                pipe_name, line = pipe_processing_queue.get(timeout=0.1)

                if line is None:
                    readers_finished_count += 1
                    print(f"[DEBUG] Reader thread for {pipe_name} signaled completion.", flush=True)
                    continue

                raw_line = line
                line = line.strip()
                print(f"LLM Pipe [{pipe_name}]: {line}", flush=True)

                if pipe_name == 'stdout':
                    if "Model loaded and ready for input." in line and not initialization_successful:
                        with llm_lock:
                            llm_ready = True
                        initialization_successful = True
                        print("LLM initialized and ready for input signal received.", flush=True)
                    elif initialization_successful and query_in_progress:
                        if line == "END_OF_RESPONSE":
                            print("[DEBUG] END_OF_RESPONSE marker DETECTED.", flush=True)
                            final_response_to_put = current_response.strip()
                            print(f"[DEBUG] Accumulated response before putting on queue: '{final_response_to_put[:200]}...'", flush=True)
                            output_queue.put(final_response_to_put)
                            print("[DEBUG] Successfully put response onto output_queue.", flush=True)
                            current_response = ""
                            query_in_progress = False
                        elif line:
                            print(f"[DEBUG] Accumulating line: '{line}'", flush=True)
                            current_response += raw_line

            except queue.Empty:
                continue

        print("Exited LLM processing loop.", flush=True)

        if process_died and query_in_progress:
            print("LLM process died before sending END_OF_RESPONSE.", flush=True)
            output_queue.put(f"Error: LLM process terminated unexpectedly. Partial response: {current_response.strip()}")
        elif process_died and not initialization_successful:
             print("LLM process died before initialization was successful.", flush=True)
             output_queue.put(f"Error: LLM process terminated before initialization completed.")

    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: FileNotFoundError during Popen execution. Command: '{' '.join(cmd)}', CWD: '{working_dir}'. Error: {e}", flush=True)
        output_queue.put(f"Error: Failed to execute LLM script (FileNotFoundError).")
    except OSError as e:
        print(f"CRITICAL ERROR: Failed to start subprocess. OSError: {e}. Command: '{' '.join(cmd)}', CWD: '{working_dir}'", flush=True)
        output_queue.put(f"Error: Failed to start LLM process (OSError).")
    except Exception as e:
        print(f"CRITICAL ERROR in run_llm thread: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        output_queue.put(f"Error: Critical failure in LLM runner thread: {type(e).__name__}")
    finally:
        with llm_lock:
            if process and process.poll() is None:
                print("Terminating leftover LLM process in finally block...", flush=True)
                process.terminate()
                try: process.wait(timeout=2)
                except subprocess.TimeoutExpired: process.kill()
            final_return_code = process.poll() if process else 'N/A (never started or cleared)'
            print(f"LLM process final return code in finally: {final_return_code}", flush=True)
            llm_ready = False
            llm_process = None

        if stdout_thread and stdout_thread.is_alive(): stdout_thread.join(timeout=0.5)
        if stderr_thread and stderr_thread.is_alive(): stderr_thread.join(timeout=0.5)

        print("LLM run thread finished.", flush=True)


@app.route('/')
def index():
    template_path = os.path.join(TEMPLATE_FOLDER, 'index.html')
    if not os.path.exists(template_path):
        return """<!doctype html><html><head><title>LLM Query</title></head><body><h1>Ask the LLM</h1><p>Error: index.html template not found in {}</p></body></html>""".format(TEMPLATE_FOLDER), 500
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_llm():
    global llm_process, llm_ready
    request_start_time = time.time()
    print(f"\n--- Received /ask request at {request_start_time:.2f} ---", flush=True)
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            print("Invalid request: Missing JSON data or 'query' field.", flush=True)
            return jsonify({"error": "Invalid Request", "response": "Request must be JSON with a 'query' field."}), 400

        query = data.get('query', '').strip()
        if not query:
            print("Invalid request: Empty query.", flush=True)
            return jsonify({"error": "Empty query", "response": "Please enter a question."}), 400

        print(f"Query received: {query[:150]}...", flush=True)

        llm_needs_start = False
        with llm_lock:
            if llm_process is None or llm_process.poll() is not None:
                print("LLM process not running or terminated. Requesting start...", flush=True)
                llm_needs_start = True
                llm_ready = False

        if llm_needs_start:
            absolute_model_runner_path = os.path.abspath(MODEL_RUNNER_PATH)
            if not os.path.exists(absolute_model_runner_path):
                print(f"ERROR: Cannot start LLM thread. model_runner.py not found at resolved path: {absolute_model_runner_path}", flush=True)
                return jsonify({"error": "Configuration Error", "response": "LLM runner script not found at configured path."}), 500

            print("Starting run_llm thread...", flush=True)
            llm_thread = threading.Thread(target=run_llm, daemon=True)
            llm_thread.start()

        print("Waiting for LLM readiness signal...", flush=True)
        wait_start_time = time.time()
        initialization_timeout = 300
        ready_check_passed = False
        llm_died_during_wait = False

        while time.time() - wait_start_time < initialization_timeout:
            with llm_lock:
                if llm_ready:
                    ready_check_passed = True
                    break
                if llm_process is not None and llm_process.poll() is not None:
                    print("LLM process terminated during initialization wait.", flush=True)
                    llm_died_during_wait = True
                    break
            time.sleep(0.5)

        if not ready_check_passed:
            elapsed_wait = time.time() - wait_start_time
            error_reason = "process terminated prematurely" if llm_died_during_wait else "timeout period reached"
            print(f"LLM failed to become ready ({error_reason}). Waited {elapsed_wait:.2f}s / {initialization_timeout}s.", flush=True)
            try:
                 error_msg = output_queue.get_nowait()
                 if isinstance(error_msg, str) and error_msg.startswith("Error:"):
                      print(f"Error message from run_llm thread: {error_msg}", flush=True)
                      return jsonify({"error": "Initialization Failed", "response": f"LLM initialization failed. Detail: {error_msg}"}), 503
            except queue.Empty:
                 pass
            return jsonify({"error": "Initialization Timeout/Failure", "response": f"The LLM failed to start or initialize in time ({initialization_timeout}s). Reason: {error_reason}. Check server logs."}), 503

        print(f"LLM is ready. Sending query to input queue... (Total time elapsed: {time.time() - request_start_time:.2f}s)", flush=True)
        input_queue.put(query)

        response_timeout = 180
        response_wait_start_time = time.time()
        try:
            response = output_queue.get(timeout=response_timeout)
            print(f"Received final response from output_queue. (Response wait: {time.time() - response_wait_start_time:.2f}s)", flush=True)

            if isinstance(response, str) and response.startswith("Error:"):
                print(f"LLM run thread reported an error: {response}", flush=True)
                return jsonify({"error": "LLM Processing Error", "response": response}), 500
            else:
                return jsonify({"response": response})

        except queue.Empty:
            elapsed_response_wait = time.time() - response_wait_start_time
            print(f"Flask response timeout: output_queue was empty after {elapsed_response_wait:.2f} seconds.", flush=True)
            with llm_lock:
                if llm_process is None or llm_process.poll() is not None:
                    print("LLM process appears to have terminated while waiting for response queue.", flush=True)
                    return jsonify({"error": "LLM Process Error", "response": "The LLM process stopped unexpectedly while waiting for the response."}), 500
                else:
                    print("LLM process still running but response queue timeout hit.", flush=True)
                    return jsonify({"error": "Response Timeout", "response": f"The LLM took too long ({response_timeout}s) to produce a final response via the queue. Check logs for details."}), 408

    except Exception as e:
        print(f"CRITICAL ERROR in /ask endpoint: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error", "response": "An unexpected error occurred in the application while processing your request."}), 500
    finally:
        print(f"--- Finished /ask request. Total time: {time.time() - request_start_time:.2f}s ---", flush=True)


if __name__ == '__main__':
    print("Verifying paths...")
    print(f"  APP_DIR: {APP_DIR}")
    absolute_model_runner_path = os.path.abspath(MODEL_RUNNER_PATH)
    print(f"  Resolved MODEL_RUNNER_PATH: {absolute_model_runner_path}")
    print(f"  TEMPLATE_FOLDER: {TEMPLATE_FOLDER}")
    print(f"  HF_CACHE_DIR: {HF_CACHE_DIR}")

    if not os.path.exists(absolute_model_runner_path):
        print(f"ERROR: Cannot start Flask app. model_runner.py not found at resolved path: {absolute_model_runner_path}", flush=True)
        sys.exit(1)

    print("Attempting to start Flask application...", flush=True)
    app.run(host='0.0.0.0', port=7860, debug=False, use_reloader=False)
    print("Flask application has stopped.", flush=True)
