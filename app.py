from flask import Flask, request, jsonify, render_template
import subprocess
import os
import threading
import queue
import time
import sys
import traceback

app = Flask(__name__)


MODEL_RUNNER_PATH = "B:\\code\\basic-slm-diabetes\\diabetes\\model_runner.py"


APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(APP_DIR, "static")
TEMPLATE_FOLDER = os.path.join(APP_DIR, "templates")
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
        llm_process = None

    pipe_processing_queue = queue.Queue()
    process = None
    stdout_thread = None
    stderr_thread = None

    try:
        if not os.path.exists(MODEL_RUNNER_PATH):
            print(f"CRITICAL ERROR: model_runner.py not found at {MODEL_RUNNER_PATH}", flush=True)
            return

        cmd = [sys.executable, MODEL_RUNNER_PATH]
        working_dir = os.path.dirname(MODEL_RUNNER_PATH)

        print(f"Starting LLM process with command: {' '.join(cmd)}", flush=True)
        print(f"Working directory for subprocess: {working_dir}", flush=True)

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace',
            bufsize=1, cwd=working_dir,
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

        while readers_finished_count < 2:
            process_died = process.poll() is not None
            if process_died and readers_finished_count < 2:
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
                    break
                except Exception as e_write:
                    print(f"Error writing to LLM stdin: {e_write}", flush=True)
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
                        print(f"[DEBUG] Checking line against END_OF_RESPONSE. Line: '{line}'", flush=True)
                        if line == "END_OF_RESPONSE":
                            print("[DEBUG] END_OF_RESPONSE marker DETECTED.", flush=True)
                            final_response_to_put = current_response.strip()
                            print(f"[DEBUG] Accumulated response before putting on queue: '{final_response_to_put[:200]}...'", flush=True)
                            output_queue.put(final_response_to_put)
                            print("[DEBUG] Successfully put response onto output_queue.", flush=True)
                            current_response = ""
                            query_in_progress = False
                        elif line:
                            print(f"[DEBUG] Accumulating line: '{raw_line.strip()}'", flush=True)
                            current_response += raw_line

            except queue.Empty:
                if process.poll() is not None and readers_finished_count < 2:
                    print("LLM process terminated while idle.", flush=True)
                    break
                continue

        print("Exited LLM processing loop.", flush=True)
        if process_died and query_in_progress:
            print("LLM process died before sending END_OF_RESPONSE.", flush=True)
            output_queue.put(f"Error: LLM process terminated unexpectedly. Partial response: {current_response.strip()}")


    except Exception as e:
        print(f"CRITICAL ERROR in run_llm thread: {e}", flush=True)
        traceback.print_exc()
        output_queue.put(f"Error: Critical failure in LLM runner thread: {e}")
    finally:
        with llm_lock:
            if process and process.poll() is None:
                print("Terminating leftover LLM process in finally block...", flush=True)
                process.terminate()
                try: process.wait(timeout=2)
                except subprocess.TimeoutExpired: process.kill()
            print(f"LLM process final return code in finally: {process.poll() if process else 'N/A'}", flush=True)
            llm_ready = False
            llm_process = None
        if stdout_thread and stdout_thread.is_alive(): stdout_thread.join(timeout=1)
        if stderr_thread and stderr_thread.is_alive(): stderr_thread.join(timeout=1)
        print("LLM run thread finished.", flush=True)


@app.route('/')
def index():
    template_path = os.path.join(TEMPLATE_FOLDER, 'index.html')
    if not os.path.exists(template_path):
        return """
        <!doctype html>
        <html><head><title>LLM Query</title></head><body>
        <h1>Ask the LLM</h1><p>Error: index.html template not found in {}</p>
        </body></html>
        """.format(TEMPLATE_FOLDER)
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
            if not os.path.exists(MODEL_RUNNER_PATH):
                print(f"ERROR: Cannot start LLM thread. model_runner.py not found at configured path: {MODEL_RUNNER_PATH}", flush=True)
                return jsonify({"error": "Configuration Error", "response": "LLM runner script not found at configured path."}), 500
            print("Starting run_llm thread...", flush=True)
            llm_thread = threading.Thread(target=run_llm, daemon=True)
            llm_thread.start()

        print("Waiting for LLM readiness signal...", flush=True)
        wait_start_time = time.time()
        initialization_timeout = 300

        ready_check_passed = False
        while time.time() - wait_start_time < initialization_timeout:
            with llm_lock:
                if llm_ready:
                    ready_check_passed = True
                    break
                if llm_process is not None and llm_process.poll() is not None and not llm_ready:
                    print("LLM process terminated during initialization wait.", flush=True)
                    break
            time.sleep(0.5)

        if not ready_check_passed:
            elapsed_wait = time.time() - wait_start_time
            print(f"LLM failed to become ready within the timeout period ({elapsed_wait:.2f}s / {initialization_timeout}s). Check LLM process logs above.", flush=True)
            return jsonify({"error": "Initialization Timeout", "response": f"The LLM failed to start or initialize in time ({initialization_timeout}s). Please check server logs or try again later."}), 503


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
        print(f"CRITICAL ERROR in /ask endpoint: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error", "response": "An unexpected error occurred in the application while processing your request."}), 500
    finally:
        print(f"--- Finished /ask request. Total time: {time.time() - request_start_time:.2f}s ---", flush=True)


if __name__ == '__main__':
    print("Verifying paths...")
    print(f"  APP_DIR: {APP_DIR}")
    print(f"  MODEL_RUNNER_PATH: {MODEL_RUNNER_PATH}")
    print(f"  TEMPLATE_FOLDER: {TEMPLATE_FOLDER}")

    if not os.path.exists(MODEL_RUNNER_PATH):
        print(f"ERROR: Cannot start Flask app. model_runner.py not found at: {MODEL_RUNNER_PATH}", flush=True)
        sys.exit(1)

    print("Starting Flask application...", flush=True)
    app.run(debug=False, host='0.0.0.0', port=5000)