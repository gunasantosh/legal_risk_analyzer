# Developer Deep Dive: Legal Risk Analyzer
This document is intended for open-source contributors and maintainers analyzing the inner workings of the `inference.py` workflow, the RL environment integration, and the specific syntax requirements of the Hackathon container evaluations.
## 1. Agent Decision-Making Loop
Unlike simple static testing scripts, the rewritten `inference.py` integrates a true LLM-based autonomous agent logic. The core agent architecture revolves around the custom synchronous/asynchronous handler `get_agent_action()`.
### The State Machine
The environment dictates three required steps passing state dynamically back to the client via `LegalObservation`:
1. **Extraction (Task 1):** The LLM is provided the raw contract string and prompted strictly to replicate and isolate the target "Limitation of Liability". The OpenEnv server intercepts this text output and awards a Token-Level F1 Score.
2. **Classification (Task 2):** Instead of parsing rigid enums, the LLM utilizes instruct-alignment to categorize the risk level ("High", "Medium", "Low"). Case-insensitive substrings guarantee correct transitioning.
3. **Remediation (Task 3):** The final loop dynamically asks the LLM to rewrite the clause mutually. *Wait heavily on the LLM parameter size here*; the accuracy and semantic grading heavily depend on the `MODEL_NAME` capability.
The agent checks the `task_id` assigned by the server loop (`obs.task_id`) and adjusts its instruction prompts accordingly to formulate the perfect `LegalAction`.
## 2. Strict STDOUT Logging Logic
Because the project is assessed inside automated OpenEnv execution harnesses designed by Meta and Hugging Face, terminal log parsers rely heavily on a highly specific exact format. 
The `inference.py` defines three primary formatters:
* `log_start()`: Fires exactly once per runtime initialized against OpenEnv. Tracks the active benchmarker environment and model string `[START] task=legal-risk-analysis env=legal-risk-analyzer model=Qwen...`
* `log_step()`: Emits exactly upon completion of `env.step()`. This captures `[STEP] step=1 action='extract(...)' reward=0.99 done=false error=null`. Crucially, quotes inside `action` are sanitized to avoid multiline breaking within the STDOUT regex parsers.
* `log_end()`: Tracks cumulative metrics universally upon completion or pipeline fracture. Emits `[END] success=true steps=3 score=0.910 rewards=0.99,0.99,0.75`.
*Note:* `asyncio.run(main())` within the `if __name__ == '__main__'` block handles asynchronous setup, but STDOUT is flushed entirely in sequence ensuring zero interleaving.
## 3. Troubleshooting the OpenEnv Server & Container
If your inference fails to mount locally or on the validation test server, review the following heuristics:
- **Validation Failure (Not Ready / main() not reachable):** Check the entry point. The system's `openenv validate` evaluates if the environment endpoints are healthy and properly expose standard classes. Ensure `app.py` has no bad imports resulting from the root namespace moving to `/server`.
- **Hugging Face Model Timeouts:** The environment server utilizes an internal instance of the `OpenAI()` python client to perform Server-Side Evaluation of Task 3. Standard open-source inference servers (like Hugging Face rate limits) may timeout. The server restricts this to `30.0s`. If hitting limits, this falls back to the heuristic floor score `0.10`.
- **Docker Mount Issues (`from_docker_image`):** If running with the ENV variable `IMAGE_NAME` present, `inference.py` attempts to orchestrate a pure spun up containerized server instance instead of listening locally on `:8000`. If this crashes, verify your built container tag strictly matches `IMAGE_NAME`, or disable the env var to test raw processes locally.
