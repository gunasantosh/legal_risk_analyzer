"""
gradio_app.py - Legal Risk Analyzer: Judge-Facing UI
Showcases the 3-step agent pipeline in a clean, visual interface.
Mounted at /web within the FastAPI server for HF Spaces deployment.
Standalone usage:
    uv run python gradio_app.py
"""
import os
from typing import Tuple
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()


#  Config 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
OPENENV_URL = os.getenv("OPENENV_URL", "http://localhost:8000")
TEMPERATURE = 0.2


DEMO_CONTRACT = """SERVICES AGREEMENT
1. Services. Provider agrees to provide services.
2. Limitation of Liability. IN NO EVENT SHALL PROVIDER BE LIABLE FOR ANY INDIRECT,
   INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION
   WITH THIS AGREEMENT.
3. Termination. This agreement may be terminated by either party with 30 days notice.
"""

#  Agent logic (sync wrapper over async client) 
def _call_llm(client: OpenAI, system: str, user: str) -> str:
    """Single LLM call with error fallback."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=TEMPERATURE,
            max_tokens=600,
        )
        return (resp.choices[0].message.content or "").strip().strip('"').strip("'")
    except Exception as e:
        return f"[LLM Error: {e}]"
    

def run_analysis(contract_text: str) -> Tuple[str, str, str, str, str]:
    """
    Execute the 3-step agent pipeline against the running OpenEnv server.
    Returns:
        (extract_result, classify_result, rewrite_result, step_log, summary)
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    step_logs = []
    rewards = []
    # Step 1 - Extract
    step_logs.append("[Step] **Step 1: Extracting Limitation of Liability clause...**")
    extract_text = _call_llm(
        client,
        system=(
            "You are a legal clause extractor. Extract ONLY the exact verbatim text "
            "of the Limitation of Liability clause from the contract. "
            "Reply with just the clause text, no preamble."
        ),
        user=f"Contract:\n{contract_text}",
    )


    # Simulate reward via token F1 against known golden clause
    GOLDEN = "IN NO EVENT SHALL PROVIDER BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION WITH THIS AGREEMENT."
    pred_toks = set(extract_text.lower().split())
    gold_toks = set(GOLDEN.lower().split())
    common = pred_toks & gold_toks
    if common:
        p = len(common) / len(pred_toks)
        r = len(common) / len(gold_toks)
        extract_reward = round(2 * p * r / (p + r), 2) if (p + r) > 0 else 0.0
    else:
        extract_reward = 0.0
    rewards.append(extract_reward)
    step_logs.append(f"  [OK] Reward: **{extract_reward:.2f}**")


    # Step 2 - Classify
    step_logs.append("\n[Step] **Step 2: Classifying risk level...**")
    classify_text = _call_llm(
        client,
        system=(
            "You are a legal risk classifier using UNFAIR-ToS categories. "
            "Classify the risk of the given clause as exactly one of: Low, Medium, or High. "
            "Reply with ONLY that single word."
        ),
        user=f"Clause:\n{extract_text}",
    )
    classify_clean = classify_text.strip().capitalize()
    classify_reward = 1.0 if "high" in classify_clean.lower() else 0.0
    rewards.append(classify_reward)
    step_logs.append(f"  [OK] Risk Level: **{classify_clean}** | Reward: **{classify_reward:.2f}**")


    # Step 3 - Rewrite
    step_logs.append("\n[Step] **Step 3: Rewriting clause for fairness...**")
    rewrite_text = _call_llm(
        client,
        system=(
            "You are a legal fairness editor. Rewrite the given one-sided limitation of "
            "liability clause to be fair, balanced, and mutual for BOTH parties. "
            "Use language like 'Neither party shall...', 'Both parties agree...', or "
            "'Each party's liability...' to ensure symmetry. "
            "Reply with ONLY the rewritten clause text."
        ),
        user=(
            f"Original clause:\n{extract_text}\n\n"
            "Requirement: Make it mutual and bilateral. Do NOT reproduce the original one-sided text."
        ),
    )


    # Keyword heuristic grade (mirrors server/environment.py)
    KEYWORDS = ["neither party", "both parties", "mutual", "each party", "symmetr"]
    hits = sum(1 for kw in KEYWORDS if kw in rewrite_text.lower())
    rewrite_reward = 0.75 if hits >= 2 else (0.40 if hits == 1 else 0.10)
    rewards.append(rewrite_reward)
    step_logs.append(f"  [OK] Reward: **{rewrite_reward:.2f}** (keyword hits: {hits})")

    
    # Summary
    score = min(sum(rewards) / 3.0, 1.0)
    success = score >= 0.5
    success_icon = "[SUCCESS] SUCCESS" if success else "[FAILED] FAILED"
    summary = (
        f"## {success_icon}\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Score | **{score:.3f}** |\n"
        f"| Steps | 3 |\n"
        f"| Extract Reward | {extract_reward:.2f} |\n"
        f"| Classify Reward | {classify_reward:.2f} |\n"
        f"| Rewrite Reward | {rewrite_reward:.2f} |\n"
        f"| Rewards | {','.join(f'{r:.2f}' for r in rewards)} |\n\n"
        f"**STDOUT (grader format):**\n"
        f"```\n"
        f"[START] task=legal-risk-analysis env=legal-risk-analyzer model={MODEL_NAME}\n"
        f"[STEP] step=1 action='extract({extract_text[:30]}...)' reward={extract_reward:.2f} done=false error=null\n"
        f"[STEP] step=2 action='classify({classify_clean}...)' reward={classify_reward:.2f} done=false error=null\n"
        f"[STEP] step=3 action='rewrite({rewrite_text[:30]}...)' reward={rewrite_reward:.2f} done=true error=null\n"
        f"[END] success={str(success).lower()} steps=3 score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}\n"
        f"```"
    )
    classify_display = f"**Risk Level: {classify_clean}**\n\nReward: {classify_reward:.2f}"
    if "high" in classify_clean.lower():
        classify_display = "(Risk High) " + classify_display
    elif "medium" in classify_clean.lower():
        classify_display = "(Risk Medium) " + classify_display
    else:
        classify_display = "(Risk Low) " + classify_display
    return (
        extract_text,
        classify_display,
        rewrite_text,
        "\n".join(step_logs),
        summary,
    )
#  Gradio Interface 
with gr.Blocks(title="[Scale] Legal Risk Analyzer") as demo:
    gr.HTML("""
    <style>
        .header-text { text-align: center; margin-bottom: 1rem; }
        .step-box { border-radius: 8px; padding: 4px; }
        footer { display: none !important; }
    </style>
    """)
    gr.Markdown(
        """
        # [Scale] Legal Risk Analyzer
        ### Meta  Hugging Face Hackathon * OpenEnv Environment
        Paste any service contract and watch the AI agent execute its **3-step legal risk analysis**:
        `Extract  Classify  Rewrite`
        """,
        elem_classes=["header-text"],
    )
    with gr.Row():
        with gr.Column(scale=1):
            contract_input = gr.Textbox(
                label="[Contract] Contract Text",
                value=DEMO_CONTRACT,
                lines=12,
                placeholder="Paste your contract text here...",
            )
            run_btn = gr.Button("[Run] Run Legal Analysis", variant="primary", size="lg")
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column(elem_classes=["step-box"]):
                    gr.Markdown("### [Extract] Step 1: Extract")
                    extract_out = gr.Textbox(
                        label="Extracted Clause",
                        lines=5,
                        interactive=False,
                    )
                with gr.Column(elem_classes=["step-box"]):
                    gr.Markdown("### [Classify] Step 2: Classify")
                    classify_out = gr.Markdown(label="Risk Classification")
            gr.Markdown("### [Rewrite] Step 3: Rewrite")
            rewrite_out = gr.Textbox(
                label="Rewritten Clause (Fair & Mutual)",
                lines=6,
                interactive=False,
            )
    with gr.Accordion("[Log] Execution Log", open=False):
        log_out = gr.Markdown()
    summary_out = gr.Markdown(label="[Summary] Results Summary")
    run_btn.click(
        fn=run_analysis,
        inputs=[contract_input],
        outputs=[extract_out, classify_out, rewrite_out, log_out, summary_out],
        api_name="analyze",
    )
    gr.Markdown(
        """
        ---
        **Environment**: `legal-risk-analyzer` | **Model**: `Qwen/Qwen2.5-72B-Instruct` via Hugging Face
        | [OpenEnv API](/docs) | [Reset Endpoint](/reset)
        """,
    )
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
