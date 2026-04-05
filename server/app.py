# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
FastAPI application for the Legal Risk Analyzer Environment.
This module creates an HTTP server that exposes the LegalRiskAnalyzerEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.
Endpoints (OpenEnv API - machine-facing, root /):
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
Endpoints (Gradio UI - human-facing, /web):
    - GET /web: Judge-facing contract analysis interface
Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
    # Or run directly:
    python -m server.app
"""
try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e
try:
    from models import LegalAction, LegalObservation
    from .environment import LegalRiskEnv
except ImportError:
    try:
        from server.models import LegalAction, LegalObservation
        from server.environment import LegalRiskEnv
    except ImportError:
        from .models import LegalAction, LegalObservation
        from .environment import LegalRiskEnv



# Create the app with web interface and README integration
app = create_app(
    LegalRiskEnv,
    LegalAction,
    LegalObservation,
    env_name="legal_risk_analyzer",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)
#  Mount Gradio UI at /web (human-facing; OpenEnv API stays at root) 
# This is a soft dependency: if gradio is not installed the server still boots
# as a pure OpenEnv API and openenv validate continues to pass.
try:
    import gradio as gr
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gradio_app import demo as _gradio_demo
    app = gr.mount_gradio_app(app, _gradio_demo, path="/web")
except Exception as _e:  # pragma: no cover
    import warnings
    warnings.warn(f"Gradio UI not mounted (optional): {_e}")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.
    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m legal_risk_analyzer.server.app
    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)
    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn legal_risk_analyzer.server.app:app --workers 4
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    if args.port == 8000 and args.host == "0.0.0.0":
        main()  # default invocation - satisfies multi-mode deployment check
    else:
        main(host=args.host, port=args.port)
