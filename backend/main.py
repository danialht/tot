from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import json

import asyncio
from tree import SolverConfig, create_default_solver

from dotenv import load_dotenv
load_dotenv()
TEST_MODE = False
app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Node:
    def __init__(self, id, subproblem, chain_of_thought_text, incoming_thoughts_texts, children_count, processing_status, terminal_status):
        self.id = id
        self.subproblem = subproblem
        self.chain_of_thought_text = chain_of_thought_text
        self.incoming_thoughts_texts = incoming_thoughts_texts
        self.children_count = children_count
        self.processing_status = processing_status
        self.terminal_status = terminal_status
        self.children = []
    

    def addChild(self, child_node):
        self.children.append(child_node)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # When accepted a socket, make a solver instance
    # Parse effort from query params
    try:
        query_params = websocket.query_params
        effort_param = (query_params.get("effort") or "").lower()
        if effort_param not in ("low", "medium", "high"):
            effort_param = "medium"
    except Exception:
        effort_param = "medium"
    
    print(f"[WS] Connected. effort={effort_param}")
    config = SolverConfig(beam_width=3, max_depth=3, reasoning_effort=effort_param)
    solver = create_default_solver(config, use_llm_validators=False)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"[WS] Received prompt len={len(data)}")
            if TEST_MODE:
                print("[WS] TEST_MODE active; sending static tree")
                test_leaf2 = {
                    'id': 'test_leaf_lskjdf',
                    'chain_of_thought_text': 'some chain of thought',
                    'description': 'This is a test leaf node',
                    'subproblem': 'A subproblem',
                    'children': []
                }
                test_leaf = {
                    'id': 'test_leaf',
                    'chain_of_thought_text': 'some chain of thought',
                    'description': 'This is a test leaf node',
                    'subproblem': 'A subproblem',
                    'children': []
                }
                test_node = {
                    'id': 'test_node',
                    'chain_of_thought_text': 'some chain of thought',
                    'description': 'This is a test node',
                    'subproblem': 'A subproblem',
                    'children': [test_leaf, test_leaf2]
                }
                await websocket.send_text(
                    json.dumps({'output': 'Test mode: not solving',
                                  'tree': {'id': 'test_root', 'subproblem': 'some subproblem', 'chain_of_thought_text': 'some chain of thought', 'description': 'This is a test root node', 'children': [test_node]}})
                )
                continue
            
            # Run solve in background and stream interim updates
            print("[WS] Starting solve...")
            solve_task = asyncio.create_task(solver.solve(data, log=False))
            last_sent_expanded = -1
            try:
                while not solve_task.done():
                    await asyncio.sleep(0.2)
                    if solver.root_node and solver.expanded_nodes != last_sent_expanded:
                        last_sent_expanded = solver.expanded_nodes
                        try:
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        'output': '',  # interim update without final text
                                        'tree': solver.root_node.to_dict()
                                    }
                                )
                            )
                            print(f"[WS] Interim tree sent. expanded_nodes={last_sent_expanded}")
                        except Exception as e:
                            print(f"[WS] Interim send failed: {e}")
                            break
                # Await final result
                output = await solve_task
                print("[WS] Solve finished.")
            except Exception as e:
                print(f"[WS] Solve loop error: {e}")
                raise
            
            if not solver.root_node:
                await websocket.send_text(f"Solver not initialized")
                print("[WS] No root node after solve.")
                continue
            
            try:
                await websocket.send_text(
                    json.dumps(
                        {
                            'output': f'{output.final_solution_text}',
                            'tree': solver.root_node.to_dict()
                        }
                    )
                )
                print("[WS] Final result sent.")
            except Exception as e:
                print(f"[WS] Final send failed: {e}")
            # Process the received message
            # await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect as e:
        print(f"[WS] Disconnected: code={getattr(e, 'code', None)}")
    except Exception as e:
        print(f"[WS] Unexpected error: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
