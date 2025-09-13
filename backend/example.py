#!/usr/bin/env python3
"""
Examples of publicly accessible attributes in the async TreeOfThoughtSolver.
Demonstrates how to access tree state, nodes, thoughts, and progress during/after solving.
"""

import asyncio
from tree import SolverConfig, create_default_solver

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

async def demonstrate_public_access():
    """Demonstrate all publicly accessible attributes and methods."""
    
    print("🌳 TreeOfThoughtSolver - Public API Examples")
    print("=" * 60)
    
    # Create solver
    
    
    

    
    tree_state = solver.get_tree_state()
    print("solver.get_tree_state() returns:")
    for key, value in tree_state.items():
        if isinstance(value, dict) and value:
            print(f"   {key}: {type(value).__name__} with {len(value)} keys")
        elif isinstance(value, list):
            print(f"   {key}: list with {len(value)} items")
        else:
            print(f"   {key}: {value}")
    print()
    
    # ============================================================================
    # 3. ROOT NODE ATTRIBUTES
    # ============================================================================
    if solver.root_node:
        print("🌱 3. ROOT NODE ATTRIBUTES")
        print("-" * 40)
        
        root = solver.root_node
        print("Root node public attributes:")
        print(f"   root.state.subproblem_text: '{root.state.subproblem_text}'")
        print(f"   root.depth: {root.depth}")
        print(f"   root.children: {len(root.children)} children")
        print(f"   root.value_estimate: {root.value_estimate}")
        print(f"   root.terminal_status: {root.terminal_status}")
        print(f"   root.processing_status: {root.processing_status}")
        print()
        
        print("Root node exposed attributes:")
        print(f"   root.input_context: {root.input_context}")
        print(f"   root.generated_thoughts: {len(root.generated_thoughts)} thoughts")
        print(f"   root.thought_scores: {len(root.thought_scores)} scores")
        print()
        
        # ========================================================================
        # 4. CHAIN OF THOUGHT ACCESS
        # ========================================================================
        print("🧠 4. CHAIN OF THOUGHT ACCESS")
        print("-" * 40)
        
        cot = root.get_chain_of_thought()
        print(f"root.get_chain_of_thought(): {len(cot)} thoughts")
        for i, thought in enumerate(cot):
            print(f"   {i+1}. '{thought.text[:50]}...' (confidence: {thought.confidence})")
        print()
        
        # ========================================================================
        # 5. GENERATED THOUGHTS DETAILS
        # ========================================================================
        print("💭 5. GENERATED THOUGHTS DETAILS")
        print("-" * 40)
        
        print(f"root.generated_thoughts: {len(root.generated_thoughts)} thoughts")
        for i, thought in enumerate(root.generated_thoughts):
            score = root.thought_scores.get(thought.text, "N/A")
            print(f"   {i+1}. Text: '{thought.text[:40]}...'")
            print(f"       Rationale: '{thought.rationale[:30]}...'")
            print(f"       Confidence: {thought.confidence}")
            print(f"       Intent: {thought.intent}")
            print(f"       Score: {score}")
            if thought.candidate:
                print(f"       Candidate: '{thought.candidate}'")
            print()
        
        # ========================================================================
        # 6. CHILDREN WITH THOUGHTS
        # ========================================================================
        print("👥 6. CHILDREN WITH THOUGHTS")
        print("-" * 40)
        
        children_info = root.get_children_with_thoughts()
        print(f"root.get_children_with_thoughts(): {len(children_info)} children")
        for i, child_info in enumerate(children_info):
            child = child_info["child"]
            incoming_thought = child_info["incoming_thought"]
            thought_score = child_info["thought_score"]
            
            print(f"   Child {i+1}:")
            print(f"     Depth: {child.depth}")
            print(f"     Processing status: {child.processing_status}")
            print(f"     Generated thoughts: {len(child.generated_thoughts)}")
            if incoming_thought:
                print(f"     Incoming thought: '{incoming_thought.text[:40]}...'")
                print(f"     Thought score: {thought_score}")
            print()
    
    # ============================================================================
    # 7. NODE SERIALIZATION
    # ============================================================================
    if solver.root_node:
        print("📄 7. NODE SERIALIZATION")
        print("-" * 40)
        
        node_dict = solver.root_node.to_dict()
        print("root.to_dict() returns:")
        for key, value in node_dict.items():
            if isinstance(value, list) and value:
                print(f"   {key}: list with {len(value)} items")
            elif isinstance(value, dict) and value:
                print(f"   {key}: dict with {len(value)} keys")
            else:
                print(f"   {key}: {value}")
        print()
    
    # ============================================================================
    # 8. SOLUTION PATH
    # ============================================================================
    print("🛤️  8. SOLUTION PATH")
    print("-" * 40)
    
    solution_path = solver.get_solution_path()
    print(f"solver.get_solution_path(): {len(solution_path)} nodes in path")
    for i, path_node in enumerate(solution_path):
        print(f"   Step {i+1}: {len(path_node['chain_of_thought'])} thoughts in chain")
    print()
    
    # ============================================================================
    # 9. REAL-TIME ACCESS DURING SOLVING
    # ============================================================================
    print("⏱️  9. REAL-TIME ACCESS PATTERN")
    print("-" * 40)
    
    print("Example pattern for real-time monitoring:")
    print("""
    # Start solving in background
    solve_task = asyncio.create_task(solver.solve(problem))
    
    # Monitor progress
    while solver.is_solving:
        state = solver.get_tree_state()
        print(f"Depth: {state['current_depth']}, Frontier: {state['frontier_size']}")
        
        if solver.current_frontier:
            for node in solver.current_frontier:
                print(f"  Node: {node.processing_status}, Thoughts: {len(node.generated_thoughts)}")
        
        await asyncio.sleep(0.1)  # Check every 100ms
    
    result = await solve_task
    """)
    
    # Clean up
    try:
        await solver.thought_generator.close()
        if hasattr(solver.idea_validator, 'close'):
            await solver.idea_validator.close()
        if hasattr(solver.solution_validator, 'close'):
            await solver.solution_validator.close()
    except:
        pass
    
    print("✨ All public attributes and methods demonstrated!")


if __name__ == "__main__":
    async def testing():
        config = SolverConfig(beam_width=3, max_depth=3)
        solver = create_default_solver(config, use_llm_validators=False)
        try:
            problem = 'We fill a glass with water up to the brim. we turn it upsidedown. give an estimate for how many water molecules are in the glass'
            # Start async solving (will fail without API key, but shows structure)
            result = await solver.solve(problem, log=False)
            breakpoint()
            # print(solver.root_node)

            # print(result)
            # print("After solving:")
            # print(f"   solver.root_node: {solver.root_node is not None}")
            # print(f"   solver.current_frontier: {len(solver.current_frontier)} nodes")
            # print(f"   solver.best_node: {solver.best_node is not None}")
            # print(f"   solver.expanded_nodes: {solver.expanded_nodes}")
            # print(f"   solver.current_depth: {solver.current_depth}")
            # print(f"   solver.is_solving: {solver.is_solving}")
            # print(f"   solver.solving_complete: {solver.solving_complete}")
            # print(f"   solver.trace: {len(solver.trace)} entries")
            
        except Exception as e:
            raise RuntimeError(f"Error: {e}")
    asyncio.run(
        testing()
    ) 