import React, { useRef, useState } from 'react';
import './App.css';
import Tree from './Tree';
import type { TreeNode } from './Tree';



// Helper function to deep clone a tree
function cloneTree<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

const initialTree: TreeNode = {
  id: 'root',
  label: 'Root',
  description: 'This is the root node.',
  children: [
    { id: '1', label: 'Child 1', description: 'First child.' },
    { id: '2', label: 'Child 2', description: 'Second child.', children: [
      { id: '2-1', label: 'Subchild 2-1', description: 'First subchild.' },
    ] },
  ],
};

// Function to add a child to a node by id
function addChildToTree(tree: TreeNode, parentId: string, child: TreeNode): TreeNode {
  if (tree.id === parentId) {
    return {
      ...tree,
      children: [...(tree.children || []), child],
    };
  }
  if (tree.children) {
    return {
      ...tree,
      children: tree.children.map(c => addChildToTree(c, parentId, child)),
    };
  }
  return tree;
}


function App() {
  const [messages, setMessages] = useState<string[]>([]);
  const [input, setInput] = useState('');
  const [tree, setTree] = useState<TreeNode>(initialTree);
  const ws = useRef<WebSocket | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  React.useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  const connect = () => {
    if (ws.current) return;
    ws.current = new WebSocket('ws://localhost:8000/ws');
    ws.current.onmessage = (event: MessageEvent) => {
      setMessages((msgs) => [...msgs, "tot: " + event.data]);
    };
    ws.current.onclose = () => {
      ws.current = null;
    };
  };

  connect();

  const sendMessage = () => {
    if (ws.current && input) {
      ws.current.send(input);
      setMessages((msgs) => [...msgs, "You: " + input]);
      setInput('');
    }
  };

  // Function to change the tree state to a new tree
  const setTreeTo = (newTree: TreeNode) => {
    setTree(cloneTree(newTree));
  };

  return (
    <>
    <div className="container">
      <div className="chatbox">
        <h1 className="totTitle">TOT</h1>
        <div
          className="messages"
          style={{ display: messages.length === 0 ? 'none' : undefined }}
        >
          {messages.map((msg, i) => (
            <div
              key={i}
              className={
                'message ' + (i % 2 === 0 ? 'user' : 'bot')
              }
            >
              {msg}
            </div>
          ))}
        </div>
        <div className="inputRow">
          <input
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Type a message..."
            className="input"
            onKeyDown={e => { if (e.key === 'Enter') sendMessage(); }}
          />
          <button
            onClick={sendMessage}
            disabled={!ws.current || !input}
            className="button"
          >
            Send
          </button>
        </div>
      </div>
  <Tree data={tree} />
    </div>
      </>
  );
}

export default App;
