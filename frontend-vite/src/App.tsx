import React, { useRef, useState } from 'react';
import './App.css';
import Tree from './Tree';
import type { TreeNode } from './Tree';


const sampleTree: TreeNode = {
  id: 'root',
  label: 'Root',
  children: [
    { id: '1', label: 'Child 1' },
    { id: '2', label: 'Child 2', children: [
      { id: '2-1', label: 'Grandchild 2-1' },
      { id: '2-2', label: 'Grandchild 2-2' },
    ] },
    { id: '3', label: 'Child 3' },
  ],
};

function App() {
  const [messages, setMessages] = useState<string[]>([]);
  const [input, setInput] = useState('');
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

      <Tree data={sampleTree} />
    </div>
      </>
  );
}

export default App;
