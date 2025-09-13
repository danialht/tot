import React, { useRef, useState } from 'react';
import './App.css';
import Tree from './Tree';
import type { TreeNode } from './Tree';
import PopupWindow from './PopupWindow';

const initialTree: TreeNode = {
  id: 'Idea',
  label: 'Idea',
  description: 'This is where a great journey begins.',
  children: [
    // { id: '1', label: 'Child 1', description: 'First child.' },
    // { id: '2', label: 'Child 2', description: 'Second child.', children: [
    //   { id: '2-1', label: 'Subchild 2-1', description: 'First subchild.' },
    // ] },
  ],
};

function makeTreeFromData(obj: any): TreeNode {
  // console.log('DEBUG');
  // console.log(obj);
  const node: TreeNode = {
    id: obj.id,
    label: 'Idea', // obj.subproblem || 'No Subproblem',
    description: ('Chain of thought: \n' + obj.chain_of_thought_text + '\nSubproblem:\n' + obj.subproblem) || '',
    children: [],
  };
  if (obj.children && Array.isArray(obj.children)) {
    node.children = obj.children.map((childObj: any) => makeTreeFromData(childObj));
  }
  return node;
}

type PopupWindowProps = {
  title: string;
  description: string;
};

function App() {
  const [messages, setMessages] = useState<string[]>([]);
  const [input, setInput] = useState('');
  const [lastUserPrompt, setLastUserPrompt] = useState<string>('');
  const [tree, setTree] = useState<TreeNode>(initialTree);
  const [hasSentFirstMessage, setHasSentFirstMessage] = useState(false);
  const ws = useRef<WebSocket | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const [popupWindowProps, setPopupWindowProps] = useState<PopupWindowProps | null>(null);

  const scrollChatToBottom = () => {
    if (messagesEndRef.current) {
      // Use rAF to ensure DOM is updated before scrolling
      requestAnimationFrame(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
      });
    }
  };

  React.useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  // Always scroll to bottom when messages change
  React.useEffect(() => {
    scrollChatToBottom();
  }, [messages]);

  const connect = () => {
    if (ws.current) return;
    ws.current = new WebSocket('ws://localhost:8000/ws');
    ws.current.onmessage = (event: MessageEvent) => {
      const obj = JSON.parse(event.data);
      setMessages((msgs) => [...msgs, "tot: " + obj['output']]);
      // console.log(obj);  
      // console.log();
      const tree = makeTreeFromData(obj.tree);
      console.log(tree);
      setTree(tree);
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
      setLastUserPrompt(input);
      setInput('');
      if (!hasSentFirstMessage) setHasSentFirstMessage(true);
    }
  };

  // // Function to change the tree state to a new tree
  // const setTreeTo = (newTree: TreeNode) => {
  //   setTree(cloneTree(newTree));
  // };

  

  return (
    <>
      {popupWindowProps ? (
        <PopupWindow
          isOpen={true}
          onClose={() => setPopupWindowProps(null)}
          title={popupWindowProps.title}
          description={popupWindowProps.description}
        />
      ) : (
        <div className="container">
          <div className={`chatbox${hasSentFirstMessage ? ' chatbox--move-up' : ''}`}>
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
              <div ref={messagesEndRef} />
            </div>
            <div className="inputRow">
              <input
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder="Type a message..."
                className="input"
                onKeyDown={e => {
                  if (e.key === 'Enter') {
                    sendMessage();
                  } else if (e.key === 'ArrowUp' && document.activeElement === inputRef.current) {
                    // Find the last user prompt from messages
                    if (lastUserPrompt) {
                      setInput(lastUserPrompt);
                      // Move cursor to end
                      setTimeout(() => {
                        inputRef.current?.setSelectionRange(lastUserPrompt.length, lastUserPrompt.length);
                      }, 0);
                    } else {
                      // Fallback: search messages for last user prompt
                      const last = [...messages].reverse().find(m => m.startsWith('You: '));
                      if (last) {
                        const prompt = last.slice(5);
                        setInput(prompt);
                        setTimeout(() => {
                          inputRef.current?.setSelectionRange(prompt.length, prompt.length);
                        }, 0);
                      }
                    }
                    e.preventDefault();
                  } else if (e.key === 'ArrowDown' && document.activeElement === inputRef.current) {
                    setInput('');
                    e.preventDefault();
                  }
                }}
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
          <div className={`tree-transition${hasSentFirstMessage ? ' tree-transition--down' : ''}`}>
            <Tree
              data={tree}
              onNodeClick={node => setPopupWindowProps({
                title: node.label,
                description: node.description || ''
              })}
            />
          </div>
        </div>
      )}
    </>
  );
}

export default App;
