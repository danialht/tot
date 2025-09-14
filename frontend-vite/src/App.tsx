import React, { useRef, useState } from "react";
import "./App.css";
import Tree from "./Tree";
import type { TreeNode } from "./Tree";
import PopupWindow from "./PopupWindow";

const initialTree: TreeNode = {
  id: "Idea",
  label: "Idea",
  description: "This is where a great journey begins.",
  children: [
    // { id: '1', label: 'Child 1', description: 'First child.', hidden: false, children: [] },
    // { id: '2', label: 'Child 2', description: 'Second child.', hidden: false, children: [
    //   { id: '2-1', label: 'Subchild 2-1', description: 'First subchild.', hidden: false, children: [] },
    // ] },
  ],
};

function makeTreeFromData(obj: any): TreeNode {
  // console.log('DEBUG');
  // console.log(obj);
  const node: TreeNode = {
    id: obj.id,
    label: "Idea", // obj.subproblem || 'No Subproblem',
    description:
      "Chain of thought: \n" +
        obj.chain_of_thought_text +
        "\nSubproblem:\n" +
        obj.subproblem || "",
    children: [],
    hidden: true,
  };
  if (obj.children && Array.isArray(obj.children)) {
    node.children = obj.children.map((childObj: any) =>
      makeTreeFromData(childObj)
    );
  }
  return node;
}

type PopupWindowProps = {
  title: string;
  description: string;
};

function App() {
  const [messages, setMessages] = useState<string[]>([]);
  const [input, setInput] = useState("");
  const [lastUserPrompt, setLastUserPrompt] = useState<string>("");
  const [tree, setTree] = useState<TreeNode>(initialTree);
  const [hasSentFirstMessage, setHasSentFirstMessage] = useState(false);
  const ws = useRef<WebSocket | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const [hasFirstPromptBeenAnswered, setHasFirstPromptBeenAnswered] = useState(false);
  const [popupWindowProps, setPopupWindowProps] =
    useState<PopupWindowProps | null>(null);

  const scrollChatToBottom = () => {
    if (messagesEndRef.current) {
      // Use rAF to ensure DOM is updated before scrolling
      requestAnimationFrame(() => {
        messagesEndRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "end",
        });
      });
    }
  };

  async function animateTree(tree: TreeNode) {
    // console.log('Animating tree', tree);
    const nodes: TreeNode[] = [];
    function bfsAndShowNodes(tree: TreeNode) {
        const queue: TreeNode[] = [tree];
        while (queue.length > 0) {
            const currentNode = queue.shift()!;
            nodes.push(currentNode);
            if (currentNode.children) {
                queue.push(...currentNode.children);
            }
        }
    }
    bfsAndShowNodes(tree);
    for (const node of nodes) {
        node.hidden = true;
    }
    for(const node of nodes) {
        node.hidden = false;
        setTree({ ...tree });
        console.log('Showing node', node);
        const delay = Math.floor(Math.random() * (500 - 100 + 1)) + 100;
        await new Promise((resolve) => setTimeout(resolve, delay));
    }
}

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
    ws.current = new WebSocket("ws://localhost:8000/ws");
    ws.current.onmessage = (event: MessageEvent) => {
        const obj = JSON.parse(event.data);
        if (!hasFirstPromptBeenAnswered) setHasFirstPromptBeenAnswered(true);
        setMessages((msgs) => [...msgs, "tot: " + obj["output"]]);
        const tree = makeTreeFromData(obj.tree);
        // console.log('Received tree from server:', tree);    
        animateTree(tree);
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
      setInput("");
      if (!hasSentFirstMessage) setHasSentFirstMessage(true);
    }
  };

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
          <div
            className={`chatbox${
              hasSentFirstMessage ? " chatbox--move-up" : ""
            }`}
          >
            <h1 className="totTitle">TOT</h1>
            <div
              className="messages"
              style={{ display: messages.length === 0 ? "none" : undefined }}
            >
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={"message " + (i % 2 === 0 ? "user" : "bot")}
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
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type a message..."
                className="input"
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    sendMessage();
                  } else if (
                    e.key === "ArrowUp" &&
                    document.activeElement === inputRef.current
                  ) {
                    // Find the last user prompt from messages
                    if (lastUserPrompt) {
                      setInput(lastUserPrompt);
                      // Move cursor to end
                      setTimeout(() => {
                        inputRef.current?.setSelectionRange(
                          lastUserPrompt.length,
                          lastUserPrompt.length
                        );
                      }, 0);
                    } else {
                      // Fallback: search messages for last user prompt
                      const last = [...messages]
                        .reverse()
                        .find((m) => m.startsWith("You: "));
                      if (last) {
                        const prompt = last.slice(5);
                        setInput(prompt);
                        setTimeout(() => {
                          inputRef.current?.setSelectionRange(
                            prompt.length,
                            prompt.length
                          );
                        }, 0);
                      }
                    }
                    e.preventDefault();
                  } else if (
                    e.key === "ArrowDown" &&
                    document.activeElement === inputRef.current
                  ) {
                    setInput("");
                    e.preventDefault();
                  }
                }}
              />
              <div className="buttonDropdownStack">
                <button
                  onClick={sendMessage}
                  disabled={!ws.current || !input}
                  className="button"
                >
                  Send
                </button>
                <select className="dropdown">
                  <option value="option1">Normal</option>
                  <option value="option2">Swift</option>
                  <option value="option3">Genius</option>
                </select>
              </div>
            </div>
          </div>
            {hasFirstPromptBeenAnswered && (
            <div
              className={`tree-transition${
              hasFirstPromptBeenAnswered ? " tree-transition--down" : ""
              }`}
            >
              <Tree
              data={tree}
              onNodeClick={(node) =>
                setPopupWindowProps({
                title: node.label,
                description: node.description || "",
                })
              }
              />
            </div>
            )}
        </div>
      )}
    </>
  );
}

export default App;
