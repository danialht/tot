import React, { useRef, useState } from "react";
import "./App.css";
import Tree from "./Tree";
import type { TreeNode } from "./Tree";
import PopupWindow from "./PopupWindow";
import MathText from "./MathText";

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

function extractAssumeIdeaSegments(text: string | undefined | null): string[] {
  if (!text || typeof text !== 'string') return [];
  const pattern = /Assume\/Idea:\s*([\s\S]*?)(?=\n?\s*Why\b|$)/gi;
  const segments: string[] = [];
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text)) !== null) {
    const raw = match[1] || '';
    const oneLine = raw.replace(/\s+/g, ' ').trim();
    if (oneLine) segments.push(oneLine);
  }
  return segments;
}

function truncateForLabel(label: string, max = 60): string {
  if (label.length <= max) return label;
  return label.slice(0, max - 1).trimEnd() + '…';
}

function makeTreeFromData(obj: any, depth: number = 0): TreeNode {
  const segments = extractAssumeIdeaSegments(obj?.chain_of_thought_text);
  const idx = Math.max(0, depth - 1);

  // Prefer backend-provided concise title/idea, with safe fallbacks
  const rawLabel = depth === 0
    ? (obj?.branch_title || obj?.subproblem)
    : (obj?.branch_title || obj?.branch_idea || obj?.incoming_thought_text || segments[idx] || obj?.subproblem);
  const computedLabel = truncateForLabel(String(rawLabel || 'Idea'));

  // Build a compact, conditional description (hide empty sections)
  const parts: string[] = [];
  const add = (title: string, value: any, multiline: boolean = false) => {
    if (value === null || value === undefined) return;
    const str = String(value).trim();
    if (!str) return;
    parts.push(multiline ? `${title}:\n${str}` : `${title}: ${str}`);
  };

  // Idea for non-root nodes
  if (depth > 0) {
    add('Idea', obj?.branch_idea || obj?.incoming_thought_text);
  }
  // Solution and Reasoning if present
  add('Solution', obj?.solution_text || obj?.solution_package);
  add('Reasoning', obj?.reasoning_text, true);
  // Removed Chain of thought from description
  // Always show subproblem at root; optional otherwise
  if (depth === 0) {
    add('Subproblem', obj?.subproblem, true);
  }

  const node: TreeNode = {
    id: obj.id,
    label: computedLabel,
    description: parts.join('\n\n'),
    children: [],
    hidden: true,
  };
  if (obj.children && Array.isArray(obj.children)) {
    node.children = obj.children.map((childObj: any) =>
      makeTreeFromData(childObj, depth + 1)
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
  const [hasFirstPromptBeenAnswered, setHasFirstPromptBeenAnswered] = useState(false);
  const [isBotTyping, setIsBotTyping] = useState(false);
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const ws = useRef<WebSocket | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
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

  // Apply theme to <html> for CSS variable overrides
  React.useEffect(() => {
    const root = document.documentElement;
    if (theme === 'light') {
      root.setAttribute('data-theme', 'light');
    } else {
      root.removeAttribute('data-theme');
    }
  }, [theme]);

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
        setIsBotTyping(false);
        setMessages((msgs) => [...msgs, "tot: " + obj["output"]]);
        const tree = makeTreeFromData(obj.tree, 0);
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
      setIsBotTyping(true);
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
          children={null}
        />
      ) : (
        <div className="container">
          {!hasFirstPromptBeenAnswered ? (
          <div
            className={`chatbox${
              hasSentFirstMessage && !hasFirstPromptBeenAnswered ? " chatbox--move-up" : ""
            }`}
          >
            <button
              className="themeToggle"
              onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
              aria-label="Toggle theme"
              title={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}
            >
              {theme === 'dark' ? '☀️' : '🌙'}
            </button>
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
                  <MathText text={msg} />
                </div>
              ))}
              {isBotTyping && (
                <div className="message bot typingIndicator" aria-live="polite">
                  <span className="typingDot" />
                  <span className="typingDot" />
                  <span className="typingDot" />
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
            {hasSentFirstMessage && !hasFirstPromptBeenAnswered && (
              <div className="skeleton">
                <div className="skeleton-line" style={{ width: '70%' }} />
                <div className="skeleton-line" style={{ width: '55%' }} />
                <div className="skeleton-line" style={{ width: '62%' }} />
              </div>
            )}
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
          ) : (
            <div className="splitLayout">
              <div className="leftPane leftPane--enter">
                <div className="chatbox">
                  <button
                    className="themeToggle"
                    onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
                    aria-label="Toggle theme"
                    title={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}
                  >
                    {theme === 'dark' ? '☀️' : '🌙'}
                  </button>
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
                        <MathText text={msg} />
                      </div>
                    ))}
                    {isBotTyping && (
                      <div className="message bot typingIndicator" aria-live="polite">
                        <span className="typingDot" />
                        <span className="typingDot" />
                        <span className="typingDot" />
                      </div>
                    )}
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
                          if (lastUserPrompt) {
                            setInput(lastUserPrompt);
                            setTimeout(() => {
                              inputRef.current?.setSelectionRange(
                                lastUserPrompt.length,
                                lastUserPrompt.length
                              );
                            }, 0);
                          } else {
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
              </div>
              <div className="rightPane rightPane--enter">
                <div className="treeScroll">
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
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
}

export default App;
