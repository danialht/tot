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
  const [input, setInput] = useState("");
  const [lastUserPrompt, setLastUserPrompt] = useState<string>("");
  const [tree, setTree] = useState<TreeNode>(initialTree);
  const [hasSentFirstMessage, setHasSentFirstMessage] = useState(false);
  const [hasFirstPromptBeenAnswered, setHasFirstPromptBeenAnswered] = useState(false);
  const [isBotTyping, setIsBotTyping] = useState(false);
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [effort, setEffort] = useState<'low' | 'medium' | 'high'>('medium');
  const ws = useRef<WebSocket | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const [popupWindowProps, setPopupWindowProps] =
    useState<PopupWindowProps | null>(null);
  const shownIdsRef = React.useRef<Set<string>>(new Set());
  const [messages, setMessages] = useState<string[]>([]);
  const [wsReady, setWsReady] = useState<boolean>(false);
  const [reasoningSeconds, setReasoningSeconds] = useState<number>(0);
  const reasoningStartRef = useRef<number | null>(null);
  const [finalThoughtSeconds, setFinalThoughtSeconds] = useState<number | null>(null);

  // Chat logs removed; no scroll handling needed

  async function animateTreeIncremental(incoming: TreeNode) {
    const nodes: TreeNode[] = [];
    const queue: TreeNode[] = [incoming];
    while (queue.length) {
      const n = queue.shift()!;
      nodes.push(n);
      if (n.children) queue.push(...n.children);
    }
    const newNodes: TreeNode[] = [];
    for (const n of nodes) {
      if (shownIdsRef.current.has(n.id)) {
        n.hidden = false;
      } else {
        n.hidden = true;
        newNodes.push(n);
      }
    }
    setTree({ ...incoming });
    for (const n of newNodes) {
      n.hidden = false;
      setTree({ ...incoming });
      shownIdsRef.current.add(n.id);
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

  React.useEffect(() => {
    if (isBotTyping) {
      const start = Date.now();
      reasoningStartRef.current = start;
      setFinalThoughtSeconds(null);
      setReasoningSeconds(0);
      const id = setInterval(() => {
        setReasoningSeconds(Math.floor((Date.now() - start) / 1000));
      }, 1000);
      return () => clearInterval(id);
    } else {
      setReasoningSeconds(0);
      reasoningStartRef.current = null;
    }
  }, [isBotTyping]);

  const connect = () => {
    if (ws.current) return;
    const url = `ws://localhost:8000/ws?effort=${encodeURIComponent(effort)}`;
    ws.current = new WebSocket(url);
    setWsReady(false);
    ws.current.onopen = () => {
      setWsReady(true);
      console.log("WS open");
    };
    ws.current.onmessage = (event: MessageEvent) => {
      const obj = JSON.parse(event.data);
      if (!hasFirstPromptBeenAnswered) setHasFirstPromptBeenAnswered(true);
      // Interim updates: obj.output may be empty; final has text
      const tree = makeTreeFromData(obj.tree, 0);
      animateTreeIncremental(tree);
      if (obj.output && String(obj.output).trim()) {
        const secs = reasoningStartRef.current ? Math.max(0, Math.round((Date.now() - reasoningStartRef.current) / 1000)) : reasoningSeconds;
        setIsBotTyping(false);
        setFinalThoughtSeconds(secs);
        setMessages((msgs) => [...msgs, `tot: ${obj.output}`]);
      }
    };
    ws.current.onclose = (ev: CloseEvent) => {
      console.warn("WS closed", { code: ev.code, reason: ev.reason });
      ws.current = null;
      setWsReady(false);
    };
    ws.current.onerror = (ev: Event) => {
      console.error("WS error", ev);
    };
  };

  // Reconnect when effort changes
  React.useEffect(() => {
    if (ws.current) {
      try { ws.current.close(); } catch {}
      ws.current = null;
    }
    connect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effort]);
  
  connect();

  const sendMessage = () => {
    if (!input) return;
    const text = input;
    setMessages((msgs) => [...msgs, `You: ${text}`]);
    setLastUserPrompt(text);
    setInput("");
    shownIdsRef.current = new Set();
    if (!hasSentFirstMessage) setHasSentFirstMessage(true);
    setIsBotTyping(true);
    setFinalThoughtSeconds(null);

    const ensureConnectedAndSend = () => {
      const socket = ws.current;
      if (!socket) {
        console.warn("WS not connected, reconnecting then sending...");
        connect();
        const trySend = () => {
          if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            try { ws.current.send(text); } catch (e) { console.error("WS send error", e); }
          } else {
            setTimeout(trySend, 50);
          }
        };
        trySend();
        return;
      }
      if (socket.readyState === WebSocket.OPEN) {
        try { socket.send(text); } catch (e) { console.error("WS send error", e); }
      } else if (socket.readyState === WebSocket.CONNECTING) {
        console.log("WS connecting, will send on open");
        const onOpen = () => {
          try { ws.current?.send(text); } catch (e) { console.error("WS send error", e); }
          ws.current?.removeEventListener('open', onOpen);
        };
        socket.addEventListener('open', onOpen);
      } else {
        console.warn("WS not open (state=", socket.readyState, "), reconnecting then sending...");
        try { socket.close(); } catch {}
        ws.current = null;
        connect();
        const trySend = () => {
          if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            try { ws.current.send(text); } catch (e) { console.error("WS send error", e); }
          } else {
            setTimeout(trySend, 50);
          }
        };
        trySend();
      }
    };

    ensureConnectedAndSend();
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
              {(() => {
                const lastUserIdx = (() => {
                  let idx = -1;
                  for (let i = messages.length - 1; i >= 0; i--) {
                    if (messages[i].startsWith('You: ')) { idx = i; break; }
                  }
                  return idx;
                })();
                const before = messages.slice(0, lastUserIdx + 1);
                const after = messages.slice(lastUserIdx + 1);
                return (
                  <>
                    {before.map((msg, i) => (
                      <div key={`b-${i}`} className={"message " + ( (i % 2 === 0) ? "user" : "bot")}>{msg}</div>
                    ))}
                    {isBotTyping && (
                      <>
                        <div className="message bot" aria-live="polite" style={{ opacity: 0.85 }}>
                          {`Reasoning on ${effort === 'medium' ? 'med' : effort} for ${reasoningSeconds}s`}
                        </div>
                        <div className="message bot typingIndicator" aria-live="polite">
                          <span className="typingDot" />
                          <span className="typingDot" />
                          <span className="typingDot" />
                        </div>
                      </>
                    )}
                    {!isBotTyping && finalThoughtSeconds !== null && (
                      <div className="message bot" aria-live="polite" style={{ opacity: 0.85 }}>
                        {`Thought for ${finalThoughtSeconds}s`}
                      </div>
                    )}
                    {after.map((msg, j) => (
                      <div key={`a-${j}`} className={"message " + ( ((before.length + j) % 2 === 0) ? "user" : "bot")}>{msg}</div>
                    ))}
                  </>
                );
              })()}
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
                  disabled={(!wsReady) || !input}
                  className="button"
                >
                  Send
                </button>
                <select className="dropdown" value={effort} onChange={(e) => {
                  const val = e.target.value as 'low' | 'medium' | 'high';
                  setEffort(val);
                }}>
                  <option value="low">Swift</option>
                  <option value="medium">Normal</option>
                  <option value="high">Genius</option>
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
                    {(() => {
                      const lastUserIdx = (() => {
                        let idx = -1;
                        for (let i = messages.length - 1; i >= 0; i--) {
                          if (messages[i].startsWith('You: ')) { idx = i; break; }
                        }
                        return idx;
                      })();
                      const before = messages.slice(0, lastUserIdx + 1);
                      const after = messages.slice(lastUserIdx + 1);
                      return (
                        <>
                          {before.map((msg, i) => (
                            <div key={`b2-${i}`} className={"message " + ( (i % 2 === 0) ? "user" : "bot")}>{msg}</div>
                          ))}
                          {isBotTyping && (
                            <>
                              <div className="message bot" aria-live="polite" style={{ opacity: 0.85 }}>
                                {`Reasoning on ${effort === 'medium' ? 'med' : effort} for ${reasoningSeconds}s`}
                              </div>
                              <div className="message bot typingIndicator" aria-live="polite">
                                <span className="typingDot" />
                                <span className="typingDot" />
                                <span className="typingDot" />
                              </div>
                            </>
                          )}
                          {!isBotTyping && finalThoughtSeconds !== null && (
                            <div className="message bot" aria-live="polite" style={{ opacity: 0.85 }}>
                              {`Thought for ${finalThoughtSeconds}s`}
                            </div>
                          )}
                          {after.map((msg, j) => (
                            <div key={`a2-${j}`} className={"message " + ( ((before.length + j) % 2 === 0) ? "user" : "bot")}>{msg}</div>
                          ))}
                        </>
                      );
                    })()}
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
                            // Fallback removed (no chat history)
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
                        disabled={(!wsReady) || !input}
                        className="button"
                      >
                        Send
                      </button>
                      <select className="dropdown" value={effort} onChange={(e) => {
                        const val = e.target.value as 'low' | 'medium' | 'high';
                        setEffort(val);
                      }}>
                        <option value="low">Swift</option>
                        <option value="medium">Normal</option>
                        <option value="high">Genius</option>
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
