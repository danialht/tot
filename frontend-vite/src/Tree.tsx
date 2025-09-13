import React, { useRef, useState, useEffect } from 'react';
import './Tree.css';

// Type for a node with layout info
interface DrawnNode {
  node: TreeNode;
  x: number;
  y: number;
  width: number;
  height: number;
}

// Layout function: returns array of DrawnNode with x/y based on depth and sibling order
function getTreeLayout(root: TreeNode): DrawnNode[] {
  const nodes: DrawnNode[] = [];
  let x = 0;
  const CHAR_WIDTH = 18; // px per character
  function traverse(node: TreeNode, depth: number): number {
    const labelLength = node.label.length;
    const nodeWidth = Math.max(48, labelLength * CHAR_WIDTH);
    const startX = x;
    let childCount = 0;
    if (node.children && node.children.length > 0) {
      for (const child of node.children) {
        traverse(child, depth + 1);
        childCount++;
      }
      // Center parent above its children
      const firstChild = nodes[nodes.length - childCount];
      const lastChild = nodes[nodes.length - 1];
      const centerX = (firstChild.x + lastChild.x) / 2;
      nodes.push({ node, x: centerX, y: depth * VERTICAL_GAP, width: nodeWidth, height: NODE_HEIGHT });
    } else {
      nodes.push({ node, x, y: depth * VERTICAL_GAP, width: nodeWidth, height: NODE_HEIGHT });
      x += nodeWidth + HORIZONTAL_GAP;
    }
    return startX;
  }
  traverse(root, 0);
  return nodes;
}

// Example tree data structure
export interface TreeNode {
  id: string;
  label: string;
  description?: string;
  children?: TreeNode[];
}

interface TreeProps {
  data: TreeNode;
}

const NODE_HEIGHT = 40;
const HORIZONTAL_GAP = 40;
const VERTICAL_GAP = 80;

const Tree: React.FC<TreeProps> = ({ data }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredNode, setHoveredNode] = useState<DrawnNode | null>(null);
  const [mouse, setMouse] = useState<{x: number, y: number} | null>(null);

  useEffect(() => {
    const nodes = getTreeLayout(data);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw connectors
    for (const node of nodes) {
      if (node.node.children && node.node.children.length > 0) {
        for (const child of node.node.children) {
          const childNode = nodes.find(n => n.node.id === child.id);
          if (childNode) {
            ctx.beginPath();
            ctx.moveTo(node.x + node.width / 2, node.y + NODE_HEIGHT);
            ctx.lineTo(childNode.x + childNode.width / 2, childNode.y);
            ctx.strokeStyle = 'green';
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        }
      }
    }

    // Draw nodes
    for (const node of nodes) {
      ctx.beginPath();
      ctx.rect(node.x, node.y, node.width, NODE_HEIGHT);
      ctx.fillStyle = hoveredNode && hoveredNode.node.id === node.node.id ? '#003300' : '#111';
      ctx.strokeStyle = 'green';
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = 'green';
      ctx.font = 'bold 16px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.node.label, node.x + node.width / 2, node.y + NODE_HEIGHT / 2);
    }
  }, [data, hoveredNode]);

  // Mouse hover logic
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const nodes = getTreeLayout(data);
    function handleMove(e: MouseEvent) {
      if (!canvas){
        throw new Error("Canvas element is not available");
      }
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      setMouse({x: mx, y: my});
      let found = null;
      for (const node of nodes) {
        if (
          mx >= node.x &&
          mx <= node.x + node.width &&
          my >= node.y &&
          my <= node.y + node.height
        ) {
          found = node;
          break;
        }
      }
      setHoveredNode(found);
    }
    function handleLeave() {
      setHoveredNode(null);
      setMouse(null);
    }
    canvas.addEventListener('mousemove', handleMove);
    canvas.addEventListener('mouseleave', handleLeave);
    return () => {
      canvas.removeEventListener('mousemove', handleMove);
      canvas.removeEventListener('mouseleave', handleLeave);
    };
  }, [data]);

  // Calculate canvas size
  const nodes = getTreeLayout(data);
  const width = nodes.length > 0 ? Math.max(...nodes.map(n => n.x + n.width)) + 20 : 400;
  const height = nodes.length > 0 ? Math.max(...nodes.map(n => n.y + NODE_HEIGHT)) + 20 : 300;

  return (
    <div style={{ position: 'relative', width, height }}>
      <canvas ref={canvasRef} width={width} height={height} style={{ background: 'transparent', display: 'block' }} />
      {hoveredNode && mouse && (
        <div
          style={{
            position: 'absolute',
            left: hoveredNode.x + hoveredNode.width + 10,
            top: hoveredNode.y,
            background: '#222',
            color: 'green',
            border: '1px solid green',
            borderRadius: 6,
            padding: '8px 12px',
            pointerEvents: 'none',
            zIndex: 10,
            minWidth: 120,
            fontSize: 14,
            boxShadow: '0 2px 8px #000a',
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: 4 }}>{hoveredNode.node.label}</div>
          <div>{hoveredNode.node.description || 'No description'}</div>
        </div>
      )}
    </div>
  );
};

export default Tree;
