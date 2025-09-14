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
  function traverse(node: TreeNode, depth: number): number {
    const nodeWidth = 50;
    const startX = x;
    let childCount = 0;
    if (node.children && node.children.length > 0) {
      for (const child of node.children) {
        if(child.hidden) continue;
        traverse(child, depth + 1);
        childCount++;
      }
      if(childCount === 0) {
        // Act as there is not child
        nodes.push({ node, x, y: depth * VERTICAL_GAP, width: nodeWidth, height: NODE_HEIGHT });
        x += nodeWidth + HORIZONTAL_GAP;
        return startX;
      }
      // Center parent above its children
      const firstChild = nodes[nodes.length - childCount];
      const lastChild = nodes[nodes.length - 1];
      console.log('First child:', firstChild, 'Last child:', lastChild, 'childcount', childCount);
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
  hidden?: boolean;
}

interface TreeProps {
  data: TreeNode;
  onNodeClick?: (node: TreeNode) => void;
}

const NODE_HEIGHT = 40;
const HORIZONTAL_GAP = 40;
const VERTICAL_GAP = 80;

const Tree: React.FC<TreeProps> = ({ data, onNodeClick }) => {
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
            
            // Create gradient for connector
            const gradient = ctx.createLinearGradient(
              node.x + node.width / 2, node.y + NODE_HEIGHT,
              childNode.x + childNode.width / 2, childNode.y
            );
            gradient.addColorStop(0, 'rgba(102, 126, 234, 0.8)');
            gradient.addColorStop(1, 'rgba(118, 75, 162, 0.8)');
            
            ctx.strokeStyle = gradient;
            ctx.lineWidth = 3;
            ctx.shadowColor = 'rgba(102, 126, 234, 0.3)';
            ctx.shadowBlur = 4;
            ctx.stroke();
            ctx.shadowBlur = 0;
          }
        }
      }
    }

    // Draw nodes
    for (const node of nodes) {
      const isHovered = hoveredNode && hoveredNode.node.id === node.node.id;
      
      // Create gradient for node background
      const nodeGradient = ctx.createLinearGradient(
        node.x, node.y,
        node.x, node.y + NODE_HEIGHT
      );
      
      if (isHovered) {
        nodeGradient.addColorStop(0, 'rgba(102, 126, 234, 0.2)');
        nodeGradient.addColorStop(1, 'rgba(118, 75, 162, 0.2)');
      } else {
        nodeGradient.addColorStop(0, 'rgba(255, 255, 255, 0.05)');
        nodeGradient.addColorStop(1, 'rgba(255, 255, 255, 0.02)');
      }
      
      // Draw node background with rounded corners
      ctx.beginPath();
      ctx.roundRect(node.x, node.y, node.width, NODE_HEIGHT, 12);
      ctx.fillStyle = nodeGradient;
      ctx.fill();
      
      // Draw node border
      ctx.strokeStyle = isHovered ? 'rgba(102, 126, 234, 0.6)' : 'rgba(102, 126, 234, 0.3)';
      ctx.lineWidth = isHovered ? 2 : 1;
      ctx.stroke();
      
      // Add shadow effect
      if (isHovered) {
        ctx.shadowColor = 'rgba(102, 126, 234, 0.3)';
        ctx.shadowBlur = 8;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 2;
      }
      
      // Draw text
      ctx.fillStyle = '#fff';
      ctx.font = '600 14px Inter, -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.node.label, node.x + node.width / 2, node.y + NODE_HEIGHT / 2);
      
      // Reset shadow
      ctx.shadowBlur = 0;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
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
    function handleClick(e: MouseEvent) {
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      for (const node of nodes) {
        if (
          mx >= node.x &&
          mx <= node.x + node.width &&
          my >= node.y &&
          my <= node.y + node.height
        ) {
          if (onNodeClick) {
            onNodeClick(node.node);
          }
          break;
        }
      }
    }
    canvas.addEventListener('mousemove', handleMove);
    canvas.addEventListener('mouseleave', handleLeave);
    canvas.addEventListener('click', handleClick);
    return () => {
      canvas.removeEventListener('mousemove', handleMove);
      canvas.removeEventListener('mouseleave', handleLeave);
      canvas.removeEventListener('click', handleClick);
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
            left: hoveredNode.x + hoveredNode.width + 15,
            top: hoveredNode.y - 10,
            background: 'rgba(255, 255, 255, 0.05)',
            backdropFilter: 'blur(20px)',
            color: '#fff',
            border: '1px solid rgba(102, 126, 234, 0.3)',
            borderRadius: 12,
            padding: '16px 20px',
            pointerEvents: 'none',
            zIndex: 10,
            minWidth: 200,
            maxWidth: 300,
            fontSize: 14,
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.05)',
            lineHeight: 1.4,
          }}
        >
          <div style={{ 
            fontWeight: '600', 
            marginBottom: 8, 
            color: '#667eea',
            fontSize: 16 
          }}>
            {hoveredNode.node.label}
          </div>
          <div style={{ 
            color: 'rgba(255, 255, 255, 0.8)',
            fontSize: 13 
          }}>
            {hoveredNode.node.description?.slice(0, 200) || 'No description'}
            {hoveredNode.node.description && hoveredNode.node.description.length > 200 && '...'}
          </div>
        </div>
      )}
    </div>
  );
};

export default Tree;
