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

// Layout function with subtree sizing: centers parents above children and distributes space evenly
function getTreeLayout(root: TreeNode): DrawnNode[] {
  const VISIBLE = (n: TreeNode) => !(n.hidden);
  const childrenOf = (n: TreeNode) => (n.children || []).filter(VISIBLE);

  function computeSubtreeWidth(node: TreeNode): number {
    const visibleChildren = childrenOf(node);
    if (visibleChildren.length === 0) return NODE_WIDTH;
    const childWidths = visibleChildren.map(computeSubtreeWidth);
    const gaps = HORIZONTAL_GAP * (childWidths.length - 1);
    return childWidths.reduce((a, b) => a + b, 0) + gaps;
  }

  const nodes: DrawnNode[] = [];

  function layout(node: TreeNode, depth: number, leftX: number) {
    const visibleChildren = childrenOf(node);
    const subtreeWidth = computeSubtreeWidth(node);
    const nodeX = leftX + subtreeWidth / 2 - NODE_WIDTH / 2;
    nodes.push({ node, x: nodeX, y: depth * VERTICAL_GAP, width: NODE_WIDTH, height: NODE_HEIGHT });
    if (visibleChildren.length === 0) return;
    let currentLeft = leftX;
    for (const child of visibleChildren) {
      const childWidth = computeSubtreeWidth(child);
      layout(child, depth + 1, currentLeft);
      currentLeft += childWidth + HORIZONTAL_GAP;
    }
  }

  layout(root, 0, 0);
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

const NODE_WIDTH = 160;
const NODE_HEIGHT = 56;
const HORIZONTAL_GAP = 110;
const VERTICAL_GAP = 160;

const Tree: React.FC<TreeProps> = ({ data, onNodeClick }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredNode, setHoveredNode] = useState<DrawnNode | null>(null);
  const [mouse, setMouse] = useState<{x: number, y: number} | null>(null);
  const [containerSize, setContainerSize] = useState<{width: number, height: number}>({ width: 0, height: 0 });

  // Observe container size to make canvas responsive
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const cr = entry.contentRect;
        setContainerSize({ width: cr.width, height: cr.height });
      }
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const nodes = getTreeLayout(data);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Compute content bounds
    const contentWidth = nodes.length > 0 ? Math.max(...nodes.map(n => n.x + n.width)) + 40 : 400;
    const contentHeight = nodes.length > 0 ? Math.max(...nodes.map(n => n.y + NODE_HEIGHT)) + 40 : 300;

    const targetWidth = Math.max(1, containerSize.width || contentWidth);
    const targetHeight = Math.max(1, containerSize.height || contentHeight);

    // Fit content inside container (letterbox) while preserving aspect ratio
    const scale = Math.min(targetWidth / contentWidth, targetHeight / contentHeight);
    const offsetX = (targetWidth - contentWidth * scale) / 2;
    const offsetY = (targetHeight - contentHeight * scale) / 2;

    // Set backing store size and CSS size for Hi-DPI
    canvas.width = Math.floor(targetWidth * dpr);
    canvas.height = Math.floor(targetHeight * dpr);
    canvas.style.width = `${targetWidth}px`;
    canvas.style.height = `${targetHeight}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, targetWidth, targetHeight);
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);

    // Pull theme variables for consistent styling
    const styles = getComputedStyle(document.documentElement);
    const surface2 = (styles.getPropertyValue('--surface-2') || 'rgba(255,255,255,0.08)').trim();
    const textPrimary = (styles.getPropertyValue('--text-primary') || '#ffffff').trim();

    // Draw connectors
    for (const node of nodes) {
      if (node.node.children && node.node.children.length > 0) {
        for (const child of node.node.children) {
          const childNode = nodes.find(n => n.node.id === child.id);
          if (childNode) {
            // Bezier connector
            const startX = node.x + node.width / 2;
            const startY = node.y + NODE_HEIGHT;
            const endX = childNode.x + childNode.width / 2;
            const endY = childNode.y;
            const cpY = (startY + endY) / 2; // vertical midpoint for curve
            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.bezierCurveTo(startX, cpY, endX, cpY, endX, endY);
            
            // Create gradient for connector
            const gradient = ctx.createLinearGradient(startX, startY, endX, endY);
            gradient.addColorStop(0, 'rgba(102, 126, 234, 0.85)');
            gradient.addColorStop(1, 'rgba(118, 75, 162, 0.85)');
            
            ctx.strokeStyle = gradient;
            ctx.lineWidth = 5;
            ctx.shadowColor = 'rgba(102, 126, 234, 0.35)';
            ctx.shadowBlur = 6;
            ctx.stroke();
            ctx.shadowBlur = 0;
          }
        }
      }
    }

    // Draw nodes
    for (const node of nodes) {
      const isHovered = hoveredNode && hoveredNode.node.id === node.node.id;
      
      // Base node background using chat response surface color
      ctx.beginPath();
      ctx.roundRect(node.x, node.y, node.width, NODE_HEIGHT, 16);
      ctx.fillStyle = surface2;
      ctx.fill();

      // Subtle top highlight for depth
      const highlight = ctx.createLinearGradient(node.x, node.y, node.x, node.y + NODE_HEIGHT);
      highlight.addColorStop(0, 'rgba(255, 255, 255, 0.06)');
      highlight.addColorStop(0.6, 'rgba(255, 255, 255, 0.02)');
      highlight.addColorStop(1, 'rgba(255, 255, 255, 0.00)');
      ctx.fillStyle = highlight;
      ctx.fill();
      
      // Draw node border
      ctx.strokeStyle = isHovered ? 'rgba(102, 126, 234, 0.9)' : 'rgba(102, 126, 234, 0.55)';
      ctx.lineWidth = isHovered ? 2.25 : 1.5;
      ctx.stroke();
      
      // Add shadow effect
      if (isHovered) {
        ctx.shadowColor = 'rgba(102, 126, 234, 0.45)';
        ctx.shadowBlur = 12;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 2;
      } else {
        ctx.shadowColor = 'rgba(0, 0, 0, 0.35)';
        ctx.shadowBlur = 6;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 2;
      }
      
      // Draw text
      ctx.fillStyle = textPrimary || '#fff';
      ctx.font = '700 17px Inter, -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.node.label, node.x + node.width / 2, node.y + NODE_HEIGHT / 2);
      
      // Reset shadow
      ctx.shadowBlur = 0;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
    }
  }, [data, hoveredNode, containerSize]);

  // Mouse hover logic
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const nodes = getTreeLayout(data);
    // Content bounds and transform for hit-testing
    const contentWidth = nodes.length > 0 ? Math.max(...nodes.map(n => n.x + n.width)) + 40 : 400;
    const contentHeight = nodes.length > 0 ? Math.max(...nodes.map(n => n.y + NODE_HEIGHT)) + 40 : 300;
    const targetWidth = Math.max(1, containerSize.width || contentWidth);
    const targetHeight = Math.max(1, containerSize.height || contentHeight);
    const scale = Math.min(targetWidth / contentWidth, targetHeight / contentHeight);
    const offsetX = (targetWidth - contentWidth * scale) / 2;
    const offsetY = (targetHeight - contentHeight * scale) / 2;
    function handleMove(e: MouseEvent) {
      if (!canvas){
        throw new Error("Canvas element is not available");
      }
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      setMouse({x: mx, y: my});
      // Convert to content space for hit testing
      const cx = (mx - offsetX) / scale;
      const cy = (my - offsetY) / scale;
      let found = null;
      for (const node of nodes) {
        if (
          cx >= node.x &&
          cx <= node.x + node.width &&
          cy >= node.y &&
          cy <= node.y + node.height
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
      const cx = (mx - offsetX) / scale;
      const cy = (my - offsetY) / scale;
      for (const node of nodes) {
        if (
          cx >= node.x &&
          cx <= node.x + node.width &&
          cy >= node.y &&
          cy <= node.y + node.height
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
  }, [data, containerSize]);

  // Calculate canvas size
  const nodes = getTreeLayout(data);
  const contentWidth = nodes.length > 0 ? Math.max(...nodes.map(n => n.x + n.width)) + 40 : 400;
  const contentHeight = nodes.length > 0 ? Math.max(...nodes.map(n => n.y + NODE_HEIGHT)) + 40 : 300;
  const targetWidth = Math.max(1, containerSize.width || contentWidth);
  const targetHeight = Math.max(1, containerSize.height || contentHeight);
  const scale = Math.min(targetWidth / contentWidth, targetHeight / contentHeight);
  const offsetX = (targetWidth - contentWidth * scale) / 2;
  const offsetY = (targetHeight - contentHeight * scale) / 2;

  return (
    <div ref={containerRef} style={{ position: 'relative', width: '100%', height: '100%' }}>
      <canvas ref={canvasRef} style={{ background: 'transparent', display: 'block', width: '100%', height: '100%' }} />
      {hoveredNode && mouse && (
        <div
          style={{
            position: 'absolute',
            left: offsetX + (hoveredNode.x + hoveredNode.width + 15) * scale,
            top: offsetY + (hoveredNode.y - 10) * scale,
            background: 'rgba(255, 255, 255, 0.05)',
            backdropFilter: 'blur(20px)',
            color: '#fff',
            border: '1px solid rgba(102, 126, 234, 0.3)',
            borderRadius: 12,
            padding: '12px 14px',
            pointerEvents: 'none',
            zIndex: 10,
            minWidth: Math.max(180, 180 * scale),
            maxWidth: Math.max(260, 260 * scale),
            fontSize: Math.max(12, 12 * scale),
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.05)',
            lineHeight: 1.4,
          }}
        >
          <div style={{ 
            fontWeight: '600', 
            marginBottom: 6, 
            color: '#667eea',
            fontSize: Math.max(14, 14 * scale)
          }}>
            {hoveredNode.node.label}
          </div>
          <div style={{ 
            color: 'rgba(255, 255, 255, 0.8)',
            fontSize: Math.max(12, 12 * scale)
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
