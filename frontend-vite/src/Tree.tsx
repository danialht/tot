import React, { useRef, useState, useEffect } from 'react';
import './Tree.css';
import MathText from './MathText';

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
const NODE_HEIGHT = 88;
const HORIZONTAL_GAP = 110;
const VERTICAL_GAP = 220;

function wrapLabelText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const words = (text || '').split(/\s+/);
  const lines: string[] = [];
  let current = '';
  for (const word of words) {
    const test = current ? current + ' ' + word : word;
    if (ctx.measureText(test).width <= maxWidth) {
      current = test;
    } else {
      if (current) lines.push(current);
      // If a single word is longer than maxWidth, hard-break it
      if (ctx.measureText(word).width > maxWidth) {
        let slice = '';
        for (const ch of word) {
          const next = slice + ch;
          if (ctx.measureText(next).width <= maxWidth) {
            slice = next;
          } else {
            if (slice) lines.push(slice);
            slice = ch;
          }
        }
        current = slice;
      } else {
        current = word;
      }
    }
  }
  if (current) lines.push(current);
  return lines.length ? lines : [''];
}

const Tree: React.FC<TreeProps> = ({ data, onNodeClick }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredNode, setHoveredNode] = useState<DrawnNode | null>(null);
  const [mouse, setMouse] = useState<{x: number, y: number} | null>(null);
  const [containerSize, setContainerSize] = useState<{width: number, height: number}>({ width: 0, height: 0 });
  const [transform, setTransform] = useState<{ scale: number; offsetX: number; offsetY: number }>({ scale: 1, offsetX: 0, offsetY: 0 });

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

    // Prepare measurement
    ctx.font = '700 18px Inter, -apple-system, BlinkMacSystemFont, sans-serif';
    const lineHeight = 22;
    const paddingV = 12;
    const textMaxWidth = NODE_WIDTH - 20;
    const renderInfo = nodes.map(n => {
      const displayLabel = n.node.id === data.id ? 'Start' : n.node.label;
      const lines = wrapLabelText(ctx, displayLabel, textMaxWidth);
      const dynamicHeight = Math.max(NODE_HEIGHT, lines.length * lineHeight + paddingV * 2);
      return { n, lines, dynamicHeight };
    });

    // Compute content bounds
    const contentWidth = renderInfo.length > 0 ? Math.max(...renderInfo.map(r => r.n.x + r.n.width)) + 40 : 400;
    const contentHeight = renderInfo.length > 0 ? Math.max(...renderInfo.map(r => r.n.y + r.dynamicHeight)) + 40 : 300;

    // Fit horizontally, allow vertical scroll
    const containerW = Math.max(1, containerSize.width || contentWidth);
    const containerH = Math.max(1, containerSize.height || contentHeight);
    const scale = Math.min(containerW / contentWidth, containerH / contentHeight);
    const targetWidth = Math.max(1, containerW);
    const targetHeight = Math.max(1, containerH);
    const offsetX = (targetWidth - contentWidth * scale) / 2;
    const offsetY = (targetHeight - contentHeight * scale) / 2;

    // Persist transform so hover/click hit-testing and tooltip align with drawing
    setTransform({ scale, offsetX, offsetY });

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
    const textPrimary = (styles.getPropertyValue('--text-primary') || '#ffffff').trim();
    const brandStart = (styles.getPropertyValue('--brand-start') || '#667eea').trim();
    const brandEnd = (styles.getPropertyValue('--brand-end') || '#764ba2').trim();

    // Draw connectors
    for (const r of renderInfo) {
      const node = r.n;
      const nodeH = r.dynamicHeight;
      if (node.node.children && node.node.children.length > 0) {
        for (const child of node.node.children) {
          const childInfo = renderInfo.find(x => x.n.node.id === child.id);
          if (childInfo) {
            // Bezier connector
            const startX = node.x + node.width / 2;
            const startY = node.y + nodeH;
            const endX = childInfo.n.x + childInfo.n.width / 2;
            const endY = childInfo.n.y;
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
    for (const r of renderInfo) {
      const node = r.n;
      const nodeH = r.dynamicHeight;
      const isHovered = hoveredNode && hoveredNode.node.id === node.node.id;
      
      // Node background: same gradient as Send button
      ctx.beginPath();
      ctx.roundRect(node.x, node.y, node.width, nodeH, 16);
      const nodeFill = ctx.createLinearGradient(node.x, node.y, node.x + node.width, node.y + nodeH);
      nodeFill.addColorStop(0, brandStart);
      nodeFill.addColorStop(1, brandEnd);
      ctx.fillStyle = nodeFill;
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
      ctx.textBaseline = 'alphabetic';
      const totalTextHeight = r.lines.length * lineHeight;
      const startY = node.y + (nodeH - totalTextHeight) / 2 + lineHeight - 2;
      const centerX = node.x + node.width / 2;
      for (let i = 0; i < r.lines.length; i++) {
        ctx.fillText(r.lines[i], centerX, startY + i * lineHeight);
      }
      
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
    // Prepare measurement context for hit-testing and bounds
    const mctx = canvas.getContext('2d');
    if (!mctx) return;
    mctx.font = '700 18px Inter, -apple-system, BlinkMacSystemFont, sans-serif';
    const lineHeight = 22;
    const paddingV = 12;
    const textMaxWidth = NODE_WIDTH - 20;
    const renderBounds = nodes.map(n => {
      const displayLabel = n.node.id === data.id ? 'Start' : n.node.label;
      const lines = wrapLabelText(mctx, displayLabel, textMaxWidth);
      const dynamicHeight = Math.max(NODE_HEIGHT, lines.length * lineHeight + paddingV * 2);
      return { ...n, height: dynamicHeight } as DrawnNode;
    });
    // Use the same transform as the drawing pass
    const { scale, offsetX, offsetY } = transform;
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
      for (const node of renderBounds) {
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
      for (const node of renderBounds) {
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
  }, [data, containerSize, transform]);

  // Calculate canvas size
  const tooltipTransform = transform;

  return (
    <div ref={containerRef} style={{ position: 'relative', width: '100%', height: '100%' }}>
      <canvas ref={canvasRef} style={{ background: 'transparent', display: 'block', width: '100%', height: '100%' }} />
      {hoveredNode && mouse && (
        <div
          style={{
            position: 'absolute',
            left: tooltipTransform.offsetX + (hoveredNode.x + hoveredNode.width + 15) * tooltipTransform.scale,
            top: tooltipTransform.offsetY + (hoveredNode.y - 10) * tooltipTransform.scale,
            background: 'rgba(0, 0, 0, 0.6)',
            backdropFilter: 'blur(20px)',
            color: '#fff',
            border: '1px solid rgba(102, 126, 234, 0.3)',
            borderRadius: 12,
            padding: '12px 14px',
            pointerEvents: 'none',
            zIndex: 10,
            minWidth: Math.max(180, 180 * tooltipTransform.scale),
            maxWidth: Math.max(260, 260 * tooltipTransform.scale),
            fontSize: Math.max(12, 12 * tooltipTransform.scale),
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.05)',
            lineHeight: 1.4,
          }}
        >
          <div style={{ 
            fontWeight: '600', 
            marginBottom: 6, 
            color: '#667eea',
            fontSize: Math.max(14, 14 * tooltipTransform.scale)
          }}>
            <MathText text={hoveredNode.node.id === data.id ? 'Start' : hoveredNode.node.label} />
          </div>
          <div style={{ 
            color: 'rgba(255, 255, 255, 0.8)',
            fontSize: Math.max(12, 12 * tooltipTransform.scale)
          }}>
            <MathText text={(hoveredNode.node.description?.slice(0, 200) || 'No description') + (hoveredNode.node.description && hoveredNode.node.description.length > 200 ? '...' : '')} />
          </div>
        </div>
      )}
    </div>
  );
};

export default Tree;
