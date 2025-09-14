# Tree of Thoughts (ToT) Implementation

A modern web application implementing and improving upon the **Tree of Thoughts** framework for deliberate problem solving with Large Language Models, as described in the research paper by Yao et al.

> **Paper Reference**: [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)  
> **Authors**: Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan  
> **Published**: NeurIPS 2023

## Demo

[Watch the intro video!](https://www.youtube.com/watch?v=DQgDHpB9s6c)

[Watch the demo video](https://youtu.be/m7taCYxkgDk)

## Screenshots

![Image 1](images/Screenshot%202025-09-14%20at%2012.12.32 PM.png)

![Image 2](images/Screenshot%202025-09-14%20at%2012.12.45%20PM.png)

![Image 3](images/Screenshot%202025-09-14%20at%2012.13.05%20PM.png)

![image 4](images/Screenshot%202025-09-14%20at%2012.13.14%20PM.png)

## Overview

This project implements the Tree of Thoughts (ToT) framework, which generalizes over the popular Chain of Thought approach to enable language models to perform deliberate decision making by considering multiple reasoning paths and self-evaluating choices. Our implementation extends the original research with a modern, interactive web interface and enhanced visualization capabilities.

## Tree of Thoughts Implementation

### Core Algorithm
Our implementation follows the ToT framework with these key components:

1. **Thought Generation**: Multiple reasoning paths for each problem state
2. **State Evaluation**: Self-assessment of intermediate solutions
3. **Search Strategy**: Breadth-first or depth-first exploration
4. **Backtracking**: Ability to revisit and revise previous decisions


## Architecture

### Frontend - React + TypeScript
```
frontend-vite/
├── src/
│   ├── App.tsx              # Main application with ToT logic
│   ├── Tree.tsx             # Interactive tree visualization
│   ├── PopupWindow.tsx      # Node detail exploration
│   └── MathText.tsx         # Mathematical expression rendering
```

**Technologies:**
- **React 18** with TypeScript for type-safe development
- **Vite** for fast development and optimized builds
- **Canvas API** for high-performance tree rendering
- **WebSocket** for real-time communication with backend

### Backend - FastAPI + Python
```
backend/
├── main.py                  # FastAPI server with WebSocket endpoints
├── tree.py                  # ToT algorithm implementation
├── prompts.py               # LLM prompt management
└── prompts.json             # Structured prompt templates
```

**Technologies:**
- **FastAPI** for high-performance API endpoints
- **Python 3.8+** for LLM integration and tree processing
- **WebSocket** for real-time bidirectional communication
- **Structured Prompts** for consistent LLM interactions

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.8+
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/tot.git
   cd tot
   ```

2. **Setup Backend (FastAPI + Python)**
   ```bash
   cd backend
   pip install -r requirements.txt
   python main.py
   ```
   Backend will be available at `http://localhost:8000`

3. **Setup Frontend (React + Vite)**
   ```bash
   cd frontend-vite
   npm install
   npm run dev
   ```
   Frontend will be available at `http://localhost:5173`

### Enhanced Features

#### Interactive Tree Visualization
- **Dynamic Layout**: Auto-sizing nodes with text wrapping
- **Real-time Updates**: Live tree modifications as thoughts develop
- **Node Interaction**: Click to explore detailed reasoning paths
- **Responsive Scaling**: Automatic zoom and pan for optimal viewing

#### Modern Web Interface
- **Glass-morphism Design**: Beautiful translucent interfaces
- **Smooth Animations**: Fluid transitions and micro-interactions
- **Responsive Layout**: Optimized for all device sizes
- **Dark Theme**: Professional color scheme with purple-blue gradients

#### Real-time Communication
- **WebSocket Integration**: Instant message delivery and tree updates
- **Live Collaboration**: Multiple users can explore the same tree
- **Message History**: Persistent conversation with scrollable chat
- **Typing Indicators**: Visual feedback for active sessions

## Configuration

### Environment Setup
```bash
# Backend Configuration
BACKEND_PORT=8000
WEBSOCKET_PORT=8000

# Frontend Configuration
VITE_API_URL=ws://localhost:8000/ws
VITE_APP_TITLE=Tree of Thoughts
```

### Customization Options
- **Tree Layout**: Adjust node spacing and sizing in `Tree.tsx`
- **Thinking Modes**: Configure Normal, Swift, and Genius modes
- **Visual Theme**: Modify CSS custom properties for styling
- **LLM Integration**: Customize prompts in `prompts.json`

## Development

### Available Scripts
```bash
# Frontend Development
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build

# Backend Development
python main.py       # Start FastAPI server
python -m pytest     # Run tests (if implemented)
```

### Code Structure
- **TypeScript**: Strict mode enabled for type safety
- **ESLint**: Configured for React/TypeScript best practices
- **Modular Design**: Separated concerns between UI and logic
- **Error Handling**: Comprehensive error boundaries and validation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

**Implementing the future of AI problem-solving with modern web technologies.**
