import React from 'react';
import './PopupWindow.css';

interface PopupWindowProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
}


const PopupWindow: React.FC<PopupWindowProps & { title?: string; description?: string }> = ({ isOpen, onClose, title, description, children }) => {
  React.useEffect(() => {
    if (!isOpen) return;
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        onClose();
      }
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);
  if (!isOpen) return null;
  return (
    <div className="popup-overlay">
      <div className="popup-window">
        <button className="popup-close" onClick={onClose} aria-label="Close">×</button>
        <div className="popup-content">
          <h2>{title}</h2>
          <p>{description}</p>
          {children}
        </div>
      </div>
    </div>
  );
};

export default PopupWindow;
