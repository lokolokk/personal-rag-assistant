import React, { useState, useRef, useEffect } from 'react';
import { PanelLeft, Pencil, Cpu } from 'lucide-react';

export interface ChatHeaderProps {
  title: string;
  isSidebarOpen: boolean;
  onToggleSidebar: () => void;
  onRename: (newTitle: string) => void | Promise<void>;
  onOpenSpecs: () => void;
  className?: string;
}

export function ChatHeader({
  title,
  isSidebarOpen,
  onToggleSidebar,
  onRename,
  onOpenSpecs,
  className = '',
}: ChatHeaderProps): React.JSX.Element {
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [editValue, setEditValue] = useState<string>(title);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    setEditValue(title);
  }, [title]);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const handleSave = (): void => {
    const trimmed = editValue.trim();
    if (trimmed && trimmed !== title) {
      onRename(trimmed);
    } else {
      setEditValue(title);
    }
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>): void => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSave();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setEditValue(title);
      setIsEditing(false);
    }
  };

  const handleBlur = (): void => {
    handleSave();
  };

  return (
    <div
      className={`h-10 border-b border-zinc-800 px-3 flex items-center justify-between bg-[#141417] shrink-0 z-20 ${className}`}
    >
      {/* Left Cluster: Sidebar toggle (when collapsed) + Conversation Title / Rename */}
      <div className="flex items-center space-x-2 min-w-0">
        {!isSidebarOpen && (
          <button
            type="button"
            onClick={onToggleSidebar}
            className="p-1.5 rounded-md text-zinc-400 hover:text-indigo-400 hover:bg-zinc-800/80 transition cursor-pointer shrink-0"
            title="Expand documents tray"
            aria-label="Expand documents tray"
          >
            <PanelLeft className="w-3.5 h-3.5" />
          </button>
        )}

        {isEditing ? (
          <input
            ref={inputRef}
            type="text"
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={handleBlur}
            aria-label="Edit conversation title"
            className="text-xs font-medium text-zinc-100 bg-[#18181c] border border-indigo-500/70 rounded px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-indigo-500/50 max-w-[240px] sm:max-w-md w-full font-sans"
          />
        ) : (
          <button
            type="button"
            onClick={() => setIsEditing(true)}
            className="group flex items-center space-x-1.5 px-1.5 py-1 rounded text-left hover:bg-zinc-800/50 transition cursor-pointer max-w-[240px] sm:max-w-md min-w-0"
            title="Click to rename conversation"
            aria-label="Rename conversation"
          >
            <span className="text-xs font-medium text-zinc-200 truncate font-sans">
              {title || 'New Conversation'}
            </span>
            <Pencil className="w-3 h-3 text-zinc-500 group-hover:text-zinc-300 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
          </button>
        )}
      </div>

      {/* Right Cluster: System Specs Trigger */}
      <div className="flex items-center space-x-1 shrink-0 ml-2">
        <button
          type="button"
          onClick={onOpenSpecs}
          className="flex items-center space-x-1.5 px-2.5 py-1 rounded-md text-xs font-mono text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/80 border border-transparent hover:border-zinc-700/60 transition cursor-pointer"
          aria-label="System Architecture & Telemetry"
          title="System Architecture & Telemetry"
        >
          <Cpu className="w-3.5 h-3.5 text-indigo-400" />
          <span>System Specs</span>
        </button>
      </div>
    </div>
  );
}
