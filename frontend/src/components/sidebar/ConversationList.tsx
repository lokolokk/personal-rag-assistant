import React, { useState } from 'react';
import {
  MessageSquare,
  Plus,
  Trash2,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import { ConversationItem } from '../../types';

export interface ConversationListProps {
  conversations: ConversationItem[];
  activeConversationId: string | null;
  isLoading?: boolean;
  onSelectConversation: (conversation: ConversationItem) => void;
  onDeleteConversation: (id: string) => void;
  onNewChat: () => void;
  hideNewChatButton?: boolean;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
  className?: string;
}

function formatRelativeTime(dateString: string): string {
  try {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (60 * 1000));
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'now';
    if (diffMins < 60) return `${diffMins}m`;
    if (diffHours < 24) return `${diffHours}h`;
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays}d`;

    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch {
    return 'Recent';
  }
}

export const ConversationList: React.FC<ConversationListProps> = ({
  conversations,
  activeConversationId,
  isLoading = false,
  onSelectConversation,
  onDeleteConversation,
  onNewChat,
  hideNewChatButton = false,
  isCollapsed,
  onToggleCollapse,
  className = '',
}) => {
  const [internalExpanded, setInternalExpanded] = useState<boolean>(true);
  const isExpanded = isCollapsed !== undefined ? !isCollapsed : internalExpanded;

  const toggleExpanded = (): void => {
    if (onToggleCollapse) {
      onToggleCollapse();
    } else {
      setInternalExpanded((prev) => !prev);
    }
  };

  return (
    <div
      className={`flex flex-col bg-[#141417] select-none text-zinc-200 ${
        isExpanded ? 'flex-1 min-h-0' : 'shrink-0'
      } ${className}`}
    >
      {/* Sleek Integrated Section Header (32px) */}
      <div className="h-8 px-2.5 border-b border-zinc-800/60 flex items-center justify-between bg-[#121215] shrink-0">
        <button
          type="button"
          onClick={toggleExpanded}
          aria-label="Toggle Session History"
          aria-expanded={isExpanded}
          className="flex items-center space-x-1.5 text-left focus:outline-hidden cursor-pointer group min-w-0"
        >
          {isExpanded ? (
            <ChevronDown className="w-3.5 h-3.5 text-zinc-400 group-hover:text-zinc-200 transition-colors shrink-0" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 text-zinc-400 group-hover:text-zinc-200 transition-colors shrink-0" />
          )}
          <span className="text-[10px] font-mono font-semibold uppercase tracking-wider text-zinc-400 group-hover:text-zinc-200 transition-colors">
            Session History
          </span>
          <span className="text-[10px] font-mono text-zinc-400">
            ({conversations.length})
          </span>
        </button>

        {/* When standalone and hideNewChatButton is false, render new chat trigger */}
        {!hideNewChatButton && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onNewChat();
            }}
            className="p-1 rounded text-zinc-400 hover:text-indigo-400 hover:bg-zinc-800/60 transition cursor-pointer"
            title="New conversation"
            aria-label="New Chat"
          >
            <Plus className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {/* Collapsible Body */}
      {isExpanded && (
        <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
          {/* Main List Area: Flat Typographic Rows */}
          <div className="flex-1 overflow-y-auto px-1.5 py-1 space-y-0.5">
            {isLoading ? (
              <div data-testid="conversation-list-loading" className="space-y-1.5 p-1.5">
                {[1, 2, 3].map((n) => (
                  <div
                    key={n}
                    className="h-7 rounded-md bg-zinc-800/40 animate-pulse flex items-center px-2 space-x-2"
                  >
                    <div className="w-3 h-3 bg-zinc-700/60 rounded-xs" />
                    <div className="h-2.5 bg-zinc-700/60 rounded-xs flex-1" />
                    <div className="w-8 h-2 bg-zinc-700/40 rounded-xs" />
                  </div>
                ))}
              </div>
            ) : conversations.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-32 px-4 text-center">
                <MessageSquare className="w-5 h-5 text-zinc-500 mb-1.5 stroke-[1.5]" />
                <p className="text-xs font-medium text-zinc-300">No past conversations</p>
                <p className="text-[11px] text-zinc-500 mt-0.5 leading-relaxed">
                  Start asking questions to build your history.
                </p>
              </div>
            ) : (
              conversations.map((conv) => {
                const isActive = activeConversationId === conv.id;
                return (
                  <div
                    key={conv.id}
                    role="button"
                    tabIndex={0}
                    aria-current={isActive ? 'true' : undefined}
                    aria-label={conv.title}
                    onClick={() => onSelectConversation(conv)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        onSelectConversation(conv);
                      }
                    }}
                    className={`group relative px-2 py-1.5 rounded-md text-xs transition-colors duration-100 cursor-pointer flex items-center justify-between gap-1.5 ${
                      isActive
                        ? 'bg-[#18181c] text-zinc-100 font-medium border-l-2 border-l-indigo-500 pl-2'
                        : 'text-zinc-300 hover:bg-zinc-800/40 hover:text-zinc-100'
                    }`}
                  >
                    {/* Left: Message Icon + Title */}
                    <div className="flex items-center space-x-1.5 min-w-0 flex-1">
                      <MessageSquare
                        className={`w-3.5 h-3.5 shrink-0 ${
                          isActive ? 'text-indigo-400' : 'text-zinc-500 group-hover:text-zinc-400'
                        }`}
                      />
                      <span className="truncate text-xs font-normal">{conv.title}</span>
                    </div>

                    {/* Right: Relative Time + Hover Delete */}
                    <div className="flex items-center space-x-1.5 shrink-0">
                      <span className="text-[10px] font-mono text-zinc-400">
                        {formatRelativeTime(conv.updatedAt)}
                      </span>

                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteConversation(conv.id);
                        }}
                        className="p-0.5 rounded text-zinc-500 hover:text-rose-400 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                        title="Delete conversation"
                        aria-label="Delete conversation"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
};
