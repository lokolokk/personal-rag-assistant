import React, { useState } from 'react';
import {
  Copy,
  Check,
  Search,
  Brain,
  Filter,
  Code2,
  Terminal,
  Activity,
} from 'lucide-react';
import { DisplayMessage, CitationDetail, StreamStatus } from '../../types';
import { CitationBadge } from './CitationBadge';

export interface MessageBubbleProps {
  message: DisplayMessage;
  onSelectCitation?: (citation: CitationDetail) => void;
  className?: string;
}

export function MessageBubble({
  message,
  onSelectCitation,
  className = '',
}: MessageBubbleProps): React.JSX.Element {
  const [copiedCodeIndex, setCopiedCodeIndex] = useState<number | null>(null);
  const [copiedMessage, setCopiedMessage] = useState<boolean>(false);

  const isUser = message.role === 'user';

  const handleCopyCode = async (code: string, index: number): Promise<void> => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedCodeIndex(index);
      setTimeout(() => setCopiedCodeIndex(null), 2000);
    } catch {
      // Ignore clipboard write failures in test/restricted environments
    }
  };

  const handleCopyMessage = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopiedMessage(true);
      setTimeout(() => setCopiedMessage(false), 2000);
    } catch {
      // Ignore clipboard failures
    }
  };

  // Render animated status stepper during retrieval lifecycle
  const renderStatusPill = (status?: StreamStatus): React.JSX.Element | null => {
    if (!status || status === 'idle' || status === 'complete' || status === 'streaming') {
      return null;
    }

    const configs: Record<
      string,
      { label: string; icon: React.ReactNode; bg: string; text: string; border: string }
    > = {
      reformulating: {
        label: 'Reformulating conversational search query...',
        icon: <Brain className="w-3.5 h-3.5 text-indigo-400 animate-spin" />,
        bg: 'bg-zinc-900/90',
        text: 'text-indigo-300',
        border: 'border-indigo-500/30',
      },
      searching: {
        label: 'Searching documents (Dense + BM25 Hybrid RRF)...',
        icon: <Search className="w-3.5 h-3.5 text-indigo-400 animate-pulse" />,
        bg: 'bg-zinc-900/90',
        text: 'text-indigo-300',
        border: 'border-indigo-500/30',
      },
      grading: {
        label: 'Grading retrieval relevance & synthesizing...',
        icon: <Filter className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />,
        bg: 'bg-zinc-900/90',
        text: 'text-emerald-300',
        border: 'border-emerald-500/30',
      },
    };

    const cfg = configs[status] ?? {
      label: 'Processing retrieval query...',
      icon: <Activity className="w-3.5 h-3.5 text-zinc-400 animate-pulse" />,
      bg: 'bg-zinc-900/90',
      text: 'text-zinc-300',
      border: 'border-zinc-800',
    };

    return (
      <div
        className={`inline-flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-mono border ${cfg.bg} ${cfg.text} ${cfg.border} animate-pulse my-2 shadow-2xs`}
      >
        {cfg.icon}
        <span>{cfg.label}</span>
      </div>
    );
  };

  // Helper to parse citations within inline text spans
  const renderInlineWithCitations = (text: string): React.ReactNode[] => {
    // Matches bracketed citations like [1], [2], or [1: attention.pdf, p.4]
    const citationRegex = /\[(\d+)(?::[^\]]+)?\]/g;
    const elements: React.ReactNode[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null = null;

    while ((match = citationRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        elements.push(
          <span key={`text_${lastIndex}`}>
            {renderFormattedInline(text.slice(lastIndex, match.index))}
          </span>
        );
      }

      const citIndex = parseInt(match[1], 10);
      const foundCitation = message.citations.find(
        (c) => c.citationIndex === citIndex || c.id === citIndex
      );

      if (foundCitation) {
        elements.push(
          <CitationBadge
            key={`cit_${citIndex}_${match.index}`}
            citation={foundCitation}
            onSelect={onSelectCitation}
          />
        );
      } else {
        // Fallback badge if citation detail metadata is not present in citations list
        const fallbackCitation: CitationDetail = {
          citationIndex: citIndex,
          chunkId: `cit_${citIndex}`,
          documentId: '',
          filename: '',
          pageNumber: 1,
          sectionTitle: '',
          sourceText: '',
        };
        elements.push(
          <CitationBadge
            key={`cit_fallback_${citIndex}_${match.index}`}
            citation={fallbackCitation}
            onSelect={onSelectCitation}
          />
        );
      }

      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
      elements.push(
        <span key={`text_end_${lastIndex}`}>
          {renderFormattedInline(text.slice(lastIndex))}
        </span>
      );
    }

    return elements;
  };

  // Helper for bold, italic, inline code formatting
  const renderFormattedInline = (raw: string): React.ReactNode[] => {
    // Regex for bold (**bold**), code (`code`), and italics (*italic*)
    const parts = raw.split(/(\*\*.*?\*\*|`.*?`|\*.*?\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={i} className="font-semibold text-zinc-100">
            {part.slice(2, -2)}
          </strong>
        );
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code
            key={i}
            className="px-1.5 py-0.5 mx-0.5 rounded text-xs font-mono bg-zinc-800 text-indigo-300 border border-zinc-700/60"
          >
            {part.slice(1, -1)}
          </code>
        );
      }
      if (part.startsWith('*') && part.endsWith('*')) {
        return (
          <em key={i} className="italic text-zinc-300">
            {part.slice(1, -1)}
          </em>
        );
      }
      return part;
    });
  };

  // Markdown block-level tokenizer and renderer
  const renderMarkdown = (content: string): React.JSX.Element => {
    const blocks: React.JSX.Element[] = [];
    const lines = content.split('\n');
    let idx = 0;
    let codeBlockCount = 0;

    while (idx < lines.length) {
      const line = lines[idx];

      // Code Block (```lang)
      if (line.trim().startsWith('```')) {
        const lang = line.trim().slice(3).trim() || 'plaintext';
        const codeLines: string[] = [];
        idx++;
        while (idx < lines.length && !lines[idx].trim().startsWith('```')) {
          codeLines.push(lines[idx]);
          idx++;
        }
        idx++; // Skip closing ```
        const fullCode = codeLines.join('\n');
        const currentCodeIdx = codeBlockCount++;

        blocks.push(
          <div
            key={`code_${idx}`}
            className="my-3 rounded-lg overflow-hidden border border-zinc-800 bg-[#0e0e11] text-zinc-100 text-xs font-mono shadow-sm"
          >
            <div className="flex items-center justify-between px-3.5 py-1.5 bg-[#141417] border-b border-zinc-800 text-[11px] text-zinc-400">
              <div className="flex items-center space-x-1.5">
                <Code2 className="w-3.5 h-3.5 text-zinc-400" />
                <span className="font-medium text-zinc-300">{lang}</span>
              </div>
              <button
                type="button"
                onClick={() => void handleCopyCode(fullCode, currentCodeIdx)}
                className="flex items-center space-x-1 hover:text-zinc-200 transition px-2 py-0.5 rounded hover:bg-zinc-800"
                aria-label="Copy code"
              >
                {copiedCodeIndex === currentCodeIdx ? (
                  <>
                    <Check className="w-3 h-3 text-emerald-400" />
                    <span className="text-emerald-400">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3 h-3" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <pre className="p-3.5 overflow-x-auto text-zinc-200 leading-relaxed font-mono">
              <code>{fullCode}</code>
            </pre>
          </div>
        );
        continue;
      }

      // GFM Table with sticky headers and scroll containment
      if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
        const tableLines: string[] = [];
        while (
          idx < lines.length &&
          lines[idx].trim().startsWith('|') &&
          lines[idx].trim().endsWith('|')
        ) {
          tableLines.push(lines[idx]);
          idx++;
        }

        if (tableLines.length >= 2) {
          const headerCells = tableLines[0]
            .split('|')
            .slice(1, -1)
            .map((c) => c.trim());
          const bodyRows = tableLines.slice(2).map((rowLine) =>
            rowLine
              .split('|')
              .slice(1, -1)
              .map((c) => c.trim())
          );

          blocks.push(
            <div
              key={`table_${idx}`}
              data-testid="markdown-table-container"
              className="my-4 overflow-x-auto rounded-lg border border-zinc-800 bg-[#101014] shadow-xs max-w-full [overscroll-behavior:contain]"
            >
              <table className="min-w-full divide-y divide-zinc-800 text-xs text-left">
                <thead className="sticky top-0 z-10 bg-[#0e0e12] text-zinc-200 font-semibold border-b border-zinc-800">
                  <tr>
                    {headerCells.map((h, hIdx) => (
                      <th
                        key={hIdx}
                        className="px-4 py-2.5 border-r last:border-r-0 border-zinc-800 font-mono text-xs uppercase tracking-wider text-zinc-300 bg-[#0e0e12]"
                      >
                        {renderInlineWithCitations(h)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/60 text-zinc-200">
                  {bodyRows.map((row, rIdx) => (
                    <tr
                      key={rIdx}
                      className={`transition-colors hover:bg-zinc-800/40 ${
                        rIdx % 2 === 1 ? 'bg-[#16161b]' : 'bg-[#101014]'
                      }`}
                    >
                      {row.map((cell, cIdx) => (
                        <td
                          key={cIdx}
                          className="px-4 py-2.5 border-r last:border-r-0 border-zinc-800/60 font-mono text-xs text-zinc-300"
                        >
                          {renderInlineWithCitations(cell)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
          continue;
        }
      }

      // Headings
      if (line.startsWith('### ')) {
        blocks.push(
          <h3 key={`h3_${idx}`} className="text-sm font-semibold text-zinc-100 mt-3 mb-1 tracking-tight">
            {renderInlineWithCitations(line.slice(4))}
          </h3>
        );
        idx++;
        continue;
      }
      if (line.startsWith('## ')) {
        blocks.push(
          <h2 key={`h2_${idx}`} className="text-base font-semibold text-zinc-100 mt-4 mb-1.5 tracking-tight">
            {renderInlineWithCitations(line.slice(3))}
          </h2>
        );
        idx++;
        continue;
      }
      if (line.startsWith('# ')) {
        blocks.push(
          <h1 key={`h1_${idx}`} className="text-lg font-bold text-white mt-4 mb-2 tracking-tight">
            {renderInlineWithCitations(line.slice(2))}
          </h1>
        );
        idx++;
        continue;
      }

      // Lists
      if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
        const listItems: string[] = [];
        while (
          idx < lines.length &&
          (lines[idx].trim().startsWith('- ') || lines[idx].trim().startsWith('* '))
        ) {
          listItems.push(lines[idx].trim().slice(2));
          idx++;
        }
        blocks.push(
          <ul key={`ul_${idx}`} className="my-2 space-y-1 list-disc list-inside text-zinc-200 text-sm">
            {listItems.map((item, lIdx) => (
              <li key={lIdx} className="leading-relaxed">
                {renderInlineWithCitations(item)}
              </li>
            ))}
          </ul>
        );
        continue;
      }

      // Blockquotes
      if (line.trim().startsWith('> ')) {
        blocks.push(
          <blockquote
            key={`bq_${idx}`}
            className="border-l-2 border-indigo-500/70 pl-3.5 my-2.5 italic text-zinc-400 text-sm bg-zinc-900/20 py-0.5 rounded-r"
          >
            {renderInlineWithCitations(line.trim().slice(2))}
          </blockquote>
        );
        idx++;
        continue;
      }

      // Standard text line
      if (line.trim()) {
        blocks.push(
          <p key={`p_${idx}`} className="my-1.5 text-sm text-zinc-200 leading-relaxed">
            {renderInlineWithCitations(line)}
          </p>
        );
      }

      idx++;
    }

    return <div className="space-y-1">{blocks}</div>;
  };

  // Render User Message as Flat Engineering Log Entry
  if (isUser) {
    return (
      <div
        data-testid="user-message-log"
        className={`w-full py-3 px-4 mb-4 rounded-md bg-[#121215] border border-zinc-800/80 text-zinc-100 ${className}`}
      >
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center space-x-2">
            <span className="text-[10px] font-mono uppercase tracking-wider text-indigo-400 font-semibold px-1.5 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/20">
              You
            </span>
            <span className="text-[11px] font-mono text-zinc-500">Query Terminal</span>
          </div>
          {message.createdAt && (
            <span className="text-[10px] font-mono text-zinc-500">
              {new Date(message.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            </span>
          )}
        </div>
        <p className="whitespace-pre-wrap text-sm font-sans text-zinc-100 leading-relaxed pl-0.5">
          {message.content}
        </p>
      </div>
    );
  }

  // Render Assistant Message as Flat Engineering Log Output on Canvas
  return (
    <div
      data-testid="assistant-message-log"
      className={`flex flex-col items-start mb-6 w-full pb-6 border-b border-zinc-800/60 last:border-b-0 ${className}`}
    >
      {/* Header Bar */}
      <div className="flex items-center justify-between w-full mb-2">
        <div className="flex items-center space-x-2">
          <span className="text-[10px] font-mono uppercase tracking-wider text-emerald-400 font-semibold px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 flex items-center gap-1.5">
            <Terminal className="w-3 h-3 text-emerald-400" />
            <span>Assistant</span>
          </span>
          <span className="text-[11px] font-mono text-zinc-500">Synthesis Engine</span>
        </div>

        {message.content && !message.isStreaming && (
          <button
            type="button"
            onClick={() => void handleCopyMessage()}
            className="flex items-center space-x-1 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/80 text-xs px-2 py-1 rounded transition"
            title="Copy answer"
            aria-label="Copy entire message"
          >
            {copiedMessage ? (
              <>
                <Check className="w-3 h-3 text-emerald-400" />
                <span className="text-emerald-400 font-medium">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3 h-3" />
                <span>Copy</span>
              </>
            )}
          </button>
        )}
      </div>

      {/* Retrieval Status Indicator Stepper */}
      {renderStatusPill(message.status)}

      {/* Assistant Content Canvas */}
      <div className="w-full text-zinc-200 text-sm leading-relaxed min-h-[1.5rem]">
        {message.content ? (
          renderMarkdown(message.content)
        ) : message.isStreaming ? (
          <div className="text-zinc-500 text-xs italic py-1 animate-pulse font-mono">
            Analyzing documents and generating answer...
          </div>
        ) : null}

        {/* Live streaming cursor */}
        {message.isStreaming && (
          <span
            data-testid="streaming-cursor"
            className="inline-block w-1.5 h-4 ml-1 bg-indigo-400 animate-pulse align-middle rounded-xs"
          />
        )}
      </div>
    </div>
  );
}
