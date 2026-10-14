import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ChatContainer } from './ChatContainer';
import { DisplayMessage } from '../../types';

describe('ChatContainer component', () => {
  const dummyMessages: DisplayMessage[] = [
    {
      id: 'msg_1',
      role: 'user',
      content: 'What is hybrid retrieval?',
      citations: [],
      createdAt: '2026-10-02T10:00:00Z',
    },
    {
      id: 'msg_2',
      role: 'assistant',
      content: 'Hybrid retrieval combines dense vector embeddings with sparse BM25 scoring.',
      citations: [],
      createdAt: '2026-10-02T10:00:01Z',
    },
  ];

  it('renders welcome empty state with clickable prompt suggestion chips when messages is empty', () => {
    const handleSend = vi.fn();
    render(
      <ChatContainer
        messages={[]}
        isStreaming={false}
        selectedDocCount={2}
        onSendMessage={handleSend}
        onStopStream={vi.fn()}
        onSelectCitation={vi.fn()}
      />
    );

    expect(screen.getByText(/Conversational Document RAG/i)).toBeInTheDocument();
    const promptChip = screen.getByRole('button', {
      name: /Summarize the key methodology/i,
    });
    expect(promptChip).toBeInTheDocument();

    fireEvent.click(promptChip);
    expect(handleSend).toHaveBeenCalledWith(
      expect.stringContaining('Summarize the key methodology')
    );
  });

  it('renders conversational messages when history is provided', () => {
    render(
      <ChatContainer
        messages={dummyMessages}
        isStreaming={false}
        selectedDocCount={1}
        onSendMessage={vi.fn()}
        onStopStream={vi.fn()}
        onSelectCitation={vi.fn()}
      />
    );

    expect(screen.getByText('What is hybrid retrieval?')).toBeInTheDocument();
    expect(
      screen.getByText(
        'Hybrid retrieval combines dense vector embeddings with sparse BM25 scoring.'
      )
    ).toBeInTheDocument();
  });

  it('submits query on Enter key press and clears textarea', () => {
    const handleSend = vi.fn();
    render(
      <ChatContainer
        messages={[]}
        isStreaming={false}
        selectedDocCount={1}
        onSendMessage={handleSend}
        onStopStream={vi.fn()}
        onSelectCitation={vi.fn()}
      />
    );

    const textarea = screen.getByPlaceholderText(/Ask questions grounded in/i);
    fireEvent.change(textarea, { target: { value: 'How does self-attention scale?' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(handleSend).toHaveBeenCalledWith('How does self-attention scale?');
    expect(textarea).toHaveValue('');
  });

  it('inserts newline on Shift+Enter without submitting query', () => {
    const handleSend = vi.fn();
    render(
      <ChatContainer
        messages={[]}
        isStreaming={false}
        selectedDocCount={1}
        onSendMessage={handleSend}
        onStopStream={vi.fn()}
        onSelectCitation={vi.fn()}
      />
    );

    const textarea = screen.getByPlaceholderText(/Ask questions grounded in/i);
    fireEvent.change(textarea, { target: { value: 'Line 1' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });

    expect(handleSend).not.toHaveBeenCalled();
  });

  it('displays active document scope indicator pill in input capsule', () => {
    render(
      <ChatContainer
        messages={[]}
        isStreaming={false}
        selectedDocCount={3}
        onSendMessage={vi.fn()}
        onStopStream={vi.fn()}
        onSelectCitation={vi.fn()}
      />
    );

    expect(screen.getByText(/3 docs scoped/i)).toBeInTheDocument();
  });

  it('shows Stop stream button during active streaming and triggers onStopStream', () => {
    const handleStop = vi.fn();
    render(
      <ChatContainer
        messages={dummyMessages}
        isStreaming={true}
        selectedDocCount={0}
        onSendMessage={vi.fn()}
        onStopStream={handleStop}
        onSelectCitation={vi.fn()}
      />
    );

    const stopButton = screen.getByRole('button', { name: /stop generating/i });
    expect(stopButton).toBeInTheDocument();

    fireEvent.click(stopButton);
    expect(handleStop).toHaveBeenCalledTimes(1);
  });

  it('submits query on Ctrl+Enter key press and clears textarea', () => {
    const handleSend = vi.fn();
    render(
      <ChatContainer
        messages={[]}
        isStreaming={false}
        selectedDocCount={1}
        onSendMessage={handleSend}
        onStopStream={vi.fn()}
        onSelectCitation={vi.fn()}
      />
    );

    const textarea = screen.getByPlaceholderText(/Ask questions grounded in/i);
    fireEvent.change(textarea, { target: { value: 'Explain reciprocal rank fusion.' } });
    fireEvent.keyDown(textarea, { key: 'Enter', ctrlKey: true });

    expect(handleSend).toHaveBeenCalledWith('Explain reciprocal rank fusion.');
    expect(textarea).toHaveValue('');
  });

  it('renders docked command console with edge-to-edge container and Ctrl+Enter shortcut cue', () => {
    render(
      <ChatContainer
        messages={[]}
        isStreaming={false}
        selectedDocCount={2}
        onSendMessage={vi.fn()}
        onStopStream={vi.fn()}
        onSelectCitation={vi.fn()}
      />
    );

    const consoleElement = screen.getByTestId('docked-command-console');
    expect(consoleElement).toBeInTheDocument();
    expect(consoleElement).toHaveClass('border-t', 'border-zinc-800', 'bg-[#0e0e11]');
    expect(screen.getByText(/Ctrl\+Enter/i)).toBeInTheDocument();
  });

  it('renders distilled empty state with headline, subtitle, and prompt suggestions', () => {
    render(
      <ChatContainer
        messages={[]}
        isStreaming={false}
        selectedDocCount={2}
        onSendMessage={vi.fn()}
        onStopStream={vi.fn()}
        onSelectCitation={vi.fn()}
      />
    );

    expect(screen.getByTestId('corpus-telemetry-grid')).toBeInTheDocument();
    expect(screen.getByText(/Conversational Document RAG/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Ask questions, compare findings, and explore your documents\./i)
    ).toBeInTheDocument();
    expect(screen.getByText(/Suggested Questions/i)).toBeInTheDocument();
  });

  it('triggers onOpenSpecs when quiet inline specs link is clicked in empty state', () => {
    const handleOpenSpecs = vi.fn();
    render(
      <ChatContainer
        messages={[]}
        isStreaming={false}
        selectedDocCount={1}
        onSendMessage={vi.fn()}
        onStopStream={vi.fn()}
        onSelectCitation={vi.fn()}
        onOpenSpecs={handleOpenSpecs}
      />
    );

    const specsBtn = screen.getByRole('button', {
      name: /view system architecture & telemetry specs/i,
    });
    expect(specsBtn).toBeInTheDocument();

    fireEvent.click(specsBtn);
    expect(handleOpenSpecs).toHaveBeenCalledTimes(1);
  });
});

