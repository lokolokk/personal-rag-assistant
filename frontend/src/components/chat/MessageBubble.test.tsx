import { render, screen, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { MessageBubble } from './MessageBubble';
import { DisplayMessage, CitationDetail } from '../../types';

describe('MessageBubble component', () => {
  const sampleCitations: CitationDetail[] = [
    {
      citationIndex: 1,
      chunkId: 'doc_att_p4_c1',
      documentId: 'doc_att',
      filename: 'attention.pdf',
      pageNumber: 4,
      sectionTitle: '3.2 Multi-Head Attention',
      sourceText: 'Multi-head attention allows the model...',
    },
  ];

  it('renders user message as flat engineering log entry', () => {
    const userMsg: DisplayMessage = {
      id: 'msg_1',
      role: 'user',
      content: 'Explain scaled dot-product attention',
      citations: [],
      createdAt: '2026-10-02T10:00:00Z',
    };

    render(<MessageBubble message={userMsg} onSelectCitation={vi.fn()} />);

    expect(screen.getByText('Explain scaled dot-product attention')).toBeInTheDocument();
    expect(screen.getByText('You')).toBeInTheDocument();
    expect(screen.getByTestId('user-message-log')).toBeInTheDocument();
  });

  it('renders assistant message in flat engineering log layout', () => {
    const assistantMsg: DisplayMessage = {
      id: 'msg_2',
      role: 'assistant',
      content: 'Scaled dot-product attention computes attention weights.',
      citations: [],
      createdAt: '2026-10-02T10:00:01Z',
    };

    render(<MessageBubble message={assistantMsg} onSelectCitation={vi.fn()} />);

    expect(
      screen.getByText('Scaled dot-product attention computes attention weights.')
    ).toBeInTheDocument();
    expect(screen.getByText('Assistant')).toBeInTheDocument();
    expect(screen.getByTestId('assistant-message-log')).toBeInTheDocument();
  });

  it('renders animated status indicator pill during retrieval lifecycle before tokens arrive', () => {
    const searchingMsg: DisplayMessage = {
      id: 'msg_3',
      role: 'assistant',
      content: '',
      citations: [],
      createdAt: '2026-10-02T10:00:02Z',
      status: 'searching',
      isStreaming: true,
    };

    render(<MessageBubble message={searchingMsg} onSelectCitation={vi.fn()} />);

    expect(screen.getByText(/Searching documents/i)).toBeInTheDocument();
  });

  it('renders markdown headings, lists, and formatted text', () => {
    const markdownMsg: DisplayMessage = {
      id: 'msg_4',
      role: 'assistant',
      content:
        '### Key Components\n\n' +
        '- Query vector\n' +
        '- Key vector\n' +
        '- Value vector\n\n' +
        'This is **critical** for performance.',
      citations: [],
      createdAt: '2026-10-02T10:00:03Z',
    };

    render(<MessageBubble message={markdownMsg} onSelectCitation={vi.fn()} />);

    expect(screen.getByText('Key Components')).toBeInTheDocument();
    expect(screen.getByText('Query vector')).toBeInTheDocument();
    expect(screen.getByText('Key vector')).toBeInTheDocument();
    expect(screen.getByText('Value vector')).toBeInTheDocument();
    expect(screen.getByText('critical')).toBeInTheDocument();
  });

  it('renders GFM tables with headers and cell rows', () => {
    const tableMsg: DisplayMessage = {
      id: 'msg_5',
      role: 'assistant',
      content:
        '| Model | Parameters | Latency |\n' +
        '| --- | --- | --- |\n' +
        '| Transformer | 65M | 24ms |\n' +
        '| Reformer | 60M | 18ms |',
      citations: [],
      createdAt: '2026-10-02T10:00:04Z',
    };

    render(<MessageBubble message={tableMsg} onSelectCitation={vi.fn()} />);

    expect(screen.getByText('Model')).toBeInTheDocument();
    expect(screen.getByText('Parameters')).toBeInTheDocument();
    expect(screen.getByText('Transformer')).toBeInTheDocument();
    expect(screen.getByText('65M')).toBeInTheDocument();
    expect(screen.getByText('Reformer')).toBeInTheDocument();
    expect(screen.getByTestId('markdown-table-container')).toBeInTheDocument();
  });

  it('renders code block with language banner and functional copy button', async () => {
    const codeMsg: DisplayMessage = {
      id: 'msg_6',
      role: 'assistant',
      content:
        '```python\n' +
        'def scaled_dot_product(q, k, v):\n' +
        '    return softmax(q @ k.T / math.sqrt(d_k)) @ v\n' +
        '```',
      citations: [],
      createdAt: '2026-10-02T10:00:05Z',
    };

    const mockClipboard = {
      writeText: vi.fn().mockResolvedValue(undefined),
    };
    Object.assign(navigator, { clipboard: mockClipboard });

    render(<MessageBubble message={codeMsg} onSelectCitation={vi.fn()} />);

    expect(screen.getByText('python')).toBeInTheDocument();
    expect(
      screen.getByText(/def scaled_dot_product\(q, k, v\):/)
    ).toBeInTheDocument();

    const copyBtn = screen.getByRole('button', { name: /copy code/i });
    expect(copyBtn).toBeInTheDocument();
    await act(async () => {
      fireEvent.click(copyBtn);
    });

    expect(mockClipboard.writeText).toHaveBeenCalledWith(
      expect.stringContaining('def scaled_dot_product(q, k, v):')
    );
  });

  it('parses bracketed citations [1] and replaces them with interactive CitationBadge', () => {
    const citationMsg: DisplayMessage = {
      id: 'msg_7',
      role: 'assistant',
      content: 'The attention mechanism is described in [1].',
      citations: sampleCitations,
      createdAt: '2026-10-02T10:00:06Z',
    };

    const handleSelectCitation = vi.fn();
    render(
      <MessageBubble
        message={citationMsg}
        onSelectCitation={handleSelectCitation}
      />
    );

    const badge = screen.getByRole('button', { name: /citation 1/i });
    expect(badge).toBeInTheDocument();

    fireEvent.click(badge);
    expect(handleSelectCitation).toHaveBeenCalledWith(sampleCitations[0]);
  });

  it('renders streaming blinking cursor when message is currently streaming', () => {
    const streamingMsg: DisplayMessage = {
      id: 'msg_8',
      role: 'assistant',
      content: 'Thinking about the',
      citations: [],
      createdAt: '2026-10-02T10:00:07Z',
      isStreaming: true,
    };

    render(<MessageBubble message={streamingMsg} onSelectCitation={vi.fn()} />);

    const cursor = screen.getByTestId('streaming-cursor');
    expect(cursor).toBeInTheDocument();
  });
});
