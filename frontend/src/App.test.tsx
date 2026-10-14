import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from './App';
import * as api from './services/api';

describe('App Integration (Document Management & Conversation Sessions)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(api, 'fetchConversations').mockResolvedValue([]);
  });

  it('fetches documents on mount and displays them in the DocumentList', async () => {
    const mockDocs = [
      {
        id: 'doc_1',
        filename: 'rag_paper.pdf',
        pageCount: 10,
        chunkCount: 24,
        uploadedAt: '2026-10-02T12:00:00Z',
        status: 'ready' as const,
      },
    ];

    vi.spyOn(api, 'fetchDocuments').mockResolvedValue(mockDocs);

    render(<App />);

    expect(screen.getByTestId('document-list-skeleton')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('rag_paper.pdf')).toBeInTheDocument();
    });

    expect(screen.getByText(/Scope: All Documents/i)).toBeInTheDocument();
  });

  it('toggles collapsible sidebar when sidebar button is clicked', async () => {
    vi.spyOn(api, 'fetchDocuments').mockResolvedValue([]);
    render(<App />);

    await waitFor(() => {
      expect(screen.queryByTestId('document-list-skeleton')).not.toBeInTheDocument();
    });

    const collapseBtn = screen.getByRole('button', { name: /collapse documents tray/i });
    fireEvent.click(collapseBtn);

    expect(
      screen.getByRole('button', { name: /expand documents tray/i })
    ).toBeInTheDocument();
  });

  it('opens and closes document upload modal via upload button', async () => {
    vi.spyOn(api, 'fetchDocuments').mockResolvedValue([]);
    render(<App />);

    await waitFor(() => {
      expect(screen.queryByTestId('document-list-skeleton')).not.toBeInTheDocument();
    });

    const uploadBtns = screen.getAllByRole('button', { name: /upload pdf/i });
    fireEvent.click(uploadBtns[0]);

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText(/upload technical documents/i)).toBeInTheDocument();

    const closeBtn = screen.getByRole('button', { name: /close modal/i });
    fireEvent.click(closeBtn);

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('auto-selects newly uploaded document and updates scope indicator', async () => {
    vi.spyOn(api, 'fetchDocuments').mockResolvedValue([]);
    const mockUpload = vi.spyOn(api, 'uploadDocument').mockResolvedValue({
      id: 'doc_new',
      filename: 'quantum_computing.pdf',
      pageCount: 6,
      chunkCount: 18,
      uploadedAt: '2026-10-02T14:00:00Z',
      status: 'ready',
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.queryByTestId('document-list-skeleton')).not.toBeInTheDocument();
    });

    const uploadBtns = screen.getAllByRole('button', { name: /upload pdf/i });
    fireEvent.click(uploadBtns[0]);

    const fileInput = screen.getByTestId('file-input');
    const validPdf = new File(['%PDF-1.4 test'], 'quantum_computing.pdf', {
      type: 'application/pdf',
    });

    fireEvent.change(fileInput, { target: { files: [validPdf] } });

    await waitFor(() => {
      expect(mockUpload).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(screen.getByText('quantum_computing.pdf')).toBeInTheDocument();
      expect(screen.getByText(/Scoped: 1 of 1 doc\(s\)/i)).toBeInTheDocument();
    });
  });

  it('sends scoped chat question through ChatContainer and transmits documentIds', async () => {
    vi.spyOn(api, 'fetchDocuments').mockResolvedValue([
      {
        id: 'doc_1',
        filename: 'rag_paper.pdf',
        pageCount: 10,
        chunkCount: 24,
        uploadedAt: '2026-10-02T12:00:00Z',
        status: 'ready' as const,
      },
    ]);

    const streamData =
      'event: token\ndata: {"token": "RAG answer"}\n\n' +
      'event: done\ndata: {"finishReason": "stop"}\n\n';

    const encoder = new TextEncoder();
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(streamData));
        controller.close();
      },
    });

    const mockFetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/api/v1/conversations')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve([]),
          body: mockStream,
        });
      }
      return Promise.resolve({
        ok: true,
        body: mockStream,
      });
    });
    vi.stubGlobal('fetch', mockFetch);

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('rag_paper.pdf')).toBeInTheDocument();
    });

    const textarea = screen.getByPlaceholderText(/Ask questions grounded in/i);
    fireEvent.change(textarea, { target: { value: 'Explain hybrid retrieval' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    await waitFor(() => {
      expect(screen.getByText('Explain hybrid retrieval')).toBeInTheDocument();
    });

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/v1/conversations/'),
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
    );
  });

  it('opens CitationDrawer when citation badge is clicked in chat and closes on backdrop click', async () => {
    vi.spyOn(api, 'fetchDocuments').mockResolvedValue([
      {
        id: 'doc_1',
        filename: 'rag_paper.pdf',
        pageCount: 10,
        chunkCount: 24,
        uploadedAt: '2026-10-02T12:00:00Z',
        status: 'ready' as const,
      },
    ]);

    const streamData =
      'event: token\ndata: {"token": "Dense retrieval captures semantics [1]."}\n\n' +
      'event: citations\ndata: [{"id": 1, "citationIndex": 1, "documentId": "doc_1", "filename": "rag_paper.pdf", "page": 4, "section": "Section 2: Dense Retrieval", "quote": "Dense retrieval utilizes vector embeddings to match semantic intent."}]\n\n' +
      'event: done\ndata: {"finishReason": "stop"}\n\n';

    const encoder = new TextEncoder();
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(streamData));
        controller.close();
      },
    });

    const mockFetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/api/v1/conversations')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve([]),
          body: mockStream,
        });
      }
      return Promise.resolve({
        ok: true,
        body: mockStream,
      });
    });
    vi.stubGlobal('fetch', mockFetch);

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('rag_paper.pdf')).toBeInTheDocument();
    });

    const textarea = screen.getByPlaceholderText(/Ask questions grounded in/i);
    fireEvent.change(textarea, { target: { value: 'How does dense retrieval work?' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    // Wait for the citation badge to render
    const badge = await screen.findByRole('button', { name: /citation 1/i });
    expect(badge).toBeInTheDocument();

    // Click the citation badge
    fireEvent.click(badge);

    // Verify CitationDrawer opens
    const drawer = await screen.findByRole('dialog', { name: /citation inspector/i });
    expect(drawer).toBeInTheDocument();
    expect(
      screen.getByText(/Dense retrieval utilizes vector embeddings/i)
    ).toBeInTheDocument();

    // Click backdrop to dismiss
    const backdrop = screen.getByTestId('citation-drawer-backdrop');
    fireEvent.click(backdrop);

    // Verify drawer is closed
    await waitFor(() => {
      expect(screen.queryByRole('dialog', { name: /citation inspector/i })).not.toBeInTheDocument();
    });
  });

  it('renders both Corpus Explorer and Session History simultaneously in unified vertical layout', async () => {
    vi.spyOn(api, 'fetchDocuments').mockResolvedValue([
      {
        id: 'doc_1',
        filename: 'rag_paper.pdf',
        pageCount: 10,
        chunkCount: 24,
        uploadedAt: '2026-10-02T12:00:00Z',
        status: 'ready' as const,
      },
    ]);
    vi.spyOn(api, 'fetchConversations').mockResolvedValue([
      {
        id: 'conv_1',
        title: 'Attention Analysis',
        createdAt: '2026-10-02T10:00:00Z',
        updatedAt: '2026-10-02T10:05:00Z',
        documentIds: ['doc_1'],
        messageCount: 2,
      },
    ]);

    render(<App />);

    // Both documents and conversations are rendered simultaneously without tab switching
    expect(await screen.findByText('rag_paper.pdf')).toBeInTheDocument();
    expect(await screen.findByText('Attention Analysis')).toBeInTheDocument();
    expect(screen.getByText(/Corpus Explorer/i)).toBeInTheDocument();
    expect(screen.getByText(/Session History/i)).toBeInTheDocument();
  });

  it('loads conversation messages and restores document scope when a past session is selected', async () => {
    vi.spyOn(api, 'fetchDocuments').mockResolvedValue([
      {
        id: 'doc_1',
        filename: 'rag_paper.pdf',
        pageCount: 10,
        chunkCount: 24,
        uploadedAt: '2026-10-02T12:00:00Z',
        status: 'ready' as const,
      },
    ]);
    vi.spyOn(api, 'fetchConversations').mockResolvedValue([
      {
        id: 'conv_1',
        title: 'Transformer Architecture Query',
        createdAt: '2026-10-02T10:00:00Z',
        updatedAt: '2026-10-02T10:05:00Z',
        documentIds: ['doc_1'],
        messageCount: 2,
      },
    ]);

    const mockMessages = [
      {
        role: 'user' as const,
        content: 'How does self-attention work?',
      },
      {
        role: 'assistant' as const,
        content: 'Self-attention calculates dot products of queries and keys.',
      },
    ];
    const fetchMsgsSpy = vi.spyOn(api, 'fetchConversationMessages').mockResolvedValue(mockMessages);

    render(<App />);

    // Click the conversation item directly from the unified sidebar tree
    const convBtn = await screen.findByRole('button', {
      name: /transformer architecture query/i,
    });
    fireEvent.click(convBtn);

    await waitFor(() => {
      expect(fetchMsgsSpy).toHaveBeenCalledWith('conv_1');
    });

    // Check that historical messages are rendered in ChatContainer
    expect(await screen.findByText('How does self-attention work?')).toBeInTheDocument();
    expect(
      screen.getByText('Self-attention calculates dot products of queries and keys.')
    ).toBeInTheDocument();

    // Verify document scope was restored to doc_1 (1 docs scoped)
    expect(screen.getByText(/1 docs scoped/i)).toBeInTheDocument();
  });

  it('resets conversation to welcome state when "+ New Chat" button is clicked', async () => {
    vi.spyOn(api, 'fetchDocuments').mockResolvedValue([]);
    vi.spyOn(api, 'fetchConversations').mockResolvedValue([
      {
        id: 'conv_1',
        title: 'Past Chat Session',
        createdAt: '2026-10-02T10:00:00Z',
        updatedAt: '2026-10-02T10:05:00Z',
        documentIds: [],
        messageCount: 2,
      },
    ]);
    vi.spyOn(api, 'fetchConversationMessages').mockResolvedValue([
      {
        role: 'user',
        content: 'Question from past session',
      },
      {
        role: 'assistant',
        content: 'Answer from past session',
      },
    ]);

    render(<App />);

    // Load past session directly from unified sidebar tree
    const convBtn = await screen.findByRole('button', { name: /past chat session/i });
    fireEvent.click(convBtn);

    expect(await screen.findByText('Question from past session')).toBeInTheDocument();

    // Click "+ New Chat" button in sidebar toolbar
    const newChatBtn = screen.getByRole('button', { name: /new chat/i });
    fireEvent.click(newChatBtn);

    // Previous conversation messages should be cleared and welcome state shown
    expect(screen.queryByText('Question from past session')).not.toBeInTheDocument();
    expect(screen.getByText(/Suggested Questions/i)).toBeInTheDocument();
  });

  it('deletes conversation from history when delete button is clicked', async () => {
    vi.spyOn(api, 'fetchDocuments').mockResolvedValue([]);
    vi.spyOn(api, 'fetchConversations').mockResolvedValue([
      {
        id: 'conv_del_1',
        title: 'To Be Deleted Session',
        createdAt: '2026-10-02T10:00:00Z',
        updatedAt: '2026-10-02T10:05:00Z',
        documentIds: [],
        messageCount: 1,
      },
    ]);
    const deleteSpy = vi.spyOn(api, 'deleteConversation').mockResolvedValue(undefined);

    render(<App />);

    expect(await screen.findByText('To Be Deleted Session')).toBeInTheDocument();

    const deleteBtn = screen.getByRole('button', { name: /delete conversation/i });
    fireEvent.click(deleteBtn);

    await waitFor(() => {
      expect(deleteSpy).toHaveBeenCalledWith('conv_del_1');
    });

    expect(screen.queryByText('To Be Deleted Session')).not.toBeInTheDocument();
  });

  it('restores rich citations when switching to past conversation and displays full attribution in CitationDrawer', async () => {
    vi.spyOn(api, 'fetchDocuments').mockResolvedValue([]);
    vi.spyOn(api, 'fetchConversations').mockResolvedValue([
      {
        id: 'conv_rich_citations',
        title: 'Attention Paper Discussion',
        createdAt: '2026-10-02T10:00:00Z',
        updatedAt: '2026-10-02T10:05:00Z',
        documentIds: ['doc_1'],
        messageCount: 2,
      },
    ]);

    const mockMessages = [
      {
        role: 'user' as const,
        content: 'Tell me about multi-head attention.',
      },
      {
        role: 'assistant' as const,
        content: 'Multi-head attention allows the model to jointly attend to information [1].',
        citations: [
          {
            id: 1,
            citationIndex: 1,
            chunk_id: 'chunk_attn_1',
            document_id: 'doc_1',
            filename: 'attention_is_all_you_need.pdf',
            page_number: 4,
            section_title: '3.2 Multi-Head Attention',
            quote: 'Multi-head attention allows the model to jointly attend to information.',
          },
        ],
      },
    ];
    vi.spyOn(api, 'fetchConversationMessages').mockResolvedValue(mockMessages);

    render(<App />);

    // Select the conversation directly from unified tree
    const convBtn = await screen.findByRole('button', {
      name: /attention paper discussion/i,
    });
    fireEvent.click(convBtn);

    // Verify assistant message is loaded
    expect(
      await screen.findByText(/Multi-head attention allows the model to jointly attend to information/i)
    ).toBeInTheDocument();

    // Click citation pill [1]
    const citationBadge = screen.getByRole('button', { name: /citation 1/i });
    fireEvent.click(citationBadge);

    // Drawer should open and display full attribution metadata
    expect(screen.getByRole('dialog', { name: /citation inspector/i })).toBeInTheDocument();
    expect(screen.getByTestId('evidence-grounding-pane')).toHaveClass('xl:static');
    expect(screen.getByTestId('citation-drawer-backdrop')).toHaveClass('xl:hidden');
    expect(screen.getByText('attention_is_all_you_need.pdf')).toBeInTheDocument();
    expect(screen.getByText('3.2 Multi-Head Attention')).toBeInTheDocument();
    expect(screen.getAllByText(/page 4/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/untitled document/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/no excerpt text provided/i)).not.toBeInTheDocument();
  });

  it('supports 3-pane workbench topology with docked command console and persistent evidence pane', async () => {
    vi.spyOn(api, 'fetchDocuments').mockResolvedValue([]);
    vi.spyOn(api, 'fetchConversations').mockResolvedValue([]);

    render(<App />);

    await waitFor(() => {
      expect(screen.queryByTestId('document-list-skeleton')).not.toBeInTheDocument();
    });

    // Verify docked command console is rendered in synthesis canvas
    const consoleEl = screen.getByTestId('docked-command-console');
    expect(consoleEl).toBeInTheDocument();
    expect(consoleEl).toHaveClass('border-t', 'border-zinc-800', 'bg-[#0e0e11]');
    expect(screen.getByText(/Ctrl\+Enter/i)).toBeInTheDocument();
  });

  it('renders compact 40px workstation toolbar with RAG Nexus brand and quick actions', async () => {
    vi.spyOn(api, 'fetchDocuments').mockResolvedValue([]);
    vi.spyOn(api, 'fetchConversations').mockResolvedValue([]);

    render(<App />);

    await waitFor(() => {
      expect(screen.queryByTestId('document-list-skeleton')).not.toBeInTheDocument();
    });

    expect(screen.getByText('RAG Nexus')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /new chat/i })).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: /upload pdf/i }).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByRole('button', { name: /collapse documents tray/i })).toBeInTheDocument();
  });

  it('renames active conversation via ChatHeader and persists update to backend', async () => {
    vi.spyOn(api, 'fetchDocuments').mockResolvedValue([]);
    vi.spyOn(api, 'fetchConversations').mockResolvedValue([
      {
        id: 'conv_1',
        title: 'Original Research Topic',
        createdAt: '2026-10-02T10:00:00Z',
        updatedAt: '2026-10-02T10:05:00Z',
        documentIds: [],
        messageCount: 1,
      },
    ]);
    vi.spyOn(api, 'fetchConversationMessages').mockResolvedValue([
      { role: 'user', content: 'Initial query' },
    ]);
    const renameSpy = vi.spyOn(api, 'renameConversation').mockResolvedValue({
      id: 'conv_1',
      title: 'Renamed Paper Discussion',
      createdAt: '2026-10-02T10:00:00Z',
      updatedAt: '2026-10-02T10:10:00Z',
      documentIds: [],
      messageCount: 1,
    });

    render(<App />);

    // Select the session from the explorer
    const sessionBtn = await screen.findByRole('button', {
      name: /original research topic/i,
    });
    fireEvent.click(sessionBtn);

    // ChatHeader should display session title
    const renameTrigger = await screen.findByRole('button', {
      name: /rename conversation/i,
    });
    expect(renameTrigger).toHaveTextContent('Original Research Topic');

    // Click to edit
    fireEvent.click(renameTrigger);

    const input = screen.getByRole('textbox', { name: /edit conversation title/i });
    fireEvent.change(input, { target: { value: 'Renamed Paper Discussion' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    await waitFor(() => {
      expect(renameSpy).toHaveBeenCalledWith('conv_1', 'Renamed Paper Discussion');
    });

    // Confirms title is updated in both ChatHeader and Session History list
    expect(screen.getAllByText('Renamed Paper Discussion').length).toBeGreaterThanOrEqual(1);
  });

  it('opens and closes System Specs modal from header trigger and distilled empty state link', async () => {
    vi.spyOn(api, 'fetchDocuments').mockResolvedValue([]);
    vi.spyOn(api, 'fetchConversations').mockResolvedValue([]);

    render(<App />);

    await waitFor(() => {
      expect(screen.queryByTestId('document-list-skeleton')).not.toBeInTheDocument();
    });

    // 1. Open via top-right ChatHeader button
    const specsHeaderBtn = screen.getByRole('button', {
      name: /^system architecture & telemetry$/i,
    });
    fireEvent.click(specsHeaderBtn);

    const dialog = screen.getByRole('dialog', { name: /system architecture & telemetry/i });
    expect(dialog).toBeInTheDocument();
    expect(screen.getByText(/Hybrid Dense \+ Sparse RRF/i)).toBeInTheDocument();

    // Close via close button in modal
    const closeBtn = screen.getByRole('button', { name: /close system specs/i });
    fireEvent.click(closeBtn);
    expect(
      screen.queryByRole('dialog', { name: /system architecture & telemetry/i })
    ).not.toBeInTheDocument();

    // 2. Open via quiet inline trigger in distilled empty state
    const inlineSpecsBtn = screen.getByRole('button', {
      name: /view system architecture & telemetry specs/i,
    });
    fireEvent.click(inlineSpecsBtn);

    expect(
      screen.getByRole('dialog', { name: /system architecture & telemetry/i })
    ).toBeInTheDocument();

    // Close via backdrop click
    const backdrop = screen.getByTestId('system-specs-backdrop');
    fireEvent.click(backdrop);
    expect(
      screen.queryByRole('dialog', { name: /system architecture & telemetry/i })
    ).not.toBeInTheDocument();
  });
});


