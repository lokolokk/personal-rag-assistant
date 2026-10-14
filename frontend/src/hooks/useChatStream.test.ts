import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useChatStream } from './useChatStream';

describe('useChatStream hook', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('initializes with default state', () => {
    const { result } = renderHook(() => useChatStream({ conversationId: 'conv_123' }));

    expect(result.current.messages).toEqual([]);
    expect(result.current.isStreaming).toBe(false);
    expect(result.current.status).toBe('idle');
    expect(result.current.error).toBeNull();
    expect(typeof result.current.sendMessage).toBe('function');
    expect(typeof result.current.stopStream).toBe('function');
  });

  it('transmits documentIds in POST body to maintain active conversation scope', async () => {
    const streamData =
      'event: status\ndata: {"status": "searching"}\n\n' +
      'event: token\ndata: {"token": "Hello"}\n\n' +
      'event: done\ndata: {"finishReason": "stop"}\n\n';

    const encoder = new TextEncoder();
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(streamData));
        controller.close();
      },
    });

    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      body: mockStream,
    });
    vi.stubGlobal('fetch', mockFetch);

    const { result } = renderHook(() => useChatStream({ conversationId: 'conv_123' }));

    await act(async () => {
      await result.current.sendMessage('What is the method?', {
        documentIds: ['doc_1', 'doc_2'],
      });
    });

    expect(mockFetch).toHaveBeenCalledWith(
      '/api/v1/conversations/conv_123/messages',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: 'What is the method?',
          documentIds: ['doc_1', 'doc_2'],
        }),
      })
    );
  });

  it('transmits A -> all documents -> B and keeps omitted scope distinct', async () => {
    const mockFetch = vi.fn().mockImplementation(async () => ({
      ok: true,
      body: new ReadableStream({ start(controller) { controller.close(); } }),
    }));
    vi.stubGlobal('fetch', mockFetch);
    const { result } = renderHook(() => useChatStream({ conversationId: 'conv_scope' }));
    for (const documentIds of [['doc_A'], [], ['doc_B']]) {
      await act(async () => {
        await result.current.sendMessage('Question', { documentIds });
      });
    }
    await act(async () => { await result.current.sendMessage('Follow-up'); });
    const payloads = mockFetch.mock.calls.map((call) => JSON.parse(call[1].body));
    expect(payloads).toEqual([
      { content: 'Question', documentIds: ['doc_A'] },
      { content: 'Question', documentIds: [] },
      { content: 'Question', documentIds: ['doc_B'] },
      { content: 'Follow-up' },
    ]);
  });

  it('handles status event updates: reformulating -> searching -> grading', async () => {
    const streamData =
      'event: status\ndata: {"status": "reformulating"}\n\n' +
      'event: status\ndata: {"status": "searching"}\n\n' +
      'event: status\ndata: {"status": "grading"}\n\n' +
      'event: token\ndata: {"token": "Result"}\n\n' +
      'event: done\ndata: {"finishReason": "stop"}\n\n';

    const encoder = new TextEncoder();
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(streamData));
        controller.close();
      },
    });

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        body: mockStream,
      })
    );

    const { result } = renderHook(() => useChatStream({ conversationId: 'conv_123' }));

    await act(async () => {
      await result.current.sendMessage('Test query');
    });

    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0].role).toBe('user');
    expect(result.current.messages[0].content).toBe('Test query');
    expect(result.current.messages[1].role).toBe('assistant');
    expect(result.current.messages[1].content).toBe('Result');
    expect(result.current.isStreaming).toBe(false);
  });

  it('accumulates streaming tokens into the assistant message', async () => {
    const chunks = [
      'event: token\ndata: {"token": "Attention "}\n\n',
      'event: token\ndata: {"token": "is all "}\n\n',
      'event: token\ndata: {"token": "you need."}\n\n',
      'event: done\ndata: {"finishReason": "stop"}\n\n',
    ];

    const encoder = new TextEncoder();
    const mockStream = new ReadableStream({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(encoder.encode(chunk));
        }
        controller.close();
      },
    });

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        body: mockStream,
      })
    );

    const { result } = renderHook(() => useChatStream({ conversationId: 'conv_123' }));

    await act(async () => {
      await result.current.sendMessage('Explain the paper');
    });

    const assistantMsg = result.current.messages.find((m) => m.role === 'assistant');
    expect(assistantMsg).toBeDefined();
    expect(assistantMsg?.content).toBe('Attention is all you need.');
  });

  it('parses citations event and attaches normalized CitationDetail objects to message', async () => {
    const rawCitations = [
      {
        id: 1,
        documentId: 'doc_att',
        filename: 'attention.pdf',
        pageNumber: 3,
        sectionTitle: '3. Architecture',
        quote: 'The Transformer follows this overall architecture...',
        chunkId: 'doc_att_p3_c1',
      },
    ];

    const streamData =
      'event: token\ndata: {"token": "Architecture details [1]."}\n\n' +
      `event: citations\ndata: ${JSON.stringify(rawCitations)}\n\n` +
      'event: done\ndata: {"finishReason": "stop"}\n\n';

    const encoder = new TextEncoder();
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(streamData));
        controller.close();
      },
    });

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        body: mockStream,
      })
    );

    const { result } = renderHook(() => useChatStream({ conversationId: 'conv_123' }));

    await act(async () => {
      await result.current.sendMessage('Show citations');
    });

    const assistantMsg = result.current.messages[1];
    expect(assistantMsg.citations).toHaveLength(1);
    expect(assistantMsg.citations[0].citationIndex).toBe(1);
    expect(assistantMsg.citations[0].filename).toBe('attention.pdf');
    expect(assistantMsg.citations[0].pageNumber).toBe(3);
    expect(assistantMsg.citations[0].sourceText).toBe(
      'The Transformer follows this overall architecture...'
    );
  });

  it('handles stream cancellation via stopStream() using AbortController', async () => {
    let streamCancelled = false;
    const mockStream = new ReadableStream({
      start(_controller) {
        // Keeps stream open
      },
      cancel() {
        streamCancelled = true;
      },
    });

    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((_url, init) => {
        if (init?.signal) {
          init.signal.addEventListener('abort', () => {
            streamCancelled = true;
          });
        }
        return Promise.resolve({
          ok: true,
          body: mockStream,
        });
      })
    );

    const { result } = renderHook(() => useChatStream({ conversationId: 'conv_123' }));

    act(() => {
      void result.current.sendMessage('Long task');
    });

    expect(result.current.isStreaming).toBe(true);

    act(() => {
      result.current.stopStream();
    });

    expect(streamCancelled).toBe(true);
    expect(result.current.isStreaming).toBe(false);
  });

  it('handles network failure and sets error message gracefully', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new Error('Network connection failed'))
    );

    const { result } = renderHook(() => useChatStream({ conversationId: 'conv_123' }));

    await act(async () => {
      await result.current.sendMessage('Failing query');
    });

    expect(result.current.isStreaming).toBe(false);
    expect(result.current.error).toContain('Network connection failed');
  });

  it('startNewConversation resets conversation ID and initial messages', () => {
    const welcomeMsg = {
      id: 'welcome',
      role: 'assistant' as const,
      content: 'Welcome',
      citations: [],
      createdAt: '2026-10-02T10:00:00Z',
    };

    const { result } = renderHook(() =>
      useChatStream({
        conversationId: 'conv_initial',
        initialMessages: [welcomeMsg],
      })
    );

    expect(result.current.conversationId).toBe('conv_initial');

    act(() => {
      result.current.startNewConversation();
    });

    expect(result.current.conversationId).not.toBe('conv_initial');
    expect(result.current.conversationId).toMatch(/^conv_/);
    expect(result.current.messages).toEqual([welcomeMsg]);
    expect(result.current.isStreaming).toBe(false);
    expect(result.current.status).toBe('idle');
  });

  it('loadConversation updates conversation ID and replaces messages', () => {
    const { result } = renderHook(() => useChatStream({ conversationId: 'conv_old' }));

    const loadedMessages = [
      {
        id: 'msg_1',
        role: 'user' as const,
        content: 'Historical question',
        citations: [],
        createdAt: '2026-10-02T09:00:00Z',
      },
      {
        id: 'msg_2',
        role: 'assistant' as const,
        content: 'Historical answer',
        citations: [],
        createdAt: '2026-10-02T09:01:00Z',
      },
    ];

    act(() => {
      result.current.loadConversation('conv_new_selected', loadedMessages);
    });

    expect(result.current.conversationId).toBe('conv_new_selected');
    expect(result.current.messages).toEqual(loadedMessages);
    expect(result.current.isStreaming).toBe(false);
  });
});

