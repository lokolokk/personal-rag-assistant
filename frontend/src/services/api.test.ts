import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  fetchDocuments,
  uploadDocument,
  deleteDocument,
  normalizeDocumentItem,
  fetchConversations,
  deleteConversation,
  renameConversation,
  fetchConversationMessages,
  normalizeConversationItem,
  ApiError,
} from './api';


describe('API Service', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('normalizes document item with snake_case and camelCase fallback fields', () => {
    const raw = {
      id: 'doc_1',
      filename: 'sample.pdf',
      page_count: 10,
      chunk_count: 25,
      uploaded_at: '2026-10-01T12:00:00Z',
      status: 'ready' as const,
    };

    const normalized = normalizeDocumentItem(raw);
    expect(normalized.id).toBe('doc_1');
    expect(normalized.filename).toBe('sample.pdf');
    expect(normalized.pageCount).toBe(10);
    expect(normalized.chunkCount).toBe(25);
    expect(normalized.uploadedAt).toBe('2026-10-01T12:00:00Z');
    expect(normalized.status).toBe('ready');
  });

  it('fetchDocuments fetches and returns normalized documents', async () => {
    const mockApiResponse = [
      {
        id: 'doc_1',
        filename: 'paper.pdf',
        page_count: 8,
        chunk_count: 20,
        uploaded_at: '2026-10-02T10:00:00Z',
        status: 'ready',
      },
    ];

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockApiResponse,
    } as unknown as Response);

    const result = await fetchDocuments();
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('doc_1');
    expect(result[0].pageCount).toBe(8);
    expect(result[0].chunkCount).toBe(20);
  });

  it('uploadDocument sends multipart/form-data and returns normalized item', async () => {
    const mockCreated = {
      id: 'doc_uploaded',
      filename: 'new.pdf',
      page_count: 5,
      chunk_count: 12,
      uploaded_at: '2026-10-02T11:00:00Z',
      status: 'ready',
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockCreated,
    } as unknown as Response);

    const mockFile = new File(['%PDF content'], 'new.pdf', { type: 'application/pdf' });
    const result = await uploadDocument(mockFile);

    expect(result.id).toBe('doc_uploaded');
    expect(result.pageCount).toBe(5);
    expect(result.chunkCount).toBe(12);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/documents/upload'),
      expect.objectContaining({
        method: 'POST',
        body: expect.any(FormData),
      })
    );
  });

  it('uploadDocument throws ApiError with message on failure', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 413,
      text: async () => JSON.stringify({ detail: 'File size exceeds 30MB limit' }),
      json: async () => ({ detail: 'File size exceeds 30MB limit' }),
    } as unknown as Response);

    const mockFile = new File([''], 'too_large.pdf', { type: 'application/pdf' });
    await expect(uploadDocument(mockFile)).rejects.toThrow(ApiError);
  });

  it('uploadDocument throws ApiError with plain text message when response is not JSON without body stream already read error', async () => {
    let bodyConsumed = false;
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      text: async () => {
        if (bodyConsumed) {
          throw new TypeError("Failed to execute 'text' on 'Response': body stream already read");
        }
        bodyConsumed = true;
        return 'Internal Server Error: DashScope 400';
      },
      json: async () => {
        if (bodyConsumed) {
          throw new TypeError("Failed to execute 'json' on 'Response': body stream already read");
        }
        bodyConsumed = true;
        throw new SyntaxError('Unexpected token I in JSON at position 0');
      },
    } as unknown as Response);

    const mockFile = new File([''], 'large_tada.pdf', { type: 'application/pdf' });
    await expect(uploadDocument(mockFile)).rejects.toThrow('Internal Server Error: DashScope 400');
  });

  it('deleteDocument sends DELETE request to document endpoint', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ message: 'Deleted', documentId: 'doc_1' }),
    } as unknown as Response);

    await deleteDocument('doc_1');
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/documents/doc_1'),
      expect.objectContaining({
        method: 'DELETE',
      })
    );
  });

  it('normalizes conversation item with fallback fields', () => {
    const raw = {
      id: 'conv_1',
      title: 'Attention Analysis',
      document_ids: ['doc_1'],
      message_count: 4,
      created_at: '2026-10-02T12:00:00Z',
      updated_at: '2026-10-02T12:05:00Z',
    };

    const normalized = normalizeConversationItem(raw);
    expect(normalized.id).toBe('conv_1');
    expect(normalized.title).toBe('Attention Analysis');
    expect(normalized.documentIds).toEqual(['doc_1']);
    expect(normalized.messageCount).toBe(4);
    expect(normalized.createdAt).toBe('2026-10-02T12:00:00Z');
    expect(normalized.updatedAt).toBe('2026-10-02T12:05:00Z');
  });

  it('fetchConversations retrieves and normalizes conversation list', async () => {
    const mockList = [
      {
        id: 'conv_1',
        title: 'Query 1',
        document_ids: ['doc_1'],
        message_count: 2,
        updated_at: '2026-10-02T14:00:00Z',
      },
    ];

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockList,
    } as unknown as Response);

    const result = await fetchConversations();
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('conv_1');
    expect(result[0].title).toBe('Query 1');
    expect(result[0].documentIds).toEqual(['doc_1']);
  });

  it('deleteConversation sends DELETE to conversations endpoint', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'deleted', id: 'conv_1' }),
    } as unknown as Response);

    await deleteConversation('conv_1');
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/conversations/conv_1'),
      expect.objectContaining({ method: 'DELETE' })
    );
  });

  it('fetchConversationMessages fetches messages for a session', async () => {
    const mockMsgs = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'World' },
    ];

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockMsgs,
    } as unknown as Response);

    const result = await fetchConversationMessages('conv_1');
    expect(result).toHaveLength(2);
    expect(result[0].role).toBe('user');
    expect(result[1].content).toBe('World');
  });

  it('fetchConversationMessages preserves rich citation objects in assistant message', async () => {
    const mockMsgs = [
      { role: 'user', content: 'What is attention?' },
      {
        role: 'assistant',
        content: 'Attention is a mechanism [1].',
        citations: [
          {
            id: 1,
            citationIndex: 1,
            chunk_id: 'chunk_1',
            document_id: 'doc_1',
            filename: 'attention.pdf',
            page_number: 3,
            section_title: 'Attention',
            quote: 'An attention function can be described as mapping a query...',
          },
        ],
      },
    ];

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockMsgs,
    } as unknown as Response);

    const result = await fetchConversationMessages('conv_attention');
    expect(result).toHaveLength(2);
    expect(result[1].citations).toHaveLength(1);
    const firstCit = result[1].citations?.[0];
    expect(firstCit).toEqual(
      expect.objectContaining({
        filename: 'attention.pdf',
        page_number: 3,
      })
    );
  });

  it('renameConversation sends PATCH with title and returns normalized item', async () => {
    const mockUpdated = {
      id: 'conv_1',
      title: 'Updated Transformer Analysis',
      document_ids: ['doc_1'],
      message_count: 4,
      updated_at: '2026-10-02T15:00:00Z',
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockUpdated,
    } as unknown as Response);

    const result = await renameConversation('conv_1', 'Updated Transformer Analysis');
    expect(result.id).toBe('conv_1');
    expect(result.title).toBe('Updated Transformer Analysis');
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/conversations/conv_1'),
      expect.objectContaining({
        method: 'PATCH',
        body: JSON.stringify({ title: 'Updated Transformer Analysis' }),
      })
    );
  });
});


