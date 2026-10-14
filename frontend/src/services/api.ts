/**
 * Base HTTP client and typed endpoints for Conversational Document RAG backend API.
 */
import { DocumentItem } from '../types';

const API_BASE_URL: string = (import.meta.env.VITE_API_URL as string | undefined) ?? '/api/v1';

export class ApiError extends Error {
  status: number;
  details?: unknown;

  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

export async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const rawText = await response.text();
    let errorData: unknown;
    try {
      errorData = JSON.parse(rawText);
    } catch {
      errorData = rawText;
    }
    const detailMessage =
      typeof errorData === 'object' && errorData !== null && 'detail' in errorData
        ? String((errorData as { detail: unknown }).detail)
        : typeof errorData === 'string' && errorData.trim().length > 0
          ? errorData.trim()
          : `API request failed with status ${response.status}`;
    throw new ApiError(detailMessage, response.status, errorData);
  }

  return response.json() as Promise<T>;
}

export function normalizeDocumentItem(
  raw: Partial<DocumentItem> & { page_count?: number; chunk_count?: number; uploaded_at?: string }
): DocumentItem {
  return {
    id: raw.id ?? raw.documentId ?? '',
    filename: raw.filename ?? 'unknown.pdf',
    pageCount: raw.pageCount ?? raw.page_count ?? 0,
    chunkCount: raw.chunkCount ?? raw.chunk_count ?? 0,
    uploadedAt: raw.uploadedAt ?? raw.uploaded_at ?? new Date().toISOString(),
    status: raw.status ?? 'ready',
  };
}

export async function fetchDocuments(): Promise<DocumentItem[]> {
  const rawList = await request<
    Array<Partial<DocumentItem> & { page_count?: number; chunk_count?: number; uploaded_at?: string }>
  >('/documents');
  return rawList.map(normalizeDocumentItem);
}

export async function uploadDocument(file: File): Promise<DocumentItem> {
  const url = `${API_BASE_URL}/documents/upload`;
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const rawText = await response.text();
    let errorData: unknown;
    try {
      errorData = JSON.parse(rawText);
    } catch {
      errorData = rawText;
    }
    const detailMessage =
      typeof errorData === 'object' && errorData !== null && 'detail' in errorData
        ? String((errorData as { detail: unknown }).detail)
        : typeof errorData === 'string' && errorData.trim().length > 0
          ? errorData.trim()
          : `Upload failed with status ${response.status}`;
    throw new ApiError(detailMessage, response.status, errorData);
  }

  const rawItem = (await response.json()) as Partial<DocumentItem> & {
    page_count?: number;
    chunk_count?: number;
    uploaded_at?: string;
  };
  return normalizeDocumentItem(rawItem);
}

export async function deleteDocument(documentId: string): Promise<void> {
  await request<{ message: string; documentId: string }>(`/documents/${documentId}`, {
    method: 'DELETE',
  });
}

export function normalizeConversationItem(
  raw: Partial<import('../types').ConversationItem> & {
    document_ids?: string[];
    created_at?: string;
    updated_at?: string;
    message_count?: number;
    conversationId?: string;
  }
): import('../types').ConversationItem {
  return {
    id: raw.id ?? raw.conversationId ?? '',
    title: raw.title ?? 'New Chat',
    createdAt: raw.createdAt ?? raw.created_at ?? new Date().toISOString(),
    updatedAt: raw.updatedAt ?? raw.updated_at ?? new Date().toISOString(),
    documentIds: raw.documentIds ?? raw.document_ids ?? [],
    messageCount: raw.messageCount ?? raw.message_count ?? 0,
  };
}

export async function fetchConversations(): Promise<import('../types').ConversationItem[]> {
  const rawList = await request<
    Array<
      Partial<import('../types').ConversationItem> & {
        document_ids?: string[];
        created_at?: string;
        updated_at?: string;
        message_count?: number;
        conversationId?: string;
      }
    >
  >('/conversations');
  return rawList.map(normalizeConversationItem);
}

export async function deleteConversation(conversationId: string): Promise<void> {
  await request<{ status: string; id: string }>(`/conversations/${conversationId}`, {
    method: 'DELETE',
  });
}

export async function renameConversation(
  conversationId: string,
  title: string
): Promise<import('../types').ConversationItem> {
  const raw = await request<
    Partial<import('../types').ConversationItem> & {
      document_ids?: string[];
      created_at?: string;
      updated_at?: string;
      message_count?: number;
      conversationId?: string;
    }
  >(`/conversations/${conversationId}`, {
    method: 'PATCH',
    body: JSON.stringify({ title }),
  });
  return normalizeConversationItem(raw);
}


export interface BackendCitationItem {
  id?: number;
  citationIndex?: number;
  chunk_id?: string;
  chunkId?: string;
  document_id?: string;
  documentId?: string;
  filename?: string;
  page_number?: number;
  pageNumber?: number;
  page?: number;
  section_title?: string;
  sectionTitle?: string;
  quote?: string;
  sourceText?: string;
}

export interface BackendChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
  citations?: Array<number | BackendCitationItem>;
}

export async function fetchConversationMessages(
  conversationId: string
): Promise<BackendChatMessage[]> {
  return request<BackendChatMessage[]>(`/conversations/${conversationId}/messages`);
}

