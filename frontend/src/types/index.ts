/**
 * Domain types and models for Conversational Document RAG frontend.
 */

export interface DocumentItem {
  id: string;
  filename: string;
  pageCount: number;
  chunkCount: number;
  uploadedAt: string;
  status: 'processing' | 'ready' | 'error';
  page_count?: number;
  chunk_count?: number;
  uploaded_at?: string;
  documentId?: string;
}

export interface CitationDetail {
  citationIndex: number;
  chunkId: string;
  documentId: string;
  filename: string;
  pageNumber: number;
  sectionTitle: string;
  sourceText: string;
  id?: number;
  quote?: string;
  page_number?: number;
  section_title?: string;
  chunk_id?: string;
  document_id?: string;
}

export type StreamStatus =
  | 'idle'
  | 'reformulating'
  | 'searching'
  | 'grading'
  | 'streaming'
  | 'complete'
  | 'error';

export interface DisplayMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations: CitationDetail[];
  createdAt: string;
  status?: StreamStatus;
  isStreaming?: boolean;
  error?: string;
}

export interface ConversationItem {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  documentIds: string[];
  messageCount: number;
  created_at?: string;
  updated_at?: string;
  document_ids?: string[];
  message_count?: number;
  conversationId?: string;
}

export interface AppState {
  documents: DocumentItem[];
  conversations: ConversationItem[];
  selectedDocumentIds: string[];
  activeConversationId: string | null;
  messages: DisplayMessage[];
  isStreaming: boolean;
  selectedCitation: CitationDetail | null;
}


