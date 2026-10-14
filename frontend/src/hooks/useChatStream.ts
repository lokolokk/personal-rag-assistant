import { useState, useRef, useCallback, useEffect } from 'react';
import { CitationDetail, DisplayMessage, StreamStatus } from '../types';

export interface UseChatStreamOptions {
  conversationId?: string | null;
  initialMessages?: DisplayMessage[];
}

export interface SendMessageOptions {
  documentIds?: string[];
}

export interface UseChatStreamReturn {
  messages: DisplayMessage[];
  isStreaming: boolean;
  status: StreamStatus;
  error: string | null;
  conversationId: string;
  sendMessage: (query: string, options?: SendMessageOptions) => Promise<void>;
  stopStream: () => void;
  clearMessages: () => void;
  setMessages: React.Dispatch<React.SetStateAction<DisplayMessage[]>>;
  startNewConversation: () => void;
  loadConversation: (newConvId: string, loadedMessages: DisplayMessage[]) => void;
}


function normalizeCitation(raw: Record<string, unknown>, index: number): CitationDetail {
  const rawId = raw.id ?? raw.citationIndex;
  const citationIndex = typeof rawId === 'number' ? rawId : index + 1;
  const chunkId =
    typeof raw.chunkId === 'string'
      ? raw.chunkId
      : typeof raw.chunk_id === 'string'
      ? raw.chunk_id
      : `chunk_${citationIndex}`;
  const documentId =
    typeof raw.documentId === 'string'
      ? raw.documentId
      : typeof raw.document_id === 'string'
      ? raw.document_id
      : '';
  const filename = typeof raw.filename === 'string' ? raw.filename : '';
  const rawPage = raw.pageNumber ?? raw.page_number ?? raw.page;
  const pageNumber = typeof rawPage === 'number' ? rawPage : 1;
  const sectionTitle =
    typeof raw.sectionTitle === 'string'
      ? raw.sectionTitle
      : typeof raw.section_title === 'string'
      ? raw.section_title
      : typeof raw.section === 'string'
      ? raw.section
      : '';
  const sourceText =
    typeof raw.sourceText === 'string'
      ? raw.sourceText
      : typeof raw.quote === 'string'
      ? raw.quote
      : typeof raw.text === 'string'
      ? raw.text
      : '';

  return {
    citationIndex,
    chunkId,
    documentId,
    filename,
    pageNumber,
    sectionTitle,
    sourceText,
  };
}

export function useChatStream({
  conversationId: initialConvId,
  initialMessages = [],
}: UseChatStreamOptions = {}): UseChatStreamReturn {
  const [messages, setMessages] = useState<DisplayMessage[]>(initialMessages);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [status, setStatus] = useState<StreamStatus>('idle');
  const [error, setError] = useState<string | null>(null);

  const [conversationId, setConversationId] = useState<string>(() => {
    if (initialConvId) {
      return initialConvId;
    }
    return `conv_${Math.random().toString(36).substring(2, 10)}`;
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  // Clean up any pending stream on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
  }, []);

  const stopStream = useCallback((): void => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
    setStatus('complete');
    setMessages((prev) =>
      prev.map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m))
    );
  }, []);

  const clearMessages = useCallback((): void => {
    setMessages([]);
    setError(null);
    setStatus('idle');
  }, []);

  const startNewConversation = useCallback((): void => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
    setStatus('idle');
    setError(null);
    setConversationId(`conv_${Math.random().toString(36).substring(2, 10)}`);
    setMessages(initialMessages);
  }, [initialMessages]);

  const loadConversation = useCallback(
    (newConvId: string, loadedMessages: DisplayMessage[]): void => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      setIsStreaming(false);
      setStatus('idle');
      setError(null);
      setConversationId(newConvId);
      setMessages(loadedMessages);
    },
    []
  );


  const sendMessage = useCallback(
    async (query: string, options?: SendMessageOptions): Promise<void> => {
      const trimmed = query.trim();
      if (!trimmed) {
        return;
      }

      setError(null);
      const userMsgId = `msg_user_${Date.now()}`;
      const assistantMsgId = `msg_asst_${Date.now() + 1}`;

      const userMsg: DisplayMessage = {
        id: userMsgId,
        role: 'user',
        content: trimmed,
        citations: [],
        createdAt: new Date().toISOString(),
      };

      const initialAssistantMsg: DisplayMessage = {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        citations: [],
        createdAt: new Date().toISOString(),
        status: 'reformulating',
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, initialAssistantMsg]);
      setIsStreaming(true);
      setStatus('reformulating');

      const controller = new AbortController();
      abortControllerRef.current = controller;

      try {
        const payload: Record<string, unknown> = {
          content: trimmed,
        };
        if (options?.documentIds !== undefined) {
          payload.documentIds = options.documentIds;
        }

        const endpoint = `/api/v1/conversations/${conversationId}/messages`;
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(
            `Streaming request failed with HTTP ${response.status} ${response.statusText}`
          );
        }

        if (!response.body) {
          throw new Error('Response body is null, SSE stream unavailable');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split(/\r?\n\r?\n/);
          // Keep trailing incomplete block in buffer
          buffer = parts.pop() ?? '';

          for (const block of parts) {
            if (!block.trim()) {
              continue;
            }

            const lines = block.split(/\r?\n/);
            let eventType = 'message';
            let dataRaw = '';

            for (const line of lines) {
              if (line.startsWith('event:')) {
                eventType = line.slice(6).trim();
              } else if (line.startsWith('data:')) {
                const chunkData = line.slice(5).trim();
                dataRaw = dataRaw ? `${dataRaw}\n${chunkData}` : chunkData;
              }
            }

            if (eventType === 'status') {
              try {
                const parsed = JSON.parse(dataRaw) as { status?: string; state?: string };
                const currentStage = (parsed.status || parsed.state || 'searching') as StreamStatus;
                setStatus(currentStage);
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMsgId ? { ...msg, status: currentStage } : msg
                  )
                );
              } catch {
                // Ignore JSON parse errors for malformed status
              }
            } else if (eventType === 'token') {
              try {
                const parsed = JSON.parse(dataRaw) as { token?: string; content?: string };
                const token = parsed.token ?? parsed.content ?? '';
                if (token) {
                  setStatus('streaming');
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMsgId
                        ? {
                            ...msg,
                            content: msg.content + token,
                            status: 'streaming',
                          }
                        : msg
                    )
                  );
                }
              } catch {
                // If not JSON, append raw text
                if (dataRaw) {
                  setStatus('streaming');
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMsgId
                        ? { ...msg, content: msg.content + dataRaw, status: 'streaming' }
                        : msg
                    )
                  );
                }
              }
            } else if (eventType === 'citations') {
              try {
                const parsed = JSON.parse(dataRaw);
                if (Array.isArray(parsed)) {
                  const normalized = parsed.map((item, idx) =>
                    normalizeCitation(item as Record<string, unknown>, idx)
                  );
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMsgId ? { ...msg, citations: normalized } : msg
                    )
                  );
                }
              } catch {
                // Ignore malformed citation payload
              }
            } else if (eventType === 'done') {
              setIsStreaming(false);
              setStatus('complete');
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMsgId
                    ? { ...msg, isStreaming: false, status: 'complete' }
                    : msg
                )
              );
            } else if (eventType === 'error') {
              try {
                const parsed = JSON.parse(dataRaw) as {
                  error?: { message?: string } | string;
                };
                const errMsg =
                  typeof parsed.error === 'object' && parsed.error?.message
                    ? parsed.error.message
                    : typeof parsed.error === 'string'
                    ? parsed.error
                    : 'Stream encountered an error';
                setError(errMsg);
                setStatus('error');
              } catch {
                setError(dataRaw || 'Stream encountered an error');
                setStatus('error');
              }
            }
          }
        }

        // Finalize stream state after reader completes
        setIsStreaming(false);
        setStatus('complete');
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? { ...msg, isStreaming: false, status: 'complete' }
              : msg
          )
        );
      } catch (err: unknown) {
        if (err instanceof Error && err.name === 'AbortError') {
          // Stream was cancelled by user
          return;
        }
        const errMsg = err instanceof Error ? err.message : 'Unknown streaming error occurred';
        setError(errMsg);
        setStatus('error');
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? {
                  ...msg,
                  isStreaming: false,
                  status: 'error',
                  error: errMsg,
                  content: msg.content || 'Failed to retrieve response due to connection error.',
                }
              : msg
          )
        );
      } finally {
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [conversationId]
  );

  return {
    messages,
    isStreaming,
    status,
    error,
    conversationId,
    sendMessage,
    stopStream,
    clearMessages,
    setMessages,
    startNewConversation,
    loadConversation,
  };
}

