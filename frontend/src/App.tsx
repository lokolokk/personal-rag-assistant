import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import {
  BookOpen,
  PanelLeftClose,
  AlertCircle,
  Plus,
  Upload,
} from 'lucide-react';
import { DocumentItem, CitationDetail, ConversationItem, DisplayMessage } from './types';
import { DocumentList } from './components/documents/DocumentList';
import { DocumentUploadModal } from './components/documents/DocumentUploadModal';
import { ChatContainer } from './components/chat/ChatContainer';
import { ChatHeader } from './components/chat/ChatHeader';
import { SystemSpecsModal } from './components/chat/SystemSpecsModal';
import { CitationDrawer } from './components/inspector/CitationDrawer';
import { ConversationList } from './components/sidebar/ConversationList';
import { useChatStream } from './hooks/useChatStream';
import {
  fetchDocuments,
  deleteDocument,
  fetchConversations,
  deleteConversation,
  renameConversation,
  fetchConversationMessages,
} from './services/api';

export default function App(): React.JSX.Element {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState<boolean>(true);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState<boolean>(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(true);
  const [isSystemSpecsOpen, setIsSystemSpecsOpen] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Conversation Session State
  const [conversations, setConversations] = useState<ConversationItem[]>([]);
  const [isLoadingConversations, setIsLoadingConversations] = useState<boolean>(false);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);

  const {
    conversationId: currentConversationId,
    messages,
    isStreaming,
    sendMessage,
    stopStream,
    startNewConversation,
    loadConversation,
    error: streamError,
  } = useChatStream({
    initialMessages: [],
  });

  const [activeCitation, setActiveCitation] = useState<CitationDetail | null>(null);
  const [isCitationDrawerOpen, setIsCitationDrawerOpen] = useState<boolean>(false);

  const handleSelectCitation = (citation: CitationDetail): void => {
    setActiveCitation(citation);
    setIsCitationDrawerOpen(true);
  };

  const handleCloseCitationDrawer = (): void => {
    setIsCitationDrawerOpen(false);
  };

  // Derive citations from the message that contains activeCitation
  const activeMessageCitations = useMemo(() => {
    if (!activeCitation) return [];
    const parentMsg = messages.find((m) =>
      m.citations.some(
        (c) =>
          c.citationIndex === activeCitation.citationIndex ||
          (c.chunkId && c.chunkId === activeCitation.chunkId)
      )
    );
    return parentMsg && parentMsg.citations.length > 0 ? parentMsg.citations : [activeCitation];
  }, [messages, activeCitation]);

  // Load conversations list
  const loadConversations = useCallback(async (): Promise<void> => {
    try {
      setIsLoadingConversations(true);
      const convList = await fetchConversations();
      setConversations(convList);
    } catch (err: unknown) {
      console.error('Failed to load conversations:', err);
    } finally {
      setIsLoadingConversations(false);
    }
  }, []);

  // Load indexed documents on mount
  useEffect(() => {
    let isMounted = true;

    async function loadInitialData(): Promise<void> {
      try {
        setIsLoadingDocs(true);
        const docs = await fetchDocuments();
        if (isMounted) {
          setDocuments(docs);
        }
      } catch (err: unknown) {
        if (isMounted) {
          const msg =
            err instanceof Error ? err.message : 'Failed to fetch indexed documents from server';
          setErrorMessage(msg);
        }
      } finally {
        if (isMounted) {
          setIsLoadingDocs(false);
        }
      }

      if (isMounted) {
        void loadConversations();
      }
    }

    void loadInitialData();

    return () => {
      isMounted = false;
    };
  }, [loadConversations]);

  // Refresh conversations list whenever streaming finishes
  const prevStreamingRef = useRef<boolean>(isStreaming);
  useEffect(() => {
    if (prevStreamingRef.current && !isStreaming) {
      void loadConversations();
    }
    prevStreamingRef.current = isStreaming;
  }, [isStreaming, loadConversations]);

  const handleToggleSelectDocument = (id: string): void => {
    setSelectedDocIds((prev) =>
      prev.includes(id) ? prev.filter((docId) => docId !== id) : [...prev, id]
    );
  };

  const handleSelectAllDocuments = (): void => {
    setSelectedDocIds(documents.map((d) => d.id));
  };

  const handleClearSelection = (): void => {
    setSelectedDocIds([]);
  };

  const handleUploadSuccess = (newDoc: DocumentItem): void => {
    setDocuments((prev) => {
      const filtered = prev.filter((d) => d.id !== newDoc.id);
      return [...filtered, newDoc];
    });
    // Auto-select newly uploaded document for immediate scoped querying
    setSelectedDocIds((prev) => (prev.includes(newDoc.id) ? prev : [...prev, newDoc.id]));
  };

  const handleDeleteDocument = async (documentId: string): Promise<void> => {
    const previousDocuments = documents;
    const previousSelected = selectedDocIds;

    // Optimistic UI update
    setDocuments((prev) => prev.filter((d) => d.id !== documentId));
    setSelectedDocIds((prev) => prev.filter((id) => id !== documentId));

    try {
      await deleteDocument(documentId);
    } catch (err: unknown) {
      // Rollback on failure
      setDocuments(previousDocuments);
      setSelectedDocIds(previousSelected);
      const msg = err instanceof Error ? err.message : 'Failed to delete document from index';
      setErrorMessage(msg);
    }
  };

  // Start fresh conversation session ("+ New Chat")
  const handleNewChat = (): void => {
    startNewConversation();
    setActiveConversationId(null);
    setActiveCitation(null);
    setIsCitationDrawerOpen(false);
  };

  // Select and switch to an existing conversation session
  const handleSelectConversation = async (conversation: ConversationItem): Promise<void> => {
    try {
      setActiveConversationId(conversation.id);
      setIsCitationDrawerOpen(false);
      setActiveCitation(null);

      // Restore document scoping from the session
      if (conversation.documentIds) {
        setSelectedDocIds(conversation.documentIds);
      }

      // Load session messages from API
      const rawMessages = await fetchConversationMessages(conversation.id);
      const convertedMessages: DisplayMessage[] = rawMessages
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .map((m, idx) => ({
          id: `conv_${conversation.id}_${idx}`,
          role: m.role as 'user' | 'assistant',
          content: m.content,
          citations: (m.citations || []).map((c, cIdx) => {
            if (typeof c === 'number') {
              return {
                id: c,
                citationIndex: c,
                chunkId: `chunk_${c}`,
                documentId: '',
                filename: '',
                pageNumber: 1,
                sectionTitle: '',
                sourceText: '',
              };
            }
            const citObj = c as Record<string, unknown>;
            const citNum =
              typeof citObj.citationIndex === 'number'
                ? citObj.citationIndex
                : typeof citObj.citation_index === 'number'
                ? citObj.citation_index
                : typeof citObj.id === 'number'
                ? citObj.id
                : cIdx + 1;
            const citFilename =
              typeof citObj.filename === 'string'
                ? citObj.filename
                : typeof citObj.document_filename === 'string'
                ? citObj.document_filename
                : '';
            const citPage =
              typeof citObj.pageNumber === 'number'
                ? citObj.pageNumber
                : typeof citObj.page_number === 'number'
                ? citObj.page_number
                : typeof citObj.page === 'number'
                ? citObj.page
                : 1;
            const citSection =
              typeof citObj.sectionTitle === 'string'
                ? citObj.sectionTitle
                : typeof citObj.section_title === 'string'
                ? citObj.section_title
                : typeof citObj.section === 'string'
                ? citObj.section
                : '';
            const citSourceText =
              typeof citObj.sourceText === 'string'
                ? citObj.sourceText
                : typeof citObj.source_text === 'string'
                ? citObj.source_text
                : typeof citObj.quote === 'string'
                ? citObj.quote
                : '';
            const citDocId =
              typeof citObj.documentId === 'string'
                ? citObj.documentId
                : typeof citObj.document_id === 'string'
                ? citObj.document_id
                : '';
            const citChunkId =
              typeof citObj.chunkId === 'string'
                ? citObj.chunkId
                : typeof citObj.chunk_id === 'string'
                ? citObj.chunk_id
                : `chunk_${citNum}`;

            return {
              id: citNum,
              citationIndex: citNum,
              chunkId: citChunkId,
              documentId: citDocId,
              filename: citFilename,
              pageNumber: citPage,
              sectionTitle: citSection,
              sourceText: citSourceText,
            };
          }),
          createdAt: new Date().toISOString(),
        }));

      loadConversation(conversation.id, convertedMessages);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to switch conversation session';
      setErrorMessage(msg);
    }
  };

  // Delete conversation session
  const handleDeleteConversation = async (conversationId: string): Promise<void> => {
    const prevConversations = conversations;
    setConversations((prev) => prev.filter((c) => c.id !== conversationId));

    if (activeConversationId === conversationId) {
      handleNewChat();
    }

    try {
      await deleteConversation(conversationId);
    } catch (err: unknown) {
      setConversations(prevConversations);
      const msg = err instanceof Error ? err.message : 'Failed to delete conversation session';
      setErrorMessage(msg);
    }
  };

  const handleRenameConversation = async (newTitle: string): Promise<void> => {
    const targetId = activeConversationId ?? currentConversationId;
    if (!targetId) return;

    const prevConversations = conversations;
    setConversations((prev) =>
      prev.map((c) => (c.id === targetId ? { ...c, title: newTitle } : c))
    );

    try {
      await renameConversation(targetId, newTitle);
    } catch (err: unknown) {
      setConversations(prevConversations);
      const msg = err instanceof Error ? err.message : 'Failed to rename conversation';
      setErrorMessage(msg);
    }
  };

  const handleSendMessage = (query: string): void => {
    void sendMessage(query, { documentIds: selectedDocIds });
  };

  const activeError = errorMessage ?? streamError;
  const effectiveActiveConvId = activeConversationId ?? currentConversationId ?? null;
  const activeConversation = useMemo(() => {
    if (!effectiveActiveConvId) return null;
    return conversations.find((c) => c.id === effectiveActiveConvId) ?? null;
  }, [conversations, effectiveActiveConvId]);
  const activeTitle = activeConversation?.title ?? 'New Conversation';

  return (
    <div className="flex h-full bg-[#09090b] text-zinc-100 overflow-hidden font-sans relative">
      {/* Left Pane: Collapsible Tray (Documents & Conversation Sessions) */}
      <aside
        className={`border-r border-zinc-800 bg-[#141417] flex flex-col transition-all duration-300 ease-in-out shrink-0 relative z-20 ${
          isSidebarOpen ? 'w-80' : 'w-0 overflow-hidden border-r-0'
        }`}
      >
        <div className="w-80 h-full flex flex-col bg-[#141417]">
          {/* Compact 40px Workstation Toolbar */}
          <div className="h-10 border-b border-zinc-800 px-3 flex items-center justify-between bg-[#141417] shrink-0">
            <div className="flex items-center space-x-2">
              <div className="p-1 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                <BookOpen className="w-3.5 h-3.5" />
              </div>
              <span className="font-semibold text-xs text-zinc-100 tracking-tight font-sans">
                RAG Nexus
              </span>
            </div>

            <div className="flex items-center space-x-1">
              <button
                type="button"
                onClick={handleNewChat}
                className="p-1.5 rounded-md text-zinc-400 hover:text-indigo-400 hover:bg-zinc-800/80 transition cursor-pointer"
                title="New conversation (Ctrl+N)"
                aria-label="New Chat"
              >
                <Plus className="w-3.5 h-3.5" />
              </button>

              <button
                type="button"
                onClick={() => setIsUploadModalOpen(true)}
                className="p-1.5 rounded-md text-zinc-400 hover:text-indigo-400 hover:bg-zinc-800/80 transition cursor-pointer"
                title="Upload technical documents"
                aria-label="Upload PDF"
              >
                <Upload className="w-3.5 h-3.5" />
              </button>

              <button
                type="button"
                onClick={() => setIsSidebarOpen(false)}
                className="p-1.5 rounded-md text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/80 transition cursor-pointer"
                title="Collapse documents tray"
                aria-label="Collapse documents tray"
              >
                <PanelLeftClose className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Unified Vertical Explorer Tree (Zero Tab Flipping) */}
          <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
            <DocumentList
              documents={documents}
              selectedDocIds={selectedDocIds}
              isLoading={isLoadingDocs}
              onToggleSelect={handleToggleSelectDocument}
              onSelectAll={handleSelectAllDocuments}
              onClearSelection={handleClearSelection}
              onDeleteDocument={handleDeleteDocument}
              onOpenUpload={() => setIsUploadModalOpen(true)}
            />
            <ConversationList
              conversations={conversations}
              activeConversationId={effectiveActiveConvId}
              isLoading={isLoadingConversations}
              onSelectConversation={(conv) => void handleSelectConversation(conv)}
              onDeleteConversation={(id) => void handleDeleteConversation(id)}
              onNewChat={handleNewChat}
              hideNewChatButton={true}
            />
          </div>
        </div>
      </aside>

      {/* Center Pane: Conversational Chat Canvas */}
      <main className="flex-1 flex flex-col bg-[#09090b] overflow-hidden min-w-0 relative">
        {/* Top 40px Integrated Canvas Header */}
        <ChatHeader
          title={activeTitle}
          isSidebarOpen={isSidebarOpen}
          onToggleSidebar={() => setIsSidebarOpen(true)}
          onRename={handleRenameConversation}
          onOpenSpecs={() => setIsSystemSpecsOpen(true)}
        />

        {/* Global Error Notification Banner */}
        {activeError && (
          <div
            role="alert"
            className="bg-rose-950/70 border-b border-rose-800/80 px-4 py-2 flex items-center justify-between text-xs text-rose-300 animate-in fade-in z-30 shrink-0"
          >
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{activeError}</span>
            </div>
            <button
              type="button"
              onClick={() => setErrorMessage(null)}
              className="text-rose-400 hover:text-rose-200 font-bold ml-4 cursor-pointer"
            >
              ✕
            </button>
          </div>
        )}

        <ChatContainer
          messages={messages}
          isStreaming={isStreaming}
          selectedDocCount={selectedDocIds.length}
          onSendMessage={handleSendMessage}
          onStopStream={stopStream}
          onSelectCitation={handleSelectCitation}
          onOpenSpecs={() => setIsSystemSpecsOpen(true)}
        />
      </main>

      {/* Interactive Citation Inspector Slide-over Drawer */}
      <CitationDrawer
        isOpen={isCitationDrawerOpen}
        citation={activeCitation}
        citations={activeMessageCitations}
        onClose={handleCloseCitationDrawer}
        onSelectCitation={setActiveCitation}
      />

      {/* Document Upload Modal */}
      <DocumentUploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onUploadSuccess={handleUploadSuccess}
      />

      {/* System Architecture & Telemetry Modal */}
      <SystemSpecsModal
        isOpen={isSystemSpecsOpen}
        onClose={() => setIsSystemSpecsOpen(false)}
        selectedDocCount={selectedDocIds.length}
        totalDocCount={documents.length}
      />
    </div>
  );
}
