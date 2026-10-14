import React, { useState, useRef, DragEvent, ChangeEvent } from 'react';
import { UploadCloud, X, FileText, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { DocumentItem } from '../../types';
import { uploadDocument } from '../../services/api';

const MAX_FILE_SIZE_BYTES = 30 * 1024 * 1024; // 30 MB

export interface DocumentUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess: (document: DocumentItem) => void;
  uploadFn?: (file: File) => Promise<DocumentItem>;
}

export const DocumentUploadModal: React.FC<DocumentUploadModalProps> = ({
  isOpen,
  onClose,
  onUploadSuccess,
  uploadFn,
}) => {
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadStage, setUploadStage] = useState<'uploading' | 'parsing' | 'indexing' | 'idle'>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const effectiveUpload = uploadFn ?? uploadDocument;

  if (!isOpen) {
    return null;
  }

  const resetState = (): void => {
    setSelectedFile(null);
    setIsUploading(false);
    setUploadStage('idle');
    setErrorMessage(null);
    setIsDragging(false);
  };

  const handleClose = (): void => {
    if (isUploading) {
      return;
    }
    resetState();
    onClose();
  };

  const validateFile = (file: File): boolean => {
    setErrorMessage(null);

    const isPdf = file.name.toLowerCase().endsWith('.pdf') || file.type === 'application/pdf';
    if (!isPdf) {
      setErrorMessage('Only .pdf files are supported.');
      return false;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      setErrorMessage('File size exceeds 30MB limit.');
      return false;
    }

    return true;
  };

  const processAndUploadFile = async (file: File): Promise<void> => {
    if (!validateFile(file)) {
      return;
    }

    setSelectedFile(file);
    setIsUploading(true);
    setUploadStage('uploading');

    try {
      // Simulate progress stages for smooth user feedback
      const stageTimer1 = setTimeout(() => {
        setUploadStage('parsing');
      }, 600);
      const stageTimer2 = setTimeout(() => {
        setUploadStage('indexing');
      }, 1400);

      const uploadedDoc = await effectiveUpload(file);

      clearTimeout(stageTimer1);
      clearTimeout(stageTimer2);

      resetState();
      onUploadSuccess(uploadedDoc);
      onClose();
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Upload failed. Please check the backend connection.';
      setErrorMessage(message);
      setIsUploading(false);
      setUploadStage('idle');
    }
  };

  const handleDragEnter = (e: DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (isUploading) {
      return;
    }

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      void processAndUploadFile(droppedFile);
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>): void => {
    if (e.target.files && e.target.files.length > 0) {
      const chosenFile = e.target.files[0];
      void processAndUploadFile(chosenFile);
    }
  };

  const triggerFilePicker = (): void => {
    if (!isUploading && fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="upload-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs transition-opacity"
      onClick={handleClose}
    >
      <div
        className="w-full max-w-lg bg-[#141417] rounded-xl shadow-2xl border border-zinc-800 overflow-hidden transform transition-all text-zinc-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-zinc-800 flex items-center justify-between bg-[#0e0e11]">
          <div>
            <h2 id="upload-modal-title" className="text-base font-semibold text-zinc-100">
              Upload Technical Documents
            </h2>
            <p className="text-xs text-zinc-400 mt-0.5">
              Extract layout, isolate tables, and index into hybrid vector store
            </p>
          </div>
          <button
            type="button"
            aria-label="Close modal"
            onClick={handleClose}
            disabled={isUploading}
            className="p-1 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition disabled:opacity-40 cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-4">
          {/* Dropzone */}
          <div
            data-testid="upload-dropzone"
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={triggerFilePicker}
            className={`cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-all ${
              isDragging
                ? 'border-indigo-500 bg-indigo-950/20 ring-4 ring-indigo-500/10'
                : 'border-zinc-700/80 hover:border-zinc-600 bg-[#09090b]/50 hover:bg-[#18181c]/60'
            } ${isUploading ? 'pointer-events-none opacity-60' : ''}`}
          >
            <input
              ref={fileInputRef}
              type="file"
              data-testid="file-input"
              accept=".pdf,application/pdf"
              className="hidden"
              onChange={handleFileInputChange}
            />

            <div className="flex flex-col items-center justify-center space-y-3">
              <div
                className={`p-3 rounded-xl ${
                  isDragging ? 'bg-indigo-900/40 text-indigo-400' : 'bg-zinc-800 text-zinc-300'
                }`}
              >
                <UploadCloud className="w-8 h-8 stroke-[1.5]" />
              </div>
              <div>
                <p className="text-sm font-medium text-zinc-200">
                  <span className="text-indigo-400 hover:underline">Click to browse</span> or drag & drop your PDF here
                </p>
                <p className="text-xs text-zinc-400 mt-1 font-mono">
                  Only .pdf files up to 30MB supported
                </p>
              </div>
            </div>
          </div>

          {/* Validation / API Error Alert */}
          {errorMessage && (
            <div
              role="alert"
              className="flex items-start space-x-2.5 p-3 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs animate-in fade-in"
            >
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
              <div className="flex-1 font-medium">{errorMessage}</div>
            </div>
          )}

          {/* Uploading Progress Stages */}
          {isUploading && selectedFile && (
            <div className="p-4 rounded-xl border border-zinc-800 bg-[#09090b]/60 space-y-3">
              <div className="flex items-center justify-between text-xs text-zinc-300">
                <div className="flex items-center space-x-2 truncate">
                  <FileText className="w-4 h-4 text-indigo-400 shrink-0" />
                  <span className="font-medium truncate">{selectedFile.name}</span>
                </div>
                <span className="text-zinc-400 font-mono shrink-0 ml-2">
                  {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
                </span>
              </div>

              {/* Progress Steps */}
              <div className="grid grid-cols-3 gap-2 pt-1 text-[11px] font-mono">
                <div
                  className={`flex items-center space-x-1.5 p-1.5 rounded-lg border ${
                    uploadStage === 'uploading'
                      ? 'border-indigo-500/40 bg-indigo-950/30 text-indigo-300'
                      : 'border-zinc-800 bg-[#18181c] text-zinc-400'
                  }`}
                >
                  {uploadStage === 'uploading' ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-400" />
                  ) : (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  )}
                  <span className="font-medium">Uploading</span>
                </div>

                <div
                  className={`flex items-center space-x-1.5 p-1.5 rounded-lg border ${
                    uploadStage === 'parsing'
                      ? 'border-indigo-500/40 bg-indigo-950/30 text-indigo-300'
                      : uploadStage === 'indexing'
                      ? 'border-indigo-800/50 bg-indigo-950/20 text-indigo-300'
                      : 'border-zinc-800 bg-[#18181c] text-zinc-400'
                  }`}
                >
                  {uploadStage === 'parsing' ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-400" />
                  ) : uploadStage === 'indexing' ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  ) : (
                    <div className="w-3.5 h-3.5 rounded-full border border-zinc-700" />
                  )}
                  <span className="font-medium">Layout Parsing</span>
                </div>

                <div
                  className={`flex items-center space-x-1.5 p-1.5 rounded-lg border ${
                    uploadStage === 'indexing'
                      ? 'border-indigo-500/40 bg-indigo-950/30 text-indigo-300'
                      : 'border-zinc-800 bg-[#18181c] text-zinc-400'
                  }`}
                >
                  {uploadStage === 'indexing' ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-400" />
                  ) : (
                    <div className="w-3.5 h-3.5 rounded-full border border-zinc-700" />
                  )}
                  <span className="font-medium">Hybrid Indexing</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3.5 bg-[#0e0e11] border-t border-zinc-800 flex items-center justify-end space-x-2">
          <button
            type="button"
            onClick={handleClose}
            disabled={isUploading}
            className="px-4 py-2 text-xs font-medium text-zinc-300 hover:text-white hover:bg-zinc-800 rounded-lg transition disabled:opacity-50 cursor-pointer"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};
