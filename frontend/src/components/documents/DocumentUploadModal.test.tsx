import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DocumentUploadModal } from './DocumentUploadModal';

describe('DocumentUploadModal Component', () => {
  it('does not render modal content when isOpen is false', () => {
    render(
      <DocumentUploadModal
        isOpen={false}
        onClose={vi.fn()}
        onUploadSuccess={vi.fn()}
      />
    );

    expect(screen.queryByText(/upload document/i)).not.toBeInTheDocument();
  });

  it('renders modal dialog with drag-and-drop zone and validation hints when isOpen is true', () => {
    render(
      <DocumentUploadModal
        isOpen={true}
        onClose={vi.fn()}
        onUploadSuccess={vi.fn()}
      />
    );

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText(/upload technical documents/i)).toBeInTheDocument();
    expect(screen.getByText(/drag & drop your pdf here/i)).toBeInTheDocument();
    expect(screen.getByText(/\.pdf files up to 30mb/i)).toBeInTheDocument();
  });

  it('calls onClose when the close or cancel button is clicked', () => {
    const handleClose = vi.fn();
    render(
      <DocumentUploadModal
        isOpen={true}
        onClose={handleClose}
        onUploadSuccess={vi.fn()}
      />
    );

    const closeBtn = screen.getByRole('button', { name: /close modal/i });
    fireEvent.click(closeBtn);
    expect(handleClose).toHaveBeenCalledTimes(1);

    const cancelBtn = screen.getByRole('button', { name: /cancel/i });
    fireEvent.click(cancelBtn);
    expect(handleClose).toHaveBeenCalledTimes(2);
  });

  it('validates file extension and shows error alert for non-PDF files', async () => {
    render(
      <DocumentUploadModal
        isOpen={true}
        onClose={vi.fn()}
        onUploadSuccess={vi.fn()}
      />
    );

    const fileInput = screen.getByTestId('file-input');
    const invalidFile = new File(['dummy content'], 'report.docx', {
      type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    });

    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    await waitFor(() => {
      expect(screen.getByText(/only \.pdf files are supported/i)).toBeInTheDocument();
    });
  });

  it('validates file size and shows error alert when file exceeds 30MB limit', async () => {
    render(
      <DocumentUploadModal
        isOpen={true}
        onClose={vi.fn()}
        onUploadSuccess={vi.fn()}
      />
    );

    const fileInput = screen.getByTestId('file-input');
    // 31 MB file
    const oversizedFile = new File([''], 'large_document.pdf', {
      type: 'application/pdf',
    });
    Object.defineProperty(oversizedFile, 'size', { value: 31 * 1024 * 1024 });

    fireEvent.change(fileInput, { target: { files: [oversizedFile] } });

    await waitFor(() => {
      expect(screen.getByText(/file size exceeds 30mb limit/i)).toBeInTheDocument();
    });
  });

  it('handles dragover and dragleave styles on the dropzone', () => {
    render(
      <DocumentUploadModal
        isOpen={true}
        onClose={vi.fn()}
        onUploadSuccess={vi.fn()}
      />
    );

    const dropzone = screen.getByTestId('upload-dropzone');
    fireEvent.dragEnter(dropzone);
    fireEvent.dragOver(dropzone);
    expect(dropzone).toHaveClass('border-indigo-500');

    fireEvent.dragLeave(dropzone);
    expect(dropzone).not.toHaveClass('border-indigo-500');
  });

  it('submits valid PDF file via drop and triggers onUploadSuccess on completion', async () => {
    const handleSuccess = vi.fn();
    const handleClose = vi.fn();

    // Mock upload implementation passed or api client
    const mockUploadService = vi.fn().mockResolvedValue({
      id: 'doc_new_123',
      filename: 'sample_research.pdf',
      pageCount: 12,
      chunkCount: 30,
      uploadedAt: '2026-10-02T12:00:00Z',
      status: 'ready',
    });

    render(
      <DocumentUploadModal
        isOpen={true}
        onClose={handleClose}
        onUploadSuccess={handleSuccess}
        uploadFn={mockUploadService}
      />
    );

    const dropzone = screen.getByTestId('upload-dropzone');
    const validPdf = new File(['%PDF-1.4 mock content'], 'sample_research.pdf', {
      type: 'application/pdf',
    });

    fireEvent.drop(dropzone, {
      dataTransfer: {
        files: [validPdf],
      },
    });

    await waitFor(() => {
      expect(mockUploadService).toHaveBeenCalledWith(validPdf);
    });

    await waitFor(() => {
      expect(handleSuccess).toHaveBeenCalledWith(
        expect.objectContaining({ id: 'doc_new_123', filename: 'sample_research.pdf' })
      );
      expect(handleClose).toHaveBeenCalled();
    });
  });

  it('displays API error message when upload fails', async () => {
    const failingUpload = vi.fn().mockRejectedValue(new Error('Ingestion pipeline failed'));

    render(
      <DocumentUploadModal
        isOpen={true}
        onClose={vi.fn()}
        onUploadSuccess={vi.fn()}
        uploadFn={failingUpload}
      />
    );

    const fileInput = screen.getByTestId('file-input');
    const validPdf = new File(['%PDF-1.4 mock content'], 'sample_research.pdf', {
      type: 'application/pdf',
    });

    fireEvent.change(fileInput, { target: { files: [validPdf] } });

    await waitFor(() => {
      expect(screen.getByText(/ingestion pipeline failed/i)).toBeInTheDocument();
    });
  });
});
