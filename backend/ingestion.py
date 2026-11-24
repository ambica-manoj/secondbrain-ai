"""Validate documents and preserve page boundaries before token-based chunking."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath

from pypdf import PdfReader

from backend.config import MAX_UPLOAD_BYTES


class DocumentError(ValueError):
    """A document cannot be imported."""


@dataclass(frozen=True)
class Page:
    text: str
    number: int  # Markdown uses 0 because it has no page numbers.


def parse_document(filename: str, content: bytes) -> tuple[str, list[Page]]:
    # Treat both Windows and POSIX filenames as untrusted display names.
    name = PurePosixPath(filename.replace('\\', '/')).name
    if not name or len(name) > 255:
        raise DocumentError('Provide a filename of at most 255 characters.')
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in {'.pdf', '.md', '.markdown'}:
        raise DocumentError('Only PDF and Markdown files are supported.')
    if not content:
        raise DocumentError('The document is empty.')
    if len(content) > MAX_UPLOAD_BYTES:
        raise DocumentError('Documents must be at most 10 MiB.')
    if suffix == '.pdf':
        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise DocumentError('Encrypted PDFs are not supported.')
            if len(reader.pages) > 500:
                raise DocumentError('PDFs must contain at most 500 pages.')
            pages = [Page(page.extract_text() or '', i + 1)
                     for i, page in enumerate(reader.pages)]
        except DocumentError:
            raise
        except Exception as error:
            raise DocumentError('Could not read this PDF.') from error
    else:
        try:
            pages = [Page(content.decode('utf-8-sig'), 0)]
        except UnicodeDecodeError as error:
            raise DocumentError('Markdown must use UTF-8 encoding.') from error
    pages = [Page(' '.join(page.text.split()), page.number) for page in pages]
    pages = [page for page in pages if page.text]
    if not pages:
        raise DocumentError('No extractable text. Scanned PDFs need OCR, which is not supported yet.')
    return name, pages


def chunk_pages(pages: list[Page], tokenizer, chunk_tokens: int = 180,
                overlap_tokens: int = 30) -> list[dict]:
    """Keep chunks below MiniLM's input limit, with overlap within each page."""
    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError('Overlap must be smaller than the chunk size.')
    chunks = []
    for page in pages:
        tokens = tokenizer.encode(page.text, add_special_tokens=False)
        for start in range(0, len(tokens), chunk_tokens - overlap_tokens):
            text = tokenizer.decode(tokens[start:start + chunk_tokens], skip_special_tokens=True).strip()
            if text:
                chunks.append({'text': text, 'page_number': page.number})
            if start + chunk_tokens >= len(tokens):
                break
    if len(chunks) > 2000:
        raise DocumentError('Too much extracted text; split this document into smaller files.')
    if not chunks:
        raise DocumentError('The document produced no indexable text.')
    return chunks
