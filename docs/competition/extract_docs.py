#!/usr/bin/env python3
"""Extract competition docs (DOCX + PDF) to Markdown with images."""

import os, sys, json, re
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
DOCX_PATH = r"F:\zw\Zulu-Walker\docs\competition\H题_车载平衡滚球运动控制系统.docx"
PDF_PATH  = r"F:\zw\Zulu-Walker\docs\competition\H题_车载平衡滚球运动控制系统.pdf"
OUT_DIR   = Path(r"F:\zw\Zulu-Walker\docs\competition\extracted")
OUT_IMAGE_DIR = OUT_DIR / "images"
OUT_MD    = OUT_DIR / "H题_车载平衡滚球运动控制系统.md"

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ── Helper: sanitize filename ───────────────────────────────────────
def sanitize(name):
    return re.sub(r'[\\/:*?"<>|]', '_', name)

# ════════════════════════════════════════════════════════════════════
#  1. EXTRACT DOCX
# ════════════════════════════════════════════════════════════════════
import docx
from docx.oxml.ns import qn
from lxml import etree

doc = docx.Document(DOCX_PATH)

# --- Extract images from DOCX ---------------------------------------
docx_image_map = {}  # blip_embed_id -> filename
img_idx = [0]

def extract_docx_images():
    """Extract all images from the docx's media and build rId->filename map."""
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            rId = rel.rId
            ext = os.path.splitext(rel.target_ref)[1] or ".png"
            img_idx[0] += 1
            fname = f"docx_img_{img_idx[0]:02d}{ext}"
            dest = OUT_IMAGE_DIR / fname
            with open(dest, "wb") as f:
                f.write(rel.target_part.blob)
            docx_image_map[rId] = fname
            print(f"  [DOCX] Extracted {fname} (rId={rId})")

extract_docx_images()

# --- Walk paragraph/run to build markdown ---------------------------
def iter_block_items(doc):
    """Yield (type, element) for paragraphs and tables in document order."""
    from docx.oxml.ns import qn
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn('w:p'):
            yield ('paragraph', docx.text.paragraph.Paragraph(child, doc))
        elif child.tag == qn('w:tbl'):
            yield ('table', docx.table.Table(child, doc))

def paragraph_text_with_images(para, docx_image_map):
    """Return (text, image_fnames) for a paragraph."""
    parts = []
    images = []
    for run in para.runs:
        # Check for drawings / inline images
        drawing_els = run._element.findall(qn('w:drawing'))
        if not drawing_els:
            drawing_els = run._element.findall(qn('w:pict'))
        for drawing in drawing_els:
            # Search for blip embed
            for blip in drawing.iter(qn('a:blip')):
                embed = blip.get(qn('r:embed'))
                if embed and embed in docx_image_map:
                    fname = docx_image_map[embed]
                    if fname not in images:
                        images.append(fname)
        txt = run.text
        if txt:
            parts.append(txt)
    text = ''.join(parts)
    return text, images

def table_to_md(table, docx_image_map):
    """Convert a docx Table to a Markdown string."""
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            cell_text = cell.text.strip().replace('\n', '<br>')
            cells.append(cell_text)
        rows.append(cells)
    if not rows:
        return ""
    # Build MD table
    col_count = max(len(r) for r in rows)
    lines = []
    # header
    header = rows[0]
    while len(header) < col_count:
        header.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * col_count) + " |")
    for row in rows[1:]:
        while len(row) < col_count:
            row.append("")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

# Collect all content
md_parts = []

# Title / header from first paragraphs
for block_type, block in iter_block_items(doc):
    if block_type == 'paragraph':
        text, imgs = paragraph_text_with_images(block, docx_image_map)
        if text.strip():
            md_parts.append(text.strip())
        for img in imgs:
            md_parts.append(f"![{img}](images/{img})")
    elif block_type == 'table':
        md_parts.append(table_to_md(block, docx_image_map))

docx_text = "\n\n".join(md_parts)

# ════════════════════════════════════════════════════════════════════
#  2. EXTRACT PDF
# ════════════════════════════════════════════════════════════════════
import fitz  # PyMuPDF

pdf_doc = fitz.open(PDF_PATH)
pdf_text_pages = []
pdf_image_refs = []

for page_num in range(len(pdf_doc)):
    page = pdf_doc[page_num]
    text = page.get_text("text")
    pdf_text_pages.append(f"## PDF - 第 {page_num+1} 页\n\n{text.strip()}")
    
    # Extract images from page
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        base_image = pdf_doc.extract_image(xref)
        img_bytes = base_image["image"]
        ext = base_image["ext"]
        fname = f"pdf_img_p{page_num+1}_{img_index+1}.{ext}"
        dest = OUT_IMAGE_DIR / fname
        with open(dest, "wb") as f:
            f.write(img_bytes)
        pdf_image_refs.append(f"![{fname}](images/{fname})")
        print(f"  [PDF] Extracted {fname} (page {page_num+1})")

pdf_text = "\n\n".join(pdf_text_pages)
pdf_images_section = "\n\n".join(pdf_image_refs)

# ════════════════════════════════════════════════════════════════════
#  3. COMBINE & WRITE
# ════════════════════════════════════════════════════════════════════

final_md = f"""# H题 车载平衡滚球运动控制系统

> 文档提取自 DOCX + PDF  
> 提取时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 第一部分: Word 文档内容

{docx_text}

---

## 第二部分: PDF 文档内容

{pdf_text}

---

## 第三部分: PDF 嵌入图片

{pdf_images_section}

---

*文档提取完成。共提取 {len(docx_image_map)} 张 DOCX 图片, {len(pdf_image_refs)} 张 PDF 图片。*
"""

OUT_MD.write_text(final_md, encoding="utf-8")
print(f"\n✅ 完成! 输出文件: {OUT_MD}")
print(f"   图片目录: {OUT_IMAGE_DIR}")
print(f"   共 {len(docx_image_map)} 张 DOCX 图片 + {len(pdf_image_refs)} 张 PDF 图片")
