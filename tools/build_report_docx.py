"""Sinh REPORT-OUTLINE.docx theo khung HCMUTE (ref/FinaltermProject_Template.pdf).

Khung: A4, font 13, lề trái 3cm, phải/trên/dưới 2cm, dãn dòng 1.3, mỗi Phần sang trang mới.
Nội dung: heading + nhãn [TVx] + dòng Nguồn; các bảng số liệu điền sẵn để không ai chép tay lại số.

Chạy:
    uv run --quiet --with python-docx --no-project python tools/build_report_docx.py
"""

import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "REPORT-OUTLINE.docx"

FONT = "Times New Roman"
MONO = "Consolas"
BODY = Pt(13)
SMALL = Pt(11)
TABLE_SIZE = Pt(12)
GRAY = RGBColor(0x60, 0x60, 0x60)
WARN = RGBColor(0xA0, 0x30, 0x10)

TITLE_VI = (
    "Đánh giá so sánh các phương pháp truyền thống và học sâu trong phân đoạn "
    "văn bản Manga và dịch Manga từ đầu đến cuối (end-to-end)"
)
TITLE_EN = (
    "Comparative Evaluation of Classical and Deep Learning Methods for "
    "Manga Text Segmentation and End-to-End Manga Translation"
)

MEMBERS = [
    ("Nguyễn Quang Huy", "TV4 — inpainting (Spec 003 làn A)"),
    ("Hà Duy Anh", "TV2 — đánh giá định tính (Spec 003 làn B)"),
    ("", "TV3 — OCR / dịch / render (Spec 004)"),
    ("", "TV1 — baseline + benchmark + web (Spec 001/002/005)"),
]

ABBREV = [
    ("DL", "Deep Learning (học sâu)"),
    ("GT", "Ground Truth (mask chuẩn)"),
    ("IoU", "Intersection over Union"),
    ("MSER", "Maximally Stable Extremal Regions"),
    ("OCR", "Optical Character Recognition"),
    ("TELEA", "Thuật toán inpainting của Telea (OpenCV INPAINT_TELEA)"),
    ("NS", "Navier–Stokes (OpenCV INPAINT_NS)"),
    ("YOLO", "You Only Look Once"),
    ("DBNet", "Differentiable Binarization Network"),
    ("LLM", "Large Language Model"),
    ("E2E", "End-to-End"),
    ("CLI", "Command-Line Interface"),
    ("LOFO", "Leave-One-Fold-Out"),
]

FIGURES = [
    ("1", "Sơ đồ khối pipeline tổng thể", "TV1", "README.md"),
    ("2", "Sơ đồ nhánh baseline cổ điển", "TV1", "src/manga_text_seg/"),
    ("3", "Kiến trúc U-Net + ResNet34", "TV4", "notebooks/deliverables/manga-text-segmentation/"),
    ("4", "Kiến trúc comic-text-detector", "TV2", "notebooks/deliverables/comic-text-detector/"),
    ("5", "Kiến trúc UNet++ + EfficientNetV2", "TV3", "notebooks/deliverables/unetpp-efficientnetv2/"),
    ("6", "Biểu đồ so sánh IoU/F1 theo phương pháp", "TV1", "deliverables/spec002/charts/accuracy-by-method.png"),
    ("7", "Mask dự đoán của các phương pháp trên cùng một trang", "TV2", "outputs/inpainting/spec003-main/boards/*/highest_iou_f1/"),
    ("8", "Board so sánh xóa và inpainting", "TV2", "outputs/inpainting/spec003-main/boards/"),
    ("9", "Ca thành công điển hình", "TV2", "board highest_iou_f1"),
    ("10", "Ca thất bại điển hình (EvaLady/006)", "TV3", "board lowest_iou_f1"),
    ("11", "Kết quả OCR → dịch → render", "TV3", "outputs/translation/spec004-vilao/"),
    ("12", "Giao diện web demo", "TV1", "web-ui/index.html"),
]

TABLES = [
    ("1", "Tiêu chí đánh giá và mục đích đo lường", "TV1", "đề cương, Bảng 2"),
    ("2", "Tham số và cấu hình từng phương pháp", "CHUNG", "deliverables/spec002/run.json"),
    ("3", "Kết quả benchmark định lượng (IoU/P/R/F1/thời gian)", "TV1", "deliverables/spec002/comparison.csv"),
    ("4", "Thời gian inpainting theo phương pháp × thuật toán", "TV4", "outputs/inpainting/spec003-main/performance.json"),
    ("5", "Năm trang được chọn cho pipeline demo", "TV2", "outputs/inpainting/spec003-main/selection.json"),
    ("6", "Cấu hình OCR / dịch / render", "TV3", "outputs/translation/*/run.json"),
    ("7", "So sánh trực quan inpainting theo năm tiêu chí", "TV2", "đề cương, Bảng 3"),
    ("8", "Mức độ đóng góp của các thành viên", "TV1", "trang bìa"),
]

BENCHMARK = [
    ("Otsu", "0.0081", "0.0088", "0.1848", "0.0154", "0.0022", "CPU"),
    ("Adaptive Threshold", "0.0711", "0.0753", "0.8744", "0.1261", "0.0090", "CPU"),
    ("MSER", "0.0435", "0.0452", "0.9549", "0.0761", "1.2676", "CPU"),
    ("Edges", "0.0350", "0.0356", "0.9857", "0.0630", "0.0248", "CPU"),
    ("Morphology", "0.0502", "0.0553", "0.4358", "0.0888", "0.0054", "CPU"),
    ("Connected Components", "0.0563", "0.0599", "0.6402", "0.1001", "0.4173", "CPU"),
    ("comic-text-detector", "0.0372", "0.1142", "0.0603", "0.0646", "0.3814", "Tesla T4"),
    ("UNet++ EfficientNetV2", "0.1375", "0.1582", "0.6944", "0.2240", "0.2790", "Tesla T4"),
    ("Manga-Text-Segmentation", "không chấm", "—", "—", "—", "—", "—"),
]

INPAINT_TIME = [
    ("classical_baseline", "1.1375", "1.0251"),
    ("comic_text_detector", "0.0746", "0.0589"),
    ("unetpp_efficientnetv2", "0.5324", "7.8853"),
]

DEMO_PAGES = [
    ("Akuhamu/000", "0.6318", "0.7744", "16.729", "91.855"),
    ("AisazuNihaIrarenai/001", "0.5866", "0.7395", "30.731", "52.290"),
    ("BEMADER_P/000", "0.5732", "0.7287", "72.846", "36.930"),
    ("BakuretsuKungFuGirl/000", "0.5415", "0.7026", "50.697", "108.972"),
    ("Hamlet/000", "0.5270", "0.6903", "72.172", "58.612"),
]

RUNS = [
    ("spec004-vilao", "unetpp_efficientnetv2", "vi", "vilao / deepseek-v4.1-flash", "28"),
    ("spec004-live", "unetpp_efficientnetv2", "vi", "openrouter / anthropic/claude-sonnet-5.5", "1"),
    ("spec004-real", "classical_baseline", "vi", "anthropic / claude-sonnet-5-5", "1"),
    ("smoke-dl", "comic_text_detector", "vi", "openai / gpt-4o-mini", "1"),
    ("smoke-translate", "classical_baseline", "vi", "openai / gpt-4o-mini", "1"),
]

REFERENCES = [
    "Sachdeva, R., Zisserman, A. (2024). The Manga Whisperer: Automatically Generating Transcriptions for Comics. CVPR 2024. — Phần 2.7.",
    "Hinami, R., Ishiwatari, S., Yasuda, K., Matsui, Y. (2021). Towards Fully Automated Manga Translation. AAAI 2021. — Phần 2.7, 4.6.",
    "Del Gobbo, J., Matuk Herrera, R. (2020). Unconstrained Text Detection in Manga: A New Dataset and Baseline. ECCVW 2020. — Phần 2.3, 4.1.",
    "dmMaze / kha-white. comic-text-detector: Manga & Comic text detection. GitHub (fork Ajatt-Tools/comic_text_detector). — Phần 2.4.",
    "ContemporaryCat (2025). Manga-Text-Segmentation (UNet++ EfficientNetV2 pretrained). GitHub. — Phần 2.5.",
    "kha-white. manga-ocr: Japanese OCR for manga. GitHub. — Phần 2.7, 4.6.",
    "zyddnys. manga-image-translator. GitHub — nguồn module render text dùng trong demo. — Phần 3.6.",
    "Gonzalez, R. C., Woods, R. E. Digital Image Processing. — Phần 2.2.",
]

SCHEDULE = [
    ("Ngày 1–2", "Phần 2 (mỗi người viết phương pháp của mình)", "cả bốn, song song"),
    ("Ngày 2–3", "Phần 3.4 [TV4], 3.5 [TV2], 3.6 [TV3]", "TV4, TV2, TV3"),
    ("Ngày 3–4", "Phần 4.4 mỗi người một mục; 4.5.2 board", "TV4, TV2, TV3"),
    ("Ngày 4–5", "Phần 4.6 [TV3]; Phần 1 + 4.1 + 4.3 + 4.7 [TV1]", "TV3, TV1"),
    ("Ngày 5", "Phần 5 — mỗi người gửi gạch đầu dòng, TV1 gộp", "cả bốn"),
    ("Ngày 6", "Mục lục, danh mục, danh sách hình/bảng, định dạng, rà mục A", "TV1"),
    ("Ngày 6", "PPT slides", "cả bốn, TV1 gộp"),
]

SHELL = """uv sync
manga-text-seg run --config configs/segmentation.json --method adaptive
manga-text-seg status --config configs/dl.json
manga-text-seg admit --config configs/dl.json --run <run_id> --method <name> <hand-off>/
manga-text-seg inpaint --config configs/inpainting.json --run spec003-main
manga-text-seg boards --config configs/inpainting.json --run spec003-main
manga-text-seg qualitative --config configs/inpainting.json --run spec003-main
manga-text-seg translate --config configs/translation.json --run spec004-vilao
manga-text-seg serve
pytest -o addopts='' -q"""


# ---------------------------------------------------------------- helpers


def _font(run, name=FONT, size=BODY, bold=None, italic=None, color=None):
    run.font.name = name
    run.font.size = size
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)


def para(doc, text="", *, align=None, size=BODY, bold=False, italic=False,
         color=None, after=Pt(6), mono=False, indent=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = after
    if indent is not None:
        p.paragraph_format.left_indent = indent
    if text:
        _font(p.add_run(text), MONO if mono else FONT, size, bold, italic, color)
    return p


def heading(doc, text, level, *, page_break=False):
    h = doc.add_heading(level=level)
    h.paragraph_format.page_break_before = page_break
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.line_spacing = 1.3
    _font(h.add_run(text), FONT, {1: Pt(15), 2: Pt(14)}.get(level, BODY),
          bold=True, italic=False, color=RGBColor(0, 0, 0))
    return h


def section(doc, text, level, who=None, source=None, note=None):
    heading(doc, text, level)
    if who or source:
        para(doc, f"Người viết: {who or '—'}   ·   Nguồn: {source or '—'}",
             size=SMALL, italic=True, color=GRAY, after=Pt(2))
    if note:
        para(doc, f"⚠ {note}", size=SMALL, italic=True, color=WARN, after=Pt(6))


def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    for i, text in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        _font(p.add_run(text), FONT, TABLE_SIZE, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            _font(p.add_run(str(value)), FONT, TABLE_SIZE)
    para(doc, "", after=Pt(6))
    return t


def cover(doc, *, english, page_break):
    p = para(doc, "HCMC UNIVERSITY OF TECHNOLOGY AND ENGINEERING" if english
             else "TRƯỜNG ĐH CÔNG NGHỆ KỸ THUẬT TP. HỒ CHÍ MINH",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, after=Pt(0))
    p.paragraph_format.page_break_before = page_break
    para(doc, "FACULTY OF INFORMATION TECHNOLOGY" if english
         else "KHOA CÔNG NGHỆ THÔNG TIN",
         align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, after=Pt(36))
    para(doc, "FINAL TERM PROJECT" if english else "TIỂU LUẬN CUỐI KỲ",
         align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, after=Pt(24))
    para(doc, "Course name: Digital Image Processing" if english
         else "Môn học: Xử lý ảnh số",
         align=WD_ALIGN_PARAGRAPH.CENTER, after=Pt(12))
    para(doc, f"TITLE OF PROJECT: {TITLE_EN}" if english
         else f"Tên tiểu luận: {TITLE_VI}",
         align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, after=Pt(36))
    para(doc, "Lecturer name: Assoc. Prof. Hoang Van Dung" if english
         else "Giảng viên: PGS.TS. Hoàng Văn Dũng",
         align=WD_ALIGN_PARAGRAPH.CENTER, after=Pt(24))
    para(doc, "List of members" if english else "Danh sách sinh viên thực hiện",
         align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, after=Pt(6))
    # Mã số SV và % đóng góp để trống — repo không có dữ liệu này, người dùng tự điền.
    table(doc,
          (("Student ID", "Student name", "Contribution (%)") if english
           else ("Mã số SV", "Họ và tên", "Mức độ đóng góp (%)")),
          [("", name, "") for name, _ in MEMBERS])
    if not english:
        for name, lane in MEMBERS:
            para(doc, f"{name or '(chưa có tên)'} — {lane}",
                 size=SMALL, italic=True, color=GRAY, after=Pt(0))
    para(doc, "", after=Pt(24))
    para(doc, "Ho Chi Minh City, 10/2026" if english
         else "TP. Hồ Chí Minh, tháng 10 năm 2026",
         align=WD_ALIGN_PARAGRAPH.CENTER)


# ---------------------------------------------------------------- build


def build(force=False):
    # ponytail: the committed REPORT-OUTLINE.docx holds hand-typed names and IDs,
    # so a bare re-run would destroy real work. Refuse unless --force.
    if OUT.exists() and not force:
        raise SystemExit(
            f"{OUT.name} da ton tai — chay lai se ghi de ten/ma SV nguoi dung da dien.\n"
            f"Muon ghi de that su: python tools/build_report_docx.py --force"
        )
    doc = Document()

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = BODY
    normal.paragraph_format.line_spacing = 1.3
    normal.paragraph_format.space_after = Pt(6)
    rfonts = normal.element.get_or_add_rPr().get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), FONT)

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin, sec.right_margin = Cm(3), Cm(2)
    sec.top_margin, sec.bottom_margin = Cm(2), Cm(2)

    cover(doc, english=True, page_break=False)
    cover(doc, english=False, page_break=True)

    heading(doc, "Mục lục", 1, page_break=True)
    para(doc, "Sinh tự động sau khi cả bốn người viết xong "
              "(Word: References → Table of Contents). Viết sau cùng, không viết trước.",
         size=SMALL, italic=True, color=GRAY)

    heading(doc, "Danh mục các cụm từ / từ viết tắt", 1, page_break=True)
    table(doc, ("Viết tắt", "Nghĩa đầy đủ"), ABBREV)

    heading(doc, "Danh sách hình vẽ", 1, page_break=True)
    table(doc, ("#", "Tên hình", "Người cấp", "Nguồn"), FIGURES)

    heading(doc, "Danh sách bảng biểu", 1, page_break=True)
    table(doc, ("#", "Tên bảng", "Người cấp", "Nguồn"), TABLES)

    # ---- Phần 1
    heading(doc, "PHẦN 1. MỞ ĐẦU (MÔ TẢ VỀ PROJECT)", 1, page_break=True)
    para(doc, "Chuyển thể từ đề cương đã nộp, viết lại theo văn phong báo cáo "
              "(không copy nguyên văn) và cập nhật theo thực tế đã làm.",
         size=SMALL, italic=True, color=GRAY)
    section(doc, "1.1. Phát biểu bài toán", 2, "[TV1]", "đề cương mục 1")
    section(doc, "1.2. Mục đích, yêu cầu cần thực hiện", 2, "[TV1]", "đề cương mục 2",
            note="Năm mục tiêu, mỗi mục ghi rõ đã đạt hay chưa. Mục tiêu 2 chỉ ĐẠT MỘT PHẦN — "
                 "Manga-Text-Segmentation không chấm được (Phụ lục A.1).")
    section(doc, "1.3. Phạm vi và đối tượng", 2, "[TV1]",
            "đề cương mục 3; README.md; ONBOARDING.md")

    # ---- Phần 2
    heading(doc, "PHẦN 2. CƠ SỞ LÝ THUYẾT", 1, page_break=True)
    section(doc, "2.1. Công cụ, môi trường, ngôn ngữ", 2, "[TV1]",
            "README.md; deliverables/spec002/run.json; "
            "outputs/inpainting/spec003-main/performance.json")
    section(doc, "2.2. Kỹ thuật xử lý ảnh cổ điển dùng trong baseline", 2, "[TV1]",
            "src/manga_text_seg/ (Spec 001); outputs/segmentation/default/metrics.csv; "
            "Gonzalez & Woods",
            note="Mỗi kỹ thuật khoảng nửa trang: xám hoá, Otsu, adaptive threshold, MSER, "
                 "phát hiện biên, hình thái học + connected components.")
    section(doc, "2.3. Manga-Text-Segmentation — U-Net + ResNet34", 2, "[TV4]",
            "notebooks/deliverables/manga-text-segmentation/README.md; benchmark/availability.json",
            note="Kết mục bằng câu chuyển ý: phương pháp đã vào quy trình nhưng KHÔNG chấm được điểm "
                 "(Phụ lục A.1, 4.4.1).")
    section(doc, "2.4. comic-text-detector — YOLOv5 + DBNet + U-Net", 2, "[TV2]",
            "notebooks/deliverables/comic-text-detector/; đề cương mục 4")
    section(doc, "2.5. UNet++ EfficientNetV2 (ContemporaryCat)", 2, "[TV3]",
            "notebooks/deliverables/unetpp-efficientnetv2/; đề cương mục 4",
            note="Nói trước cảnh báo nhiễm bẩn dữ liệu ở đây; phân tích ở 4.4.3 và Phụ lục A.2.")
    section(doc, "2.6. Inpainting: TELEA và Navier–Stokes", 2, "[TV4]",
            "src/manga_text_seg/inpaint.py; configs/inpainting.json")
    section(doc, "2.7. OCR, dịch máy, render chữ", 2, "[TV3]",
            "src/manga_text_seg/translation/; outputs/translation/*/run.json; đề cương mục 5.5",
            note="Ghi rõ ba bước này là demo minh hoạ, KHÔNG thuộc phạm vi đánh giá kỹ thuật chính.")

    # ---- Phần 3
    heading(doc, "PHẦN 3. PHÂN TÍCH, THIẾT KẾ GIẢI PHÁP", 1, page_break=True)
    section(doc, "3.1. Sơ đồ khối tổng thể", 2, "[TV1]",
            "README.md (khối pipeline); specs/002-*/spec.md",
            note="Chuỗi: Dataset → Text segmentation → Text removal & inpainting → Japanese OCR "
                 "→ Translation → Text rendering → Summary & web demo. Trên sơ đồ phải phân biệt "
                 "phạm vi benchmark chính với phần demo minh hoạ.")
    section(doc, "3.2. Thiết kế baseline cổ điển", 2, "[TV1]",
            "src/manga_text_seg/ (Spec 001); outputs/segmentation/default/metrics.csv",
            note="Sáu biến thể: otsu, adaptive, mser, edges, morphology, components.")
    section(doc, "3.3. Hợp đồng mask và quy trình kết nạp kết quả", 2, "[TV1]",
            "ONBOARDING.md §2, §7, §8; benchmark/availability.json; deliverables/spec002/run.json",
            note="Phần thiết kế quan trọng nhất về kỹ thuật — viết kỹ. Mask một kênh uint8, "
                 "kích thước (1170, 1654), giá trị {0, 255}. page_list_identity trích nguyên văn: "
                 "b52aa60ded6c5c2cca115dce4335295d6795b0601b9cfa8324c675fb4cab88ef. "
                 "Prediction mask là mask DUY NHẤT được mở.")
    section(doc, "3.4. Thiết kế quy trình xóa và phục hồi", 2, "[TV4]",
            "configs/inpainting.json; outputs/inpainting/spec003-main/run.json")
    section(doc, "3.5. Thiết kế lựa chọn trang và board so sánh", 2, "[TV2]",
            "src/manga_text_seg/boards.py, selection.py, qualitative.py; "
            "outputs/inpainting/spec003-main/selection.json",
            note="Bốn quy tắc × 5 trang: highest_iou_f1, lowest_iou_f1, most_false_positives, "
                 "most_false_negatives. Quy tắc thứ năm manual_artifact_flags hiện TRỐNG "
                 "(shortfall: 5) — xem Phụ lục A.4.")
    section(doc, "3.6. Thiết kế trích vùng, OCR, dịch, render", 2, "[TV3]",
            "src/manga_text_seg/translation/; outputs/translation/spec004-live/run.json",
            note="Ghi provider/model nhưng TUYỆT ĐỐI không ghi giá trị khoá API (Phụ lục A.5).")
    section(doc, "3.7. Thiết kế web demo", 2, "[TV1]",
            "src/manga_text_seg/web/demo.py, schemas.py, security.py; web-ui/index.html; "
            "stitch_manga_translation_pipeline_workbench/",
            note="Nói thẳng: bản demo chạy 3 route, KHÔNG phải 17 endpoint trong "
                 "specs/005-web-e2e-reporting/plan.md. Phần còn lại là hướng phát triển (mục 5.2).")

    # ---- Phần 4
    heading(doc, "PHẦN 4. THỰC NGHIỆM, ĐÁNH GIÁ, PHÂN TÍCH KẾT QUẢ", 1, page_break=True)
    section(doc, "4.1. Dữ liệu và giao thức đánh giá", 2, "[TV1]",
            "benchmark/; deliverables/spec002/run.json; đề cương mục 5.1, 5.3",
            note="390 ảnh / 45 manga. KHÔNG dùng Pixel Accuracy. KHÔNG dùng PSNR/SSIM cho inpainting. "
                 "Thư mục data/no-need-to-read/ nằm ngoài phạm vi, không được đọc (Phụ lục A.6).")
    section(doc, "4.2. Cấu hình phần cứng và tham số", 2, "[CHUNG]",
            "deliverables/spec002/run.json; outputs/inpainting/spec003-main/performance.json",
            note="Mỗi người ghi cấu hình phần mình chạy. Ghi cả sha256 của checkpoint đã dùng.")

    section(doc, "4.3. Kết quả benchmark định lượng", 2, "[TV1] (bảng), [TV4][TV2][TV3] (đọc số của mình)",
            "deliverables/spec002/comparison.csv, comparison.json, run.json",
            note="Phải ghi kèm ngay dưới bảng: độ lệch chuẩn từng chỉ số (IoU của UNet++ có std 0.1219 "
                 "— phân tán rất lớn); cột công bố nhiễm bẩn; dòng Manga-Text-Segmentation ghi rõ lý do "
                 "không chấm (Phụ lục A.1).")
    table(doc,
          ("Phương pháp", "IoU", "Precision", "Recall", "F1", "Thời gian/ảnh (s)", "Thiết bị"),
          BENCHMARK)
    para(doc, "Hình: deliverables/spec002/charts/accuracy-by-method.png",
         size=SMALL, italic=True, color=GRAY)

    section(doc, "4.4. Phân tích từng phương pháp", 2)
    section(doc, "4.4.1. Manga-Text-Segmentation", 3, "[TV4]",
            "benchmark/availability.json; REISSUE.md; notebooks/deliverables/manga-text-segmentation/",
            note="Thiếu 5 checkpoint fold (~344 MB mỗi file), không xác nhận được revision đã ghim "
                 "de8f148c78978d70ad0e0ae3242566da6d70f3a5 → chọn REISSUE Path 1, trạng thái "
                 "unavailable. Đây là kết quả âm hợp lệ, báo cáo trung thực (Phụ lục A.1). "
                 "Không suy diễn, không lấy số từ nguồn khác, không bỏ dòng khỏi bảng.")
    section(doc, "4.4.2. comic-text-detector", 3, "[TV2]",
            "deliverables/spec002/comparison.csv; deliverables/spec002/comic-text-detector/",
            note="IoU 0.0372 / F1 0.0646 — thấp hơn cả baseline cổ điển tốt nhất. "
                 "Công bố nhiễm bẩn: contamination extent unknown.")
    section(doc, "4.4.3. UNet++ EfficientNetV2", 3, "[TV3]",
            "deliverables/spec002/comparison.csv; deliverables/spec002/unetpp-efficientnetv2/; "
            "outputs/inpainting/spec003-main/selection.json",
            note="IoU 0.1375 / F1 0.2240 — cao nhất trong các phương pháp chấm được, NHƯNG nhiễm bẩn "
                 "hoàn toàn: checkpoint huấn luyện trên toàn bộ 45 cuốn sách của tập ground truth, "
                 "nên mọi điểm số là điểm trên tập huấn luyện. Câu này phải nằm ngay cạnh bảng kết quả, "
                 "không để ở phụ lục (Phụ lục A.2). Ca thất bại điển hình: EvaLady/006, IoU = 0, "
                 "310.000 pixel dương tính giả.")
    section(doc, "4.4.4. Baseline cổ điển", 3, "[TV1]",
            "deliverables/spec002/comparison.csv; outputs/segmentation/default/metrics.csv",
            note="Adaptive Threshold là biến thể tốt nhất và được chọn làm đại diện. "
                 "Nêu bất đối xứng phần cứng khi so thời gian: baseline chạy CPU, model DL chạy GPU.")

    section(doc, "4.5. Kết quả xóa và phục hồi", 2)
    section(doc, "4.5.1. Thời gian inpainting", 3, "[TV4]",
            "outputs/inpainting/spec003-main/performance.json",
            note="390 mẫu mỗi tổ hợp, 0 lỗi, CPU Intel64, Python 3.12.13. Phân tích bắt buộc: NS trên "
                 "mask của UNet++ chậm ~14.8 lần so với TELEA trên cùng mask — quan hệ thuật toán × "
                 "mật độ mask, không phải thuật toán chậm hơn.")
    table(doc, ("Phương pháp", "TELEA (s/ảnh)", "NS (s/ảnh)"), INPAINT_TIME)

    section(doc, "4.5.2. Đánh giá định tính", 3, "[TV2]",
            "outputs/inpainting/spec003-main/boards/; src/manga_text_seg/boards.py, qualitative.py",
            note="Nhận xét theo đúng năm tiêu chí của đề cương: mức độ xóa hoàn toàn văn bản; mức độ "
                 "bỏ sót; mức độ xóa nhầm vùng nền; độ tự nhiên vùng sau inpainting; artifact/viền/nhiễu. "
                 "60 ảnh board (3 phương pháp × 4 quy tắc × 5 trang). Board KHÔNG được chứa ground truth "
                 "(Phụ lục A.3).")
    section(doc, "4.5.3. Năm trang được chọn cho pipeline demo", 3, "[TV2]",
            "outputs/inpainting/spec003-main/selection.json",
            note="Quy tắc highest_iou_f1, phương pháp unetpp_efficientnetv2.")
    table(doc, ("Trang", "IoU", "F1", "FP", "FN"), DEMO_PAGES)

    section(doc, "4.6. Kết quả OCR, dịch, render", 3, "[TV3]",
            "outputs/translation/*/run.json, errors.json; outputs/translation/spec004-vilao/",
            note="Ghi TÊN biến môi trường chứa khoá (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, "
                 "MANGA_TRANSLATE_API_KEY) — tuyệt đối không ghi giá trị khoá (Phụ lục A.5). "
                 "Ghi nhận các vùng/ảnh lỗi, không che.")
    table(doc, ("Run", "Phương pháp phân đoạn", "Đích", "Provider / Model", "Số trang"), RUNS)

    section(doc, "4.7. Giao diện phần mềm và hướng dẫn thực thi", 2, "[TV1]",
            "README.md; src/manga_text_seg/cli.py; web-ui/index.html",
            note="Ảnh chụp giao diện web demo + hướng dẫn chạy từng bước. Ghi rõ demo chạy trên máy "
                 "người trình bày, mở browser tại loopback.")
    for line in SHELL.splitlines():
        para(doc, line, size=Pt(11), mono=True, after=Pt(0))

    # ---- Phần 5
    heading(doc, "PHẦN 5. KẾT LUẬN", 1, page_break=True)
    section(doc, "5.1. Đánh giá những kết quả đã thực hiện được", 2, "[CHUNG], TV1 chốt",
            "—",
            note="Mỗi người 2–3 gạch đầu dòng cho phần mình, TV1 gộp và viết đoạn mở/đoạn kết. "
                 "Kết luận kỹ thuật trung thực cần có: adaptive vượt comic-text-detector về IoU/F1 "
                 "nhưng thua UNet++ — mà UNet++ nhiễm bẩn hoàn toàn nên không thể dùng để khẳng định "
                 "ưu thế của học sâu.")
    section(doc, "5.2. Định hướng phát triển", 2, "[CHUNG]", "—")

    heading(doc, "TÀI LIỆU THAM KHẢO", 1, page_break=True)
    para(doc, "Giữ nguyên thứ tự tám tài liệu từ đề cương; mỗi tài liệu ghi rõ vị trí tham chiếu "
              "trong báo cáo.", size=SMALL, italic=True, color=GRAY)
    for i, ref in enumerate(REFERENCES, 1):
        para(doc, f"[{i}] {ref}", after=Pt(4))

    # ---- Phụ lục A: việc PHẢI nói thật (đọc trước khi viết bất cứ phần nào)
    heading(doc, "PHỤ LỤC A. VIỆC PHẢI NÓI THẬT — KHÔNG ĐƯỢC CHE, KHÔNG ĐƯỢC BỊA",
            1, page_break=True)
    para(doc, "Đọc mục này trước khi viết bất cứ phần nào. Bốn điểm dưới đây là điểm mạnh của báo cáo "
              "nếu trình bày trung thực, và là gian lận học thuật nếu bị làm mờ. Phụ lục này KHÔNG in "
              "vào báo cáo nộp — nó là danh sách rà soát.",
         size=SMALL, italic=True, color=GRAY)

    section(doc, "A.1. Manga-Text-Segmentation không có điểm", 2, "[TV4]",
            "benchmark/availability.json; REISSUE.md",
            note="Phương pháp KHÔNG chấm được, lý do là thiếu checkpoint. KHÔNG được suy diễn từ kết quả "
                 "Colab thăm dò, KHÔNG được lấy số từ nguồn khác, KHÔNG được bỏ dòng này khỏi bảng. "
                 "Ghi trạng thái unavailable và lý do.")
    section(doc, "A.2. UNet++ EfficientNetV2 nhiễm bẩn hoàn toàn", 2, "[TV3]",
            "deliverables/spec002/unetpp-efficientnetv2/",
            note="Đây là phương pháp có điểm cao nhất VÀ là phương pháp duy nhất có công bố nhiễm bẩn ở "
                 "mức nặng nhất: checkpoint được huấn luyện trên toàn bộ 45 cuốn sách của tập ground "
                 "truth. Mọi điểm số của nó LÀ điểm trên tập huấn luyện. Phải viết câu này ngay cạnh "
                 "bảng kết quả, không được để ở phụ lục.")
    section(doc, "A.3. Không có ground truth trong bất kỳ hình nào", 2, "[TV2][TV3]",
            "outputs/inpainting/spec003-main/boards/",
            note="Mask dự đoán là mask duy nhất được mở. Board, hình so sánh, ảnh minh họa KHÔNG được "
                 "chứa ground truth. Không có chế độ GT-assisted.")
    section(doc, "A.4. Quy tắc manual_artifact_flags trả về rỗng", 2, "[TV2]",
            "outputs/inpainting/spec003-main/selection.json",
            note="selection.json ghi manual_artifact_flags với shortfall: 5 — tức KHÔNG có trang nào "
                 "được gắn cờ thủ công. Nếu phần đánh giá định tính có nhắc tới quy tắc này thì phải nói "
                 "rõ nó chưa được thực hiện, KHÔNG được viết như thể đã có kết quả.")
    section(doc, "A.5. Không ghi giá trị khoá API", 2, "[TV3]",
            "outputs/translation/*/run.json",
            note="Run record ghi TÊN biến môi trường (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, "
                 "MANGA_TRANSLATE_API_KEY). Giá trị khoá KHÔNG được xuất hiện ở bất kỳ đâu trong báo "
                 "cáo, hình, hay phụ lục.")
    section(doc, "A.6. Không nêu data/no-need-to-read/ như một phần dữ liệu", 2, "[TV1]", "—",
            note="Thư mục này nằm ngoài phạm vi xử lý. Chỉ nhắc một câu ở Phần 4.1 nếu cần, và nói rõ "
                 "nó KHÔNG được đọc.")

    # ---- Phụ lục B: nhịp làm việc
    heading(doc, "PHỤ LỤC B. NHỊP LÀM VIỆC ĐỀ XUẤT", 1, page_break=True)
    table(doc, ("Mốc", "Việc", "Ai"), SCHEDULE)
    para(doc, "Chốt trước khi nộp: rà lại toàn báo cáo xem có câu nào vô tình khẳng định ưu thế của học "
              "sâu dựa trên điểm của UNet++ mà chưa kèm cảnh báo nhiễm bẩn. Đó là lỗi dễ mắc nhất và là "
              "lỗi nặng nhất.", size=SMALL, bold=True, color=WARN)

    doc.core_properties.title = "Báo cáo cuối kỳ — Xử lý ảnh số — Nhóm 9"
    doc.core_properties.comments = (
        "Khung sinh từ tools/build_report_docx.py. Nội dung chi tiết đọc ở REPORT-OUTLINE.md."
    )
    doc.save(OUT)
    print(f"wrote {OUT}")


def verify():
    """Kiểm tra khung còn đúng sau khi sinh: khổ giấy, lề, số bảng, số lần sang trang."""
    d = Document(OUT)
    s = d.sections[0]
    assert (round(s.page_width.cm, 1), round(s.page_height.cm, 1)) == (21.0, 29.7), "not A4"
    assert round(s.left_margin.cm, 1) == 3.0, "left margin must be 3cm"
    assert (round(s.right_margin.cm, 1), round(s.top_margin.cm, 1),
            round(s.bottom_margin.cm, 1)) == (2.0, 2.0, 2.0), "right/top/bottom margin must be 2cm"
    assert d.styles["Normal"].font.size == BODY, "body font must be 13pt"
    breaks = sum(1 for p in d.paragraphs if p.paragraph_format.page_break_before)
    assert breaks == 13, f"page break count changed: {breaks}"
    assert len(d.tables) == 10, f"table count changed: {len(d.tables)}"
    print(f"ok: A4, margins 3/2/2/2, 13pt, {breaks} page breaks, {len(d.tables)} tables")


if __name__ == "__main__":
    build(force="--force" in sys.argv)
    verify()
