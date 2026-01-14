import argparse
from pathlib import Path
from pdf2image import convert_from_path

def main():
    parser = argparse.ArgumentParser(description="Convert PDF to images (remove first & last page)")
    parser.add_argument("pdf_path", help="Path to PDF file")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return

    output_dir = pdf_path.parent
    base_name = pdf_path.stem

    print(f"📄 Reading PDF: {pdf_path.name}")

    # Convert all pages
    pages = convert_from_path(pdf_path, dpi=300)

    if len(pages) <= 2:
        print("❌ PDF has 2 or fewer pages — nothing to save after removing first & last")
        return

    # Remove first and last page
    pages = pages[1:-1]

    print(f"🖼 Saving {len(pages)} images...")

    for i, page in enumerate(pages, start=1):
        img_name = f"{base_name}_img{i:03d}.jpg"
        img_path = output_dir / img_name
        page.save(img_path, "JPEG")
        print(f"✔ Saved: {img_name}")

    print("✅ Done")

if __name__ == "__main__":
    main()
