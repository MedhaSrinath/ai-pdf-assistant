from pypdf import PdfReader

# Make sure filename matches EXACTLY
reader = PdfReader("sample.pdf")

text = ""

# Loop through all pages
for page in reader.pages:
    extracted = page.extract_text()
    if extracted:  # avoid None errors
        text += extracted

# Print first 500 characters
print("\nExtracted Text:\n")
print(text[:500])