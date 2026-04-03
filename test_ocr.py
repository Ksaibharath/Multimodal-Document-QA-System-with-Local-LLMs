import easyocr

print("Starting EasyOCR model download/load...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR loaded successfully.")