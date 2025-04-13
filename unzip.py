import zipfile

zip_path = r"C:\Users\Me\Desktop\skin2\dermnet.zip"
extract_to = r"C:\Users\Me\Desktop\skin2\dermnet"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("✅ Unzipped dermnet.zip into /dermnet folder.")
