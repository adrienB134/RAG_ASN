"""organize the docs in subfolders to make it easier to track and avoid rate limits"""
import os
import shutil


folder = os.listdir(f"ASN/lettres_de_suivi/")
a = 0
while folder:
    for i in range(1, 40):
        try:
            file = folder[0]
        except IndexError:
            break
        if not os.path.exists(f"ASN/lettres_de_suivi/{str(a)}"):
            os.mkdir(f"ASN/lettres_de_suivi/{str(a)}")
        dest = f"ASN/lettres_de_suivi/{str(a)}"
        current_file = f"ASN/lettres_de_suivi/{file}"
        shutil.move(current_file, dest)
        folder.pop(0)
    a = a + 1
    print(a)

print("Done")
