with open('test data/v1_annotate.html', 'r', encoding='utf-8') as f:
    html = f.read()

marker = "btnSaveBeepEl.addEventListener('click'"
idx = html.find(marker)
chunk = html[idx:idx+800]
with open('tmp/debug_out.txt', 'w', encoding='utf-8') as f:
    f.write(chunk)
print("written")
