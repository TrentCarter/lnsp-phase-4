from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import time

app = FastAPI()

class FileItem(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int
    modified_date: str

def list_files(directory):
    items = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path):
            items.append(FileItem(name=item, path=full_path, is_dir=True, size=-1, modified_date=""))
        else:
            items.append(FileItem(name=item, path=full_path, is_dir=False, size=os.path.getsize(full_path), modified_date=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(full_path)))))
    return items

@app.get("/api/files/list")
async def list_directory_contents(directory="/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4"):
    if not os.path.exists(directory):
        raise HTTPException(status_code=404, detail="Directory not found")
    return list_files(directory)

@app.get("/api/files/search")
async def search_files(query: str, directory="/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4"):
    if not os.path.exists(directory):
        raise HTTPException(status_code=404, detail="Directory not found")
    results = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if query.lower() in file.lower():
                full_path = os.path.join(root, file)
                results.append(FileItem(name=file, path=full_path, is_dir=False, size=os.path.getsize(full_path), modified_date=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(full_path)))))
    return results

@app.get("/api/files/content")
async def get_file_content(path: str):
    if not os.path.exists(path) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(path, 'r') as file:
        content = file.read()
    return {"content": content}

@app.put("/api/files/edit")
async def edit_file_content(path: str, content: str):
    if not os.path.exists(path) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(path, 'w') as file:
        file.write(content)
    return {"message": "File updated successfully"}
