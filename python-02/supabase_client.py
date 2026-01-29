"""
Supabase Storage Client

Handles file uploads to Supabase Storage.
"""

import os
from pathlib import Path
from typing import Optional
import httpx


class SupabaseUploader:
    """Handles file uploads to Supabase Storage."""
    
    def __init__(self):
        self.url = os.environ.get("SUPABASE_URL")
        self.key = os.environ.get("SUPABASE_SERVICE_KEY")
        
        if not self.url or not self.key:
            print("WARNING: Supabase credentials not set. Uploads will be skipped.")
            self.enabled = False
        else:
            self.enabled = True
            self.storage_url = f"{self.url}/storage/v1/object"
            self.headers = {
                "apikey": self.key,
                "Authorization": f"Bearer {self.key}",
            }
    
    def upload_file(
        self,
        bucket: str,
        file_path: str,
        destination_path: str,
        content_type: Optional[str] = None
    ) -> Optional[str]:
        """Upload a file to Supabase Storage."""
        if not self.enabled:
            print(f"Supabase disabled, skipping upload: {file_path}")
            return None
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        if content_type is None:
            ext = Path(file_path).suffix.lower()
            content_types = {
                '.mp4': 'video/mp4',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.json': 'application/json',
            }
            content_type = content_types.get(ext, 'application/octet-stream')
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            upload_url = f"{self.storage_url}/{bucket}/{destination_path}"
            
            headers = {
                **self.headers,
                "Content-Type": content_type,
                "x-upsert": "true",
            }
            
            with httpx.Client(timeout=300) as client:
                response = client.post(upload_url, content=file_data, headers=headers)
            
            if response.status_code in [200, 201]:
                public_url = f"{self.url}/storage/v1/object/public/{bucket}/{destination_path}"
                print(f"Uploaded: {destination_path}")
                return public_url
            else:
                print(f"Upload failed ({response.status_code}): {response.text}")
                return None
                
        except Exception as e:
            print(f"Upload error: {e}")
            return None


_uploader = None

def get_uploader() -> SupabaseUploader:
    global _uploader
    if _uploader is None:
        _uploader = SupabaseUploader()
    return _uploader
