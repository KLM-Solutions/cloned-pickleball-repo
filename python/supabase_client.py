"""
Supabase Storage Client for uploading analysis results.

STORAGE ONLY - NO DATABASE WRITES
All database operations are handled by TypeScript (Next.js).

This module only uploads:
- Annotated video to 'analysis-results' bucket

Environment variables required:
- SUPABASE_URL: Your Supabase project URL
- SUPABASE_SERVICE_KEY: Service role key (not anon key)
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
import httpx


class SupabaseUploader:
    """Handles file uploads to Supabase Storage ONLY (no database writes)."""
    
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
        """
        Upload a file to Supabase Storage.
        
        Args:
            bucket: Bucket name (e.g., 'analysis-results')
            file_path: Local file path
            destination_path: Path in bucket (e.g., 'job123/video.mp4')
            content_type: MIME type (auto-detected if None)
        
        Returns:
            Public URL of uploaded file, or None on failure
        """
        if not self.enabled:
            print(f"Supabase disabled, skipping upload: {file_path}")
            return None
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        # Auto-detect content type
        if content_type is None:
            ext = Path(file_path).suffix.lower()
            content_types = {
                '.mp4': 'video/mp4',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
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
                "x-upsert": "true",  # Overwrite if exists
            }
            
            with httpx.Client(timeout=300) as client:  # 5 min timeout for large files
                response = client.post(upload_url, content=file_data, headers=headers)
            
            if response.status_code in [200, 201]:
                public_url = f"{self.url}/storage/v1/object/public/{bucket}/{destination_path}"
                print(f"Uploaded: {destination_path} -> {public_url}")
                return public_url
            else:
                print(f"Upload failed ({response.status_code}): {response.text}")
                return None
                
        except Exception as e:
            print(f"Upload error: {e}")
            return None
    
    def upload_directory(
        self,
        bucket: str,
        local_dir: str,
        destination_prefix: str,
        file_pattern: str = "*.png"
    ) -> List[Dict[str, str]]:
        """
        Upload all matching files in a directory.
        
        Args:
            bucket: Bucket name
            local_dir: Local directory path
            destination_prefix: Prefix in bucket (e.g., 'job123/frames/')
            file_pattern: Glob pattern for files
        
        Returns:
            List of {filename, url} dicts
        """
        if not self.enabled:
            return []
        
        uploaded = []
        local_path = Path(local_dir)
        
        for file_path in sorted(local_path.glob(file_pattern)):
            dest_path = f"{destination_prefix}{file_path.name}"
            url = self.upload_file(bucket, str(file_path), dest_path)
            if url:
                uploaded.append({
                    "filename": file_path.name,
                    "url": url
                })
        
        return uploaded


# Singleton instance
_uploader = None

def get_uploader() -> SupabaseUploader:
    """Get singleton SupabaseUploader instance."""
    global _uploader
    if _uploader is None:
        _uploader = SupabaseUploader()
    return _uploader
