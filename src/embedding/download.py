"""
Download utilities for pre-signed S3 URLs and other HTTP/HTTPS URLs
Uses requests for downloads with progress bars via tqdm
"""
import os
import tempfile
import time
import requests
from typing import Optional, Callable
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    tqdm = None


def download_from_url(url: str, output_path: Optional[str] = None, chunk_size: int = 8 * 1024 * 1024, progress_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """
    Downloads a file from a pre-signed S3 URL or HTTP/HTTPS URL using requests.
    Supports progress bars via tqdm.
    
    Args:
        url: Pre-signed S3 URL, HTTP/HTTPS URL, or local file path
        output_path: Optional output path. If None, creates a temporary file (for URLs only).
        chunk_size: Chunk size for streaming download (default: 8MB for better performance)
        progress_callback: Optional callback function(status_message) for progress updates
        
    Returns:
        Path to downloaded file or local file path, or None if download failed
    """
    if progress_callback is None:
        progress_callback = print
    
    # Check if it's a local file path (file:// protocol or absolute path)
    local_path = None
    if url.startswith('file://'):
        local_path = url[7:]  # Remove 'file://' prefix
    elif url.startswith('/') and os.path.exists(url):
        local_path = url
    
    # If it's a local file, return it directly
    if local_path:
        if os.path.exists(local_path):
            progress_callback(f"Using local file: {local_path}")
            return local_path
        else:
            raise Exception(f"Local file path does not exist: {local_path}")
    
    # Download from URL (pre-signed S3 URL or HTTP/HTTPS)
    return _download_from_http(url, output_path, chunk_size, progress_callback)


def _download_from_http(url: str, output_path: Optional[str], chunk_size: int, progress_callback: Callable) -> Optional[str]:
    """Download from HTTP/HTTPS URL using requests."""
    session = None
    try:
        if output_path is None:
            # Create temporary file
            file_ext = os.path.splitext(url.split('?')[0])[1] or '.tmp'
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            output_path = temp_file.name
            temp_file.close()
        
        progress_callback("Downloading...")
        
        # Use session for connection pooling and better performance
        session = requests.Session()
        session.headers.update({
            'Connection': 'keep-alive',
            'Accept-Encoding': 'gzip, deflate'
        })
        
        # Download with streaming - use longer timeout and verify connection
        progress_callback(f"Connecting to: {url[:80]}...")
        try:
            response = session.get(url, stream=True, timeout=(30, 300), allow_redirects=True)
            progress_callback(f"Connected. Status: {response.status_code}")
        except requests.exceptions.Timeout as e:
            progress_callback(f"Connection timeout: {e}")
            raise
        except requests.exceptions.RequestException as e:
            progress_callback(f"Connection error: {e}")
            raise
        
        # Check status code
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.reason}")
        
        # Get total file size from headers if available
        total_size = int(response.headers.get('content-length', 0))
        if total_size > 0:
            progress_callback(f"File size: {total_size / (1024*1024):.2f} MB")
        
        # Download with progress bar
        downloaded = 0
        chunk_count = 0
        start_time = time.time()
        last_chunk_time = start_time
        last_file_size_check = start_time
        
        with open(output_path, 'wb') as f:
            if tqdm and total_size > 0:
                # Use tqdm progress bar if available and we know the file size
                with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, 
                         desc="Downloading", miniters=1, mininterval=0.1, 
                         ncols=100, ascii=False, dynamic_ncols=True) as pbar:
                    try:
                        pbar.n = 0  # Initialize progress bar position
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            current_time = time.time()
                            if chunk:
                                f.write(chunk)
                                f.flush()  # Ensure data is written to disk
                                downloaded += len(chunk)
                                chunk_count += 1
                                last_chunk_time = current_time
                                pbar.update(len(chunk))
                                
                                # Periodically check actual file size and sync progress bar
                                if current_time - last_file_size_check > 0.5:  # Check every 0.5 seconds
                                    try:
                                        actual_size = os.path.getsize(output_path)
                                        if actual_size > pbar.n:
                                            # File size is ahead of progress bar, update it
                                            pbar.n = actual_size
                                            pbar.refresh()
                                        last_file_size_check = current_time
                                    except:
                                        pass  # Ignore errors checking file size
                                
                                # Check for timeout (no data for 60 seconds)
                                if chunk_count == 1:
                                    progress_callback("Receiving data...")
                            else:
                                # Empty chunk - check if we've been waiting too long
                                elapsed = current_time - last_chunk_time
                                if chunk_count == 0:
                                    # No data received yet - check connection timeout
                                    if elapsed > 30:
                                        raise Exception("Connection timeout: No data received for 30 seconds")
                                    elif elapsed > 10:
                                        progress_callback("Warning: No data received for 10 seconds...")
                                elif elapsed > 60:
                                    raise Exception("Download timeout: No data received for 60 seconds")
                        
                        # Final sync: check file size one more time after download completes
                        try:
                            final_size = os.path.getsize(output_path)
                            if final_size > pbar.n:
                                pbar.n = final_size
                                pbar.refresh()
                        except:
                            pass
                    except KeyboardInterrupt:
                        progress_callback("Download interrupted by user")
                        raise
                    except Exception as e:
                        progress_callback(f"Error during download: {e}")
                        raise
            elif tqdm:
                # Use tqdm without total size (unknown size)
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, 
                         desc="Downloading", miniters=1, mininterval=0.1,
                         ncols=100, ascii=False, dynamic_ncols=True) as pbar:
                    try:
                        pbar.n = 0  # Initialize progress bar position
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            current_time = time.time()
                            if chunk:
                                f.write(chunk)
                                f.flush()  # Ensure data is written to disk
                                downloaded += len(chunk)
                                chunk_count += 1
                                last_chunk_time = current_time
                                pbar.update(len(chunk))
                                
                                # Periodically check actual file size and sync progress bar
                                if current_time - last_file_size_check > 0.5:  # Check every 0.5 seconds
                                    try:
                                        actual_size = os.path.getsize(output_path)
                                        if actual_size > pbar.n:
                                            # File size is ahead of progress bar, update it
                                            pbar.n = actual_size
                                            pbar.refresh()
                                        last_file_size_check = current_time
                                    except:
                                        pass  # Ignore errors checking file size
                                
                                if chunk_count == 1:
                                    progress_callback("Receiving data...")
                            else:
                                elapsed = current_time - last_chunk_time
                                if chunk_count == 0:
                                    if elapsed > 30:
                                        raise Exception("Connection timeout: No data received for 30 seconds")
                                    elif elapsed > 10:
                                        progress_callback("Warning: No data received for 10 seconds...")
                                elif elapsed > 60:
                                    raise Exception("Download timeout: No data received for 60 seconds")
                        
                        # Final sync: check file size one more time after download completes
                        try:
                            final_size = os.path.getsize(output_path)
                            if final_size > pbar.n:
                                pbar.n = final_size
                                pbar.refresh()
                        except:
                            pass
                    except KeyboardInterrupt:
                        progress_callback("Download interrupted by user")
                        raise
                    except Exception as e:
                        progress_callback(f"Error during download: {e}")
                        raise
            else:
                # Fallback: download without progress bar
                last_chunk_time = time.time()
                for chunk in response.iter_content(chunk_size=chunk_size):
                    current_time = time.time()
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        chunk_count += 1
                        last_chunk_time = current_time
                        if chunk_count % 100 == 0:  # Print progress every 100 chunks
                            progress_callback(f"Downloaded: {downloaded / (1024*1024):.2f} MB")
                    elif current_time - last_chunk_time > 60:
                        raise Exception("Download timeout: No data received for 60 seconds")
        
        # Verify file was downloaded
        if not os.path.exists(output_path):
            raise Exception("Downloaded file does not exist")
        
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise Exception("Downloaded file is empty")
        
        if total_size > 0 and abs(file_size - total_size) > 1024:  # Allow 1KB difference
            progress_callback(f"Warning: File size mismatch. Expected: {total_size}, Got: {file_size}")
        
        progress_callback(f"Download complete: {file_size / (1024*1024):.2f} MB ({chunk_count} chunks)")
        return output_path
        
    except requests.exceptions.Timeout:
        error_msg = "Download timeout (300s exceeded)"
        progress_callback(f"Error: {error_msg}")
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return None
    except requests.exceptions.RequestException as e:
        error_msg = f"Request error: {str(e)}"
        progress_callback(f"Error: {error_msg}")
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return None
    except Exception as e:
        error_msg = f"Error downloading from URL: {type(e).__name__}: {str(e)}"
        progress_callback(f"Error: {error_msg}")
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return None
    finally:
        if session:
            session.close()


def download_transcript_from_url(url: str) -> Optional[list]:
    """
    Downloads and parses a transcript JSON from a pre-signed S3 URL or local file path.
    
    Args:
        url: Pre-signed S3 URL to transcript JSON file, or local file path (file:// or /path/to/file)
        
    Returns:
        List of transcript segments, or None if download/parse failed
    """
    import json
    
    # Check if it's a local file path (file:// protocol or absolute path)
    local_path = None
    if url.startswith('file://'):
        local_path = url[7:]  # Remove 'file://' prefix
    elif url.startswith('/') and os.path.exists(url):
        local_path = url
    
    # If it's a local file, read it directly
    if local_path:
        try:
            if not os.path.exists(local_path):
                print(f"Error: Local transcript file does not exist: {local_path}")
                return None
            
            with open(local_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Handle different transcript formats
            if isinstance(transcript_data, list):
                return transcript_data
            elif isinstance(transcript_data, dict) and 'segments' in transcript_data:
                return transcript_data['segments']
            elif isinstance(transcript_data, dict) and 'transcript' in transcript_data:
                return transcript_data['transcript']
            else:
                print(f"Warning: Unexpected transcript format in local file. Returning as-is.")
                return transcript_data if isinstance(transcript_data, list) else None
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from local transcript file: {e}")
            return None
        except Exception as e:
            print(f"Error reading local transcript file: {e}")
            return None
    
    # Otherwise, download from URL
    try:
        # Download transcript with progress bar
        if tqdm:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            content = b''
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc="Downloading transcript") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            content += chunk
                            pbar.update(len(chunk))
            else:
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc="Downloading transcript") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            content += chunk
                            pbar.update(len(chunk))
            transcript_data = json.loads(content.decode('utf-8'))
        else:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            transcript_data = response.json()
        
        # Handle different transcript formats
        if isinstance(transcript_data, list):
            return transcript_data
        elif isinstance(transcript_data, dict) and 'segments' in transcript_data:
            return transcript_data['segments']
        elif isinstance(transcript_data, dict) and 'transcript' in transcript_data:
            return transcript_data['transcript']
        else:
            print(f"Warning: Unexpected transcript format. Returning as-is.")
            return transcript_data if isinstance(transcript_data, list) else None
            
    except Exception as e:
        print(f"Error downloading/parsing transcript from URL: {e}")
        return None
