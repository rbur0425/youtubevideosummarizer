import os
import math
import tempfile
import whisper
import anthropic
import subprocess
import cv2
import numpy as np
import time
import argparse
import gc  # For garbage collection
import random
import backoff
import shutil
import requests
import re
import os
from PIL import Image, ImageDraw, ImageFont
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import urllib.request
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# Load environment variables from .env file
load_dotenv()

def download_audio(youtube_url):
    """Download audio from YouTube video using subprocess and yt-dlp"""
    print("Fetching video information...")
    video_id = extract_video_id(youtube_url)
    if not video_id:
        print("Error: Could not extract video ID from URL")
        return None, None
    
    # Get video metadata first
    video_title, video_length, _ = get_youtube_info(youtube_url)
    
    if not video_title or video_title.startswith("YouTube Video"):
        print("Warning: Could not get proper video metadata, but will try to download audio anyway")
    else:
        print(f"Video title: {video_title}")
        if video_length > 0:
            print(f"Video length: {video_length} seconds ({video_length//60}:{video_length%60:02d})")
    
    # Create directories
    temp_dir = os.path.join(tempfile.gettempdir(), 'youtube_summarizer')
    os.makedirs(temp_dir, exist_ok=True)
    audio_path = os.path.join(temp_dir, f"{video_id}.mp3")
    
    # Check if file already exists and is not empty
    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        print(f"✓ Audio already downloaded: {os.path.basename(audio_path)}")
        return audio_path, video_title
    
    print("Downloading audio...")
    
    # Retry decorator with backoff strategy
    @backoff.on_exception(backoff.expo,
                         (subprocess.SubprocessError, subprocess.CalledProcessError),
                         max_tries=5,
                         max_time=60,
                         jitter=backoff.full_jitter)
    def download_with_retry():
        try:
            # Use yt-dlp via subprocess
            cmd = [
                'yt-dlp',
                '--extract-audio',
                '--audio-format', 'mp3',
                '--audio-quality', '192K',
                '--output', os.path.join(temp_dir, f"{video_id}.%(ext)s"),
                '--no-playlist',
                youtube_url
            ]
            
            # Run the command
            print(f"Running yt-dlp with command: {' '.join(cmd)}")
            process = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Check if the file was created
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                print(f"✓ Audio downloaded: {os.path.basename(audio_path)}")
                return audio_path
            else:
                print("Warning: Download command completed but file not found or empty")
                raise FileNotFoundError("Expected audio file not found after download")
        except subprocess.CalledProcessError as e:
            print(f"yt-dlp failed with error: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise
    
    try:
        # Execute the download with retry
        return download_with_retry(), video_title
    except Exception as e:
        print(f"Failed to download audio after several attempts: {e}")
        
        # Alternative approach with different options
        try:
            print("Trying alternate download method...")
            cmd = [
                'yt-dlp',
                '--extract-audio',
                '--audio-format', 'mp3',
                '--audio-quality', '128K',  # Lower quality as fallback
                '--format', 'worstaudio/worst',  # Try worst quality as a fallback
                '--output', os.path.join(temp_dir, f"{video_id}.%(ext)s"),
                '--no-playlist',
                '--force-ipv4',  # Force IPv4 as a fallback
                youtube_url
            ]
            
            print(f"Running alternative yt-dlp command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                print(f"✓ Audio downloaded with alternate method: {os.path.basename(audio_path)}")
                return audio_path, video_title
            else:
                print("Alternative download failed: File not found or empty")
                return None, None
        except Exception as fallback_error:
            print(f"Alternate download method also failed: {fallback_error}")
            return None, None

def transcribe_audio(audio_file):
    """Transcribe audio using Whisper"""
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("Model loaded. Transcribing audio (this may take a while)...")
    result = model.transcribe(audio_file)
    print("✓ Transcription complete!")
    return result["text"]

def extract_video_id(youtube_url):
    """Extract video ID from different YouTube URL formats"""
    if "youtube.com/watch" in youtube_url:
        video_id = youtube_url.split("v=")[1].split("&")[0] if "&" in youtube_url else youtube_url.split("v=")[1]
    elif "youtu.be/" in youtube_url:
        video_id = youtube_url.split("youtu.be/")[1].split("?")[0] if "?" in youtube_url else youtube_url.split("youtu.be/")[1]
    else:
        print("Invalid YouTube URL format")
        return None
    return video_id

def get_youtube_info(youtube_url):
    """Get YouTube video info using yt-dlp via subprocess"""
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return f"YouTube Video Unknown", 0, {}
    
    try:
        # Use yt-dlp to get video metadata as JSON
        cmd = [
            'yt-dlp',
            '--dump-json',
            '--no-playlist',
            '--skip-download',
            youtube_url
        ]
        
        print("Getting video metadata with yt-dlp...")
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Parse the JSON output
        import json
        video_info = json.loads(process.stdout)
        
        # Extract title and duration
        video_title = video_info.get('title', f"YouTube Video {video_id}")
        video_length = int(video_info.get('duration', 0))
        
        print(f"✓ Successfully retrieved video info using yt-dlp")
        return video_title, video_length, video_info
        
    except Exception as e:
        print(f"Error getting video info with yt-dlp: {e}")
        return f"YouTube Video {video_id}", 0, {}

def get_transcript(youtube_url):
    """Try to get transcript directly from YouTube"""
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return None
    
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ' '.join([item['text'] for item in transcript_list])
        return transcript
    except Exception as e:
        print(f"Transcript error: {e}")
        print("No transcript available, falling back to audio download and transcription")
        return None

def chunk_text(text, chunk_size=1500, overlap=200):
    """Split text into chunks with overlap using a memory-efficient generator"""
    start = 0
    text_len = len(text)
    chunk_count = 0
    
    print(f"Starting text chunking (total length: {text_len:,} characters)")
    print(f"Settings: chunk_size={chunk_size}, overlap={overlap}")
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Log progress every 5 chunks
        if chunk_count % 5 == 0:
            progress = (start / text_len) * 100
            print(f"Progress: {progress:.1f}% (processing character {start:,} of {text_len:,})")
        
        # Adjust end to avoid cutting words
        if end < text_len:
            # Look for the last space within the chunk size limit
            last_space = text.rfind(' ', start, end)
            if last_space != -1 and last_space > start:
                end = last_space + 1  # Include the space
        
        # Get the chunk
        chunk = text[start:end]
        chunk_count += 1
        
        # Log chunk details
        if chunk_count % 5 == 0:
            print(f"Created chunk {chunk_count}: {len(chunk):,} characters "
                  f"(from char {start:,} to {end:,})")
        
        yield chunk
        
        # Move start position forward, ensuring we make progress
        if end == text_len:
            break  # We've reached the end
        start = end - overlap
        if start <= end - chunk_size:  # Ensure we're making forward progress
            start = end - (chunk_size // 2)  # Move forward at least half a chunk
    
    print(f"\n✓ Chunking complete! Created {chunk_count} chunks")

def summarize_chunk(chunk, client):
    """Summarize a single chunk using Claude API"""
    prompt = f"""Please provide a concise summary of the following transcript section:

{chunk}

Summary:"""
    
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=250,
        temperature=0.7,
        system="You are an expert at summarizing video content. Create clear, accurate, and concise summaries that capture the key points.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text

def summarize_batch(batch_chunks, client, batch_idx, total_batches, start_idx):
    """Summarize a batch of chunks"""
    batch_summaries = []
    for i, chunk in enumerate(batch_chunks):
        chunk_idx = start_idx + i
        print(f"Summarizing chunk {chunk_idx+1}/{total_chunks} (batch {batch_idx+1}/{total_batches})...")
        summary = summarize_chunk(chunk, client)
        print(f"✓ Completed chunk {chunk_idx+1}/{total_chunks}")
        batch_summaries.append(summary)
    
    return batch_summaries

def extract_video_screenshots(youtube_url, output_dir, interval=30, max_screenshots=10):
    """Extract screenshots from video using yt-dlp via subprocess"""
    print("\n=== VISUAL TIMELINE EXTRACTION ===")
    print("Fetching video information...")
    
    video_id = extract_video_id(youtube_url)
    if not video_id:
        print("Error: Could not extract video ID from URL")
        return []
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"{video_id}.mp4")
    
    # Check if file already exists and is not empty
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        print(f"✓ Video already downloaded: {os.path.basename(video_path)}")
        return create_timeline_from_video(video_path, interval, max_screenshots)
    
    print("Downloading video for screenshots...")
    
    @backoff.on_exception(backoff.expo,
                        (subprocess.SubprocessError, subprocess.CalledProcessError),
                        max_tries=3,
                        max_time=60,
                        jitter=backoff.full_jitter)
    def download_video_with_subprocess():
        try:
            # Download command for best video with height <= 480px
            cmd = [
                'yt-dlp',
                '-f', 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]',
                '--merge-output-format', 'mp4',
                '--output', video_path,
                '--no-playlist',
                youtube_url
            ]
            
            print(f"Running yt-dlp for video: {' '.join(cmd)}")
            process = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Check if the file was created
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                print(f"✓ Video downloaded: {os.path.basename(video_path)}")
                return True
            else:
                print("Warning: Download command completed but video file not found or empty")
                raise FileNotFoundError("Expected video file not found after download")
        except subprocess.CalledProcessError as e:
            print(f"Video download failed: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise
    
    try:
        # Execute the download with retry
        if download_video_with_subprocess():
            return create_timeline_from_video(video_path, interval, max_screenshots)
        else:
            print("Failed to download video")
            return []
    except Exception as e:
        print(f"Failed to download video after multiple attempts: {e}")
        
        # Try with alternative format
        try:
            print("Trying alternative video format...")
            cmd = [
                'yt-dlp',
                '-f', 'bestvideo[height<=360]/best[height<=360]',
                '--merge-output-format', 'mp4',
                '--output', video_path,
                '--no-playlist',
                '--force-ipv4',
                youtube_url
            ]
            
            print(f"Running alternative yt-dlp command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                print(f"✓ Video downloaded with alternative format: {os.path.basename(video_path)}")
                return create_timeline_from_video(video_path, interval, max_screenshots)
            else:
                print("Alternative download failed: File not found or empty")
                return []
        except Exception as alt_error:
            print(f"Alternative video download also failed: {alt_error}")
            return []

def create_timeline_from_thumbnail(thumbnail_path, video_id, interval, max_screenshots):
    """Create timeline screenshots from a single thumbnail"""
    try:
        print(f"Creating {max_screenshots} visual markers from thumbnail...")
        screenshots = []
        img = Image.open(thumbnail_path)
        
        for i in range(max_screenshots):
            timestamp = i * interval
            screenshot = img.copy()
            draw = ImageDraw.Draw(screenshot)
            
            # Add timestamp overlay
            timestamp_str = f"{int(timestamp//60)}:{int(timestamp%60):02d}"
            font_size = 30
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                font = ImageFont.load_default()
            
            # Add semi-transparent overlay
            overlay_height = 40
            overlay = Image.new('RGBA', (screenshot.width, overlay_height), (0, 0, 0, 128))
            screenshot.paste(overlay, (0, screenshot.height - overlay_height), overlay)
            
            # Add timestamp text
            text_bbox = draw.textbbox((0, 0), timestamp_str, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x = (screenshot.width - text_width) // 2
            y = screenshot.height - overlay_height + (overlay_height - text_height) // 2
            draw.text((x, y), timestamp_str, fill='white', font=font)
            
            if screenshot.mode != 'RGB':
                screenshot = screenshot.convert('RGB')
            
            screenshot_path = os.path.join(tempfile.gettempdir(), 'youtube_summarizer', f"{video_id}_{i}.jpg")
            screenshot.save(screenshot_path, "JPEG")
            screenshots.append({
                'path': screenshot_path,
                'timestamp': timestamp,
                'timestamp_str': timestamp_str
            })
        
        print(f"✓ Created {len(screenshots)} visual markers")
        return screenshots
    except Exception as e:
        print(f"Error creating timeline from thumbnail: {e}")
        return []

def create_timeline_from_video(video_path, interval, max_screenshots):
    """Create timeline screenshots from video file"""
    try:
        print(f"Extracting {max_screenshots} screenshots from video...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return []
        
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        actual_interval = video_length / max_screenshots if video_length > max_screenshots * interval else interval
        
        screenshots = []
        for i in range(max_screenshots):
            timestamp = i * actual_interval
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            screenshot = Image.fromarray(frame_rgb)
            
            # Add timestamp overlay (same as thumbnail version)
            timestamp_str = f"{int(timestamp//60)}:{int(timestamp%60):02d}"
            draw = ImageDraw.Draw(screenshot)
            
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
            except:
                font = ImageFont.load_default()
            
            overlay_height = 40
            overlay = Image.new('RGBA', (screenshot.width, overlay_height), (0, 0, 0, 128))
            screenshot.paste(overlay, (0, screenshot.height - overlay_height), overlay)
            
            text_bbox = draw.textbbox((0, 0), timestamp_str, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x = (screenshot.width - text_width) // 2
            y = screenshot.height - overlay_height + (overlay_height - text_height) // 2
            draw.text((x, y), timestamp_str, fill='white', font=font)
            
            screenshot_path = os.path.join(tempfile.gettempdir(), 'youtube_summarizer', f"{os.path.basename(video_path)}_{i}.jpg")
            screenshot.save(screenshot_path, "JPEG")
            screenshots.append({
                'path': screenshot_path,
                'timestamp': timestamp,
                'timestamp_str': timestamp_str
            })
        
        cap.release()
        os.remove(video_path)  # Clean up the video file
        print(f"✓ Created {len(screenshots)} screenshots")
        return screenshots
        
    except Exception as e:
        print(f"Error creating timeline from video: {e}")
        if os.path.exists(video_path):
            os.remove(video_path)  # Clean up on error
        return []
        
    except Exception as e:
        print(f"Error creating timeline from video: {e}")
        if os.path.exists(video_path):
            os.remove(video_path)  # Clean up on error
        return []

def create_visual_timeline(screenshots, grid_size=(2, 5)):
    """Create a grid of screenshots with timestamps"""
    rows, cols = grid_size
    max_images = rows * cols
    
    # Use only the number of screenshots we can fit
    screenshots = screenshots[:max_images]
    
    if not screenshots:
        print("Error: No screenshots available to create timeline.")
        return None
    
    print(f"Creating {rows}x{cols} grid of screenshots...")
    
    # Resize all images to a standard size
    width, height = 320, 180  # 16:9 aspect ratio
    
    # Create a blank canvas
    timeline_width = cols * width
    timeline_height = rows * height
    timeline_image = Image.new('RGB', (timeline_width, timeline_height), color='black')
    
    # Add each screenshot to the grid
    draw = ImageDraw.Draw(timeline_image)
    
    try:
        # Try to load a font, fall back to default if not available
        font = ImageFont.truetype("arial.ttf", 16)
        print("Using Arial font for timestamps")
    except:
        font = ImageFont.load_default()
        print("Using default font for timestamps (Arial not found)")
    
    for i, screenshot in enumerate(screenshots):
        # Calculate position
        row = i // cols
        col = i % cols
        x = col * width
        y = row * height
        
        # Resize image
        img = Image.open(screenshot['path'])
        img_resized = img.resize((width, height))
        
        # Paste into the canvas
        timeline_image.paste(img_resized, (x, y))
        
        # Add timestamp
        text_color = (255, 255, 255)  # White text
        
        # Draw text with shadow for better visibility
        text_x = x + 10
        text_y = y + height - 30
        
        # Draw background
        text_width = draw.textlength(screenshot['timestamp_str'], font=font)
        text_height = 20
        draw.rectangle(
            [(text_x - 5, text_y - 5), (text_x + text_width + 5, text_y + text_height)],
            fill=(0, 0, 0, 180)
        )
        
        # Draw text
        draw.text((text_x, text_y), screenshot['timestamp_str'], font=font, fill=text_color)
        
        # Draw frame number
        frame_num = f"Frame {i+1}"
        draw.text((x + 10, y + 10), frame_num, font=font, fill=text_color)
    
    print(f"✓ Visual grid created successfully: {timeline_width}x{timeline_height}px")
    return timeline_image

def create_final_summary_with_visuals(summaries_file, video_title, screenshots_info, client):
    """Create final summary with visual context"""
    # Read summaries from file to avoid memory issues
    with open(summaries_file, 'r', encoding='utf-8') as f:
        all_summaries = f.read()
    
    # Create a description of the visual timeline if we have screenshots
    visual_context_text = ""
    if screenshots_info:
        visual_context = []
        for i, screenshot in enumerate(screenshots_info[:10]):
            mins, secs = map(int, screenshot['timestamp_str'].split(':'))
            timestamp_sec = mins * 60 + secs
            visual_context.append(f"Frame {i+1}: {mins}:{secs:02d} ({timestamp_sec} seconds into the video)")
        
        visuals_desc = "\n".join(visual_context)
        visual_context_text = f"""
### Key Visual Frames:
{visuals_desc}

Where relevant, reference the visual frames to give context to what was happening visually during important points of the discussion.
"""
    
    prompt = f"""You are tasked with creating a comprehensive summary of a YouTube video titled "{video_title}".
Below are summaries of different sections of the video{", as well as information about key frames captured from the video" if screenshots_info else ""}:

### Video Section Summaries:
{all_summaries}

{visual_context_text}
Please create a cohesive, well-structured final summary of the entire video.

Final Summary:"""
    
    system_prompt = "You are an expert at synthesizing information into comprehensive summaries"
    if screenshots_info:
        system_prompt += " that blend textual and visual information. Create a summary that references both what was said and what was shown in the video."
    else:
        system_prompt += ". Create a well-structured summary that captures the key points from the video transcript."
    
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=800,
        temperature=0.7,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text

def process_transcript_chunks(transcript, chunk_size=1500, overlap=200):
    """Process transcript in chunks and generate summaries"""
    print("\n=== PROCESSING TRANSCRIPT ===")
    print("Chunking transcript for analysis...")
    
    # Adjust chunk size if transcript is very large
    total_length = len(transcript)
    if total_length > 10000:
        original_chunk_size = chunk_size
        chunk_size = min(chunk_size, total_length // 10)  # Ensure we have at least 10 chunks
        print(f"Large transcript detected ({total_length:,} characters), reducing chunk size from {original_chunk_size} to {chunk_size}...")
    
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    print("\nStarting chunk summarization...")
    batch_summaries = []
    chunk_count = 0
    
    try:
        for chunk in chunk_text(transcript, chunk_size, overlap):
            chunk_count += 1
            print(f"\nProcessing chunk {chunk_count}...")
            print(f"Chunk size: {len(chunk):,} characters")
            
            try:
                summary = summarize_chunk(chunk, client)
                print(f"✓ Successfully summarized chunk {chunk_count}")
                batch_summaries.append(summary)
            except Exception as e:
                print(f"Error summarizing chunk {chunk_count}: {e}")
                print("Continuing with next chunk...")
                continue
            
            # Free up memory
            gc.collect()
    except Exception as e:
        print(f"Error during chunk processing: {e}")
    
    print(f"\n✓ Completed processing {chunk_count} chunks")
    print(f"Generated {len(batch_summaries)} summaries")
    
    return batch_summaries

def sanitize_filename(filename):
    """Convert a string to a valid filename by removing or replacing invalid characters"""
    # Replace invalid characters with underscore
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove any leading/trailing spaces or dots
    filename = filename.strip('. ')
    
    # Limit length to avoid filesystem issues
    if len(filename) > 200:
        filename = filename[:197] + '...'
    
    return filename

def summarize_youtube_video(
    youtube_url, 
    claude_api_key, 
    output_dir="output",
    screenshot_interval=30,
    max_screenshots=10,
    chunk_size=1500,
    overlap=200
):
    """Main function to process YouTube video and generate summary"""
    os.environ['ANTHROPIC_API_KEY'] = claude_api_key
    
    print("\n=== STARTING VIDEO PROCESSING ===")
    print(f"Processing URL: {youtube_url}")
    
    # Get video info and title first
    video_title, video_length, video_info = get_youtube_info(youtube_url)
    video_id = extract_video_id(youtube_url)
    
    # Create output directory using video title or ID
    folder_name = sanitize_filename(video_title) if video_title else video_id
    output_dir = os.path.join("output", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print("\n=== TRANSCRIPT EXTRACTION ===")
    # Try to get transcript directly first
    print("Attempting to get transcript directly from YouTube...")
    transcript = get_transcript(youtube_url)
    
    if transcript:
        print("✓ Successfully retrieved transcript from YouTube!")
    else:
        # Fall back to downloading and transcribing
        print("YouTube transcript not available.")
        print("\n--- Falling back to audio transcription ---")
        audio_file, _ = download_audio(youtube_url)
        transcript = transcribe_audio(audio_file)
        print("Cleaning up temporary audio file...")
        os.remove(audio_file)
        print("✓ Audio file removed")
    
    print(f"\nTranscript length: {len(transcript)} characters")
    
    # Extract screenshots
    screenshots = extract_video_screenshots(
    youtube_url, 
    output_dir,  # Pass the output directory 
    interval=screenshot_interval, 
    max_screenshots=max_screenshots
)
    
    # Create visual timeline only if we have screenshots
    timeline_path = None
    if screenshots:
        print("\n=== CREATING VISUAL TIMELINE ===")
        print(f"Generating timeline grid ({2}x{5})...")
        timeline_image = create_visual_timeline(screenshots, grid_size=(2, 5))
        
        if timeline_image:
            # Save the timeline image
            timeline_path = os.path.join(output_dir, "visual_timeline.jpg")
            timeline_image.save(timeline_path)
            print(f"✓ Visual timeline saved to {timeline_path}")
    else:
        print("\n=== SKIPPING VISUAL TIMELINE ===")
        print("No screenshots available, proceeding with text-only summary")
    
    # Process transcript
    print("\n=== PROCESSING TRANSCRIPT ===")
    print("Chunking transcript for analysis...")
    
    # Use smaller chunks for large transcripts
    if len(transcript) > 20000:
        print("Large transcript detected, using smaller chunk size...")
        chunk_size = 1000
    else:
        chunk_size = 1500
    
    # First, count the chunks without storing them
    global total_chunks
    total_chunks = sum(1 for _ in chunk_text(transcript, chunk_size=chunk_size))
    
    # Create a new generator for processing
    if total_chunks > 80:
        print(f"⚠️ Very large transcript detected ({total_chunks} chunks)!")
        print(f"Limiting to {80} chunks to prevent memory issues...")
        total_chunks = 80
    
    print(f"✓ Split transcript into {total_chunks} chunks for processing")
    
    # Prepare for batch processing
    print("\n=== GENERATING SUMMARY WITH CLAUDE API ===")
    print(f"Processing in batches of {5} chunks...")
    
    # File to store all summaries
    summaries_file = os.path.join(output_dir, "chunk_summaries.txt")
    if os.path.exists(summaries_file):
        os.remove(summaries_file)
    
    # Process chunks in batches
    num_batches = (total_chunks + 5 - 1) // 5
    chunks_processed = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * 5
        end_idx = min(start_idx + 5, total_chunks)
        
        # Create a new generator for each batch
        batch_chunks = []
        chunk_gen = chunk_text(transcript, chunk_size=chunk_size)
        
        # Skip chunks before start_idx
        for _ in range(start_idx):
            next(chunk_gen, None)
        
        # Collect chunks for current batch
        for _ in range(end_idx - start_idx):
            chunk = next(chunk_gen, None)
            if chunk is not None:
                batch_chunks.append(chunk)
                chunks_processed += 1
            if chunks_processed >= 40:
                break
        
        print(f"\nProcessing batch {batch_idx+1}/{num_batches} (chunks {start_idx+1}-{end_idx}/{total_chunks})...")
        batch_summaries = summarize_batch(batch_chunks, anthropic.Anthropic(api_key=claude_api_key), batch_idx, num_batches, start_idx)
        
        # Write batch summaries to file
        with open(summaries_file, "a", encoding="utf-8") as f:
            for summary in batch_summaries:
                f.write(summary + "\n\n---\n\n")
        
        print(f"✓ Batch {batch_idx+1}/{num_batches} processed and saved to disk")
        
        # Clear memory
        del batch_chunks
        del batch_summaries
        gc.collect()  # Force garbage collection
    
    # Create final summary
    print("\nGenerating final comprehensive summary...")
    final_summary = create_final_summary_with_visuals(
        summaries_file, video_title, screenshots, anthropic.Anthropic(api_key=claude_api_key)
    )
    print("✓ Final summary generated!")
    
    # Save transcript and summary
    print("\n=== SAVING RESULTS ===")
    transcript_path = os.path.join(output_dir, "transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    summary_path = os.path.join(output_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(final_summary)
    
    print(f"✓ Full transcript saved to {transcript_path}")
    print(f"✓ Summary saved to {summary_path}")
    
    print("\n==================================================")
    print("  SUMMARY GENERATION COMPLETE")
    print("==================================================")
    
    return {
        "title": video_title,
        "summary": final_summary,
        "timeline_image_path": timeline_path,
        "transcript_path": transcript_path,
        "summary_path": summary_path
    }

def main():
    """Main entry point with command-line argument support"""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="YouTube Video Summarizer with Visual Timeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments
    parser.add_argument(
        "--url", 
        type=str, 
        help="YouTube video URL"
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="Claude API key (can also use ANTHROPIC_API_KEY env variable)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output", 
        help="Output directory for files"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=30, 
        help="Seconds between screenshots"
    )
    parser.add_argument(
        "--screenshots", 
        type=int, 
        default=10, 
        help="Maximum number of screenshots"
    )
    parser.add_argument(
        "--grid-rows", 
        type=int, 
        default=2, 
        help="Rows in the visual timeline grid"
    )
    parser.add_argument(
        "--grid-cols", 
        type=int, 
        default=5, 
        help="Columns in the visual timeline grid"
    )
    parser.add_argument(
        "--max-chunk-size", 
        type=int, 
        default=1000, 
        help="Maximum size of transcript chunks"
    )
    parser.add_argument(
        "--max-chunks", 
        type=int, 
        default=40, 
        help="Maximum number of chunks to process"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=5, 
        help="Number of chunks to process in each batch"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    print("\n🎬 YouTube Video Summarizer with Visual Timeline 🎬\n")
    
    # Get YouTube URL (command line or prompt)
    youtube_url = args.url
    if not youtube_url:
        youtube_url = input("Enter YouTube URL: ")
    
    # Get API key (command line, .env file, environment variable, or prompt)
    claude_api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not claude_api_key:
        claude_api_key = input("Enter your Anthropic API key: ")
    
    # Track execution time
    start_time = time.time()
    
    # Process the video
    result = summarize_youtube_video(
        youtube_url=youtube_url, 
        claude_api_key=claude_api_key,
        output_dir=args.output_dir,
        screenshot_interval=args.interval,
        max_screenshots=args.screenshots,
        chunk_size=args.max_chunk_size,
        overlap=200
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    # Display results
    print(f"\n=== RESULTS ===")
    print(f"Video: {result['title']}")
    print(f"Processing time: {minutes} minutes, {seconds} seconds")
    print("\n=== SUMMARY ===")
    print(result['summary'])
    print("\n=== OUTPUT FILES ===")
    if result['timeline_image_path']:
        print(f"📊 Visual timeline: {result['timeline_image_path']}")
    else:
        print(f"📊 Visual timeline: Not available (video download blocked)")
    print(f"📝 Full transcript: {result['transcript_path']}")
    print(f"📋 Summary text: {result['summary_path']}")
    print("\nDone! 🎉")

if __name__ == "__main__":
    main()