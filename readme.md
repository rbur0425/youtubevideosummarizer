```
██╗   ██╗██╗██████╗ ███████╗ ██████╗     ███████╗██╗   ██╗███╗   ███╗
██║   ██║██║██╔══██╗██╔════╝██╔═══██╗    ██╔════╝██║   ██║████╗ ████║
██║   ██║██║██║  ██║█████╗  ██║   ██║    ███████╗██║   ██║██╔████╔██║
╚██╗ ██╔╝██║██║  ██║██╔══╝  ██║   ██║    ╚════██║██║   ██║██║╚██╔╝██║
 ╚████╔╝ ██║██████╔╝███████╗╚██████╔╝    ███████║╚██████╔╝██║ ╚═╝ ██║
  ╚═══╝  ╚═╝╚═════╝ ╚══════╝ ╚═════╝     ╚══════╝ ╚═════╝ ╚═╝     ╚═╝
```

# YouTube Video Summarizer with Visual Timeline

This tool automatically creates comprehensive summaries of YouTube videos with both textual content and visual context. It:

1. 📝 Extracts the full transcript (either from YouTube captions or audio transcription)
2. 🖼️ Captures screenshots at regular intervals throughout the video
3. 📊 Creates a visual timeline with timestamps
4. 🤖 Uses Claude AI to generate an intelligent summary
5. 📄 Outputs both summary and visual timeline as files

## Features

- **Transcript Extraction**: Automatically fetches captions or transcribes audio
- **Visual Timeline**: Creates a grid of screenshots with timestamps
- **Intelligent Summarization**: Uses Claude API to create concise, accurate summaries
- **Visual Context**: Incorporates visual elements in the summary
- **Detailed Progress Reporting**: Shows the status of each step in real-time
- **Robust YouTube Processing**: Multiple fallback strategies for reliable video downloads and metadata extraction
- **Advanced Error Handling**: Retry mechanisms with exponential backoff to handle temporary failures
- **Multi-layer Fallback System**: Uses a combination of YouTube API, yt-dlp, direct HTTP requests, and Invidious API to ensure maximum reliability even when primary methods fail

## 🚀 Installation

### Prerequisites

- Python 3.8+
- [ffmpeg](https://ffmpeg.org/download.html) installed on your system
- Claude API key from Anthropic
- YouTube API key (optional, enhances metadata retrieval but not required)

### Setup with Virtual Environment

1. **Clone or download this repository**

   ```bash
   git clone https://github.com/rbur0425/youtubevideosummarizer.git
   cd video-summarizer
   ```

2. **Create and activate a virtual environment**

   On macOS and Linux:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

   On Windows:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Note: The initial installation might take several minutes due to the size of some packages.

4. **Download the Whisper base model**

   ```bash
   python -c "import whisper; whisper.load_model('base')"
   ```

5. **Set up your API keys** (Choose one method)

   - Create a `.env` file from the example:

     ```bash
     cp .env.example .env
     # Then edit the .env file with your API keys
     ```

   - As environment variables:

     ```bash
     # On macOS/Linux
     export ANTHROPIC_API_KEY=your-api-key-here
     export YOUTUBE_API_KEY=your-youtube-api-key-here

     # On Windows
     set ANTHROPIC_API_KEY=your-api-key-here
     set YOUTUBE_API_KEY=your-youtube-api-key-here
     ```

   - You can also enter the Claude API key when prompted by the program

## 📋 Usage

### Basic Usage

Run the script with a YouTube URL:

```bash
python youtube_summarizer.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

The program will:

1. Download the video content
2. Extract the transcript
3. Capture screenshots
4. Generate the summary
5. Save all output files to the `output` directory

### Command-Line Arguments

```bash
python youtube_summarizer.py [options]
```

Options:

- `--url URL`: YouTube video URL (required)
- `--api-key KEY`: Claude API key (optional if set as environment variable)
- `--output-dir DIR`: Output directory (default: "output")
- `--interval SECONDS`: Seconds between screenshots (default: 30)
- `--screenshots NUMBER`: Maximum number of screenshots (default: 10)
- `--grid-rows ROWS`: Rows in the visual timeline grid (default: 2)
- `--grid-cols COLS`: Columns in the visual timeline grid (default: 5)

Example with all options:

```bash
python youtube_summarizer.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --api-key "your-api-key" \
  --output-dir "my_summaries" \
  --interval 45 \
  --screenshots 12 \
  --grid-rows 3 \
  --grid-cols 4
```

### Interactive Mode

Simply run the script without arguments to enter interactive mode:

```bash
python youtube_summarizer.py
```

## 📊 Output Files

The program generates the following files in the output directory:

- **`visual_timeline.jpg`**: Grid of screenshots with timestamps
- **`transcript.txt`**: Full transcript of the video
- **`summary.txt`**: Comprehensive summary created by Claude

## ⚠️ Troubleshooting

- **FFmpeg not found**: Make sure ffmpeg is installed and in your PATH
- **Audio transcription errors**: Check if the video has audio content
- **API authentication errors**: Verify your Claude API key
- **Video download issues**: The tool implements multiple fallback methods, but some videos may still be restricted
- **YouTube API quota exceeded**: The tool will automatically fall back to alternate methods if the YouTube API quota is exceeded

## 📚 Technical Details

The program uses:

- **pytube** for downloading video content with advanced fallback mechanisms
- **YouTube Data API** for enhanced metadata retrieval (when available)
- **Whisper** for audio transcription
- **OpenCV** for frame extraction
- **Pillow** for image processing
- **Claude API** for AI-powered summarization
- **Backoff** for intelligent retry mechanisms

## 📄 License

This project is released under the MIT License - see the LICENSE file for details.

---

```
                       📹 Happy Summarizing! 📝
```
