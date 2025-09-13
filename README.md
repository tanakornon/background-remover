# Background Remover Tool

A Python tool for automatically removing white and light-colored backgrounds from images using the flood fill algorithm.

## Features

- Remove white/light backgrounds from images
- Support for single image and batch processing
- Customizable flood fill thresholds (lower_diff and upper_diff)
- Specify starting points for more control
- Automatic cropping of transparent borders
- Works with common image formats: PNG, JPG, JPEG, BMP, TIFF, WEBP

## Installation

### Prerequisites

- Python 3.6 or higher
- OpenCV
- PIL (Pillow)
- NumPy

### Install Dependencies

```bash
pip install opencv-python pillow numpy
```

### Download the Tool

```bash
git clone https://github.com/yourusername/background-remover-tool.git
cd background-remover-tool
```

## Quick Start

### Single Image Processing

```bash
# Basic usage - removes background and saves as [filename]_no_bg.png
python background_remover.py input.png

# Specify output file
python background_remover.py input.jpg -o output.png
```

### Batch Processing

```bash
# Process all images in a directory
python background_remover.py ./images/ --batch

# Process with custom output directory
python background_remover.py ./images/ --batch -od ./processed/

# Process specific file types
python background_remover.py ./images/ --batch --pattern "*.jpg"
```

## Advanced Usage

### Custom Sensitivity Settings

```bash
# Increase sensitivity for subtle backgrounds
python background_remover.py input.png --lower-diff 5 5 5 --upper-diff 5 5 5

# Decrease sensitivity for varied backgrounds
python background_remover.py input.png --lower-diff 25 25 25 --upper-diff 25 25 25
```

### Custom Start Points

```bash
# Define specific flood fill starting points
python background_remover.py input.png --start-points "0,0" "50,100" "200,300"
```

### Other Options

```bash
# Disable auto-cropping
python background_remover.py input.png --no-crop

# Enable verbose logging
python background_remover.py input.png -v
```

## Python API Usage

```python
from background_remover import BackgroundRemover

# Initialize with default settings
remover = BackgroundRemover()

# Process single image
output_path = remover.remove_background("input.png", "output.png")

# Batch process directory
processed_files = remover.batch_process("./images/", "./processed/")

# Custom configuration
remover = BackgroundRemover(
    lower_diff=(5, 5, 5),        # More sensitive
    upper_diff=(5, 5, 5),        # More sensitive
    start_points=[(0, 0), (100, 50)],  # Custom start points
    auto_crop=True               # Enable auto-cropping
)
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input` | Input image file or directory | Required |
| `-o, --output` | Output file path | `[input]_no_bg.png` |
| `--batch` | Enable batch processing mode | False |
| `-od, --output-dir` | Output directory for batch processing | `./processed/` |
| `--pattern` | File pattern for batch processing | `*` |
| `--lower-diff` | Lower threshold for flood fill (B G R) | `10 10 10` |
| `--upper-diff` | Upper threshold for flood fill (B G R) | `10 10 10` |
| `--start-points` | Custom flood fill start points | Corners |
| `--no-crop` | Disable automatic cropping | False |
| `-v, --verbose` | Enable verbose logging | False |

## How It Works

The tool uses OpenCV's flood fill algorithm to identify and remove backgrounds:

1. **Image Loading**: Loads the image and converts it to BGRA format for transparency support
2. **Flood Fill**: Starting from corner points (or custom points), flood fill identifies connected pixels within the specified color threshold
3. **Background Removal**: Identified background pixels are made transparent
4. **Auto-Cropping**: Removes unnecessary transparent borders (optional)
5. **Output**: Saves the result as a PNG with transparency

## Troubleshooting

### Common Issues

**Background not completely removed**:

- Increase the threshold values (`--lower-diff` and `--upper-diff`)
- Add custom start points with `--start-points`

**Too much of the image removed**:

- Decrease the threshold values
- Check if the background is truly uniform

**Memory errors with large images**:

- Process images individually rather than in batch
- Resize images before processing if possible

### Error Messages

- `Image file not found`: Check file path and permissions
- `Unsupported image format`: Use supported formats (PNG, JPG, etc.)
- `Invalid start points format`: Use 'x,y' format for coordinates

## Examples

### Product Photography

```bash
python background_remover.py product.jpg --lower-diff 15 15 15 --upper-diff 15 15 15
```

### Logo/Icon Creation

```bash
python background_remover.py logo.png --start-points "0,0" --no-crop
```

### Batch E-commerce Processing

```bash
python background_remover.py ./product_photos/ --batch -od ./clean_products/ --pattern "*.jpg"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
