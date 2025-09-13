#!/usr/bin/env python3
"""
Background Remover Tool

A tool for removing white/light colored backgrounds from images using flood fill algorithm.
Supports batch processing and various customization options.

Author: Tanakorn Onlamoon
License: MIT
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


class BackgroundRemover:
    """A class for removing backgrounds from images using flood fill."""

    def __init__(
        self,
        lower_diff: Tuple[int, int, int] = (10, 10, 10),
        upper_diff: Tuple[int, int, int] = (10, 10, 10),
        start_points: Optional[List[Tuple[int, int]]] = None,
        auto_crop: bool = True,
    ):
        """
        Initialize the BackgroundRemover.

        Args:
            lower_diff: Lower threshold for flood fill (BGR values)
            upper_diff: Upper threshold for flood fill (BGR values)
            start_points: List of starting points for flood fill. Defaults to corners.
            auto_crop: Whether to automatically crop the result to remove empty space
        """
        self.lower_diff = lower_diff
        self.upper_diff = upper_diff
        self.start_points = start_points
        self.auto_crop = auto_crop
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _validate_image_path(self, path: Union[str, Path]) -> Path:
        """Validate that the image path exists and is a valid image file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        
        return path

    def _load_image(self, input_path: Path) -> np.ndarray:
        """Load image with proper alpha channel handling."""
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError(f"Could not load image: {input_path}")

        # Ensure BGRA (4-channel)
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:  # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        elif img.shape[2] == 4:  # Already BGRA
            pass
        else:
            raise ValueError(f"Unsupported image format with {img.shape[2]} channels")

        return img

    def _get_flood_fill_points(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """Get starting points for flood fill operation."""
        if self.start_points:
            return self.start_points

        h, w = img.shape[:2]
        # Default to all four corners
        return [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]

    def _apply_flood_fill(self, img: np.ndarray) -> np.ndarray:
        """Apply flood fill to remove background."""
        h, w = img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        bgr = img[:, :, :3].copy()
        result = img.copy()

        points = self._get_flood_fill_points(img)
        
        for point in points:
            x, y = point
            if 0 <= x < w and 0 <= y < h:
                # Reset mask for each flood fill operation
                temp_mask = np.zeros((h + 2, w + 2), np.uint8)
                
                cv2.floodFill(
                    bgr, temp_mask, (x, y),
                    (0, 0, 0),  # dummy color
                    self.lower_diff,
                    self.upper_diff,
                    flags=cv2.FLOODFILL_FIXED_RANGE
                )
                
                # Combine masks
                mask = cv2.bitwise_or(mask, temp_mask)

        # Apply transparency to flood-filled areas
        filled_area = mask[1:h+1, 1:w+1] == 1
        result[filled_area] = (0, 0, 0, 0)  # Transparent

        return result

    def _crop_image(self, img: np.ndarray) -> np.ndarray:
        """Crop image to remove transparent borders."""
        if not self.auto_crop:
            return img

        # Convert to PIL for easier cropping
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
        bbox = pil_img.getbbox()
        
        if bbox:
            pil_img = pil_img.crop(bbox)
            # Convert back to numpy array
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
        
        return img

    def remove_background(
        self, 
        input_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Remove background from a single image.

        Args:
            input_path: Path to input image
            output_path: Path to output image (optional)

        Returns:
            Path to the output image
        """
        input_path = self._validate_image_path(input_path)
        
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_no_bg.png"
        else:
            output_path = Path(output_path)

        self.logger.info(f"Processing: {input_path}")

        try:
            # Load and process image
            img = self._load_image(input_path)
            result = self._apply_flood_fill(img)
            result = self._crop_image(result)

            # Save result
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
            pil_img.save(output_path, "PNG")

            self.logger.info(f"Saved: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to process {input_path}: {e}")
            raise

    def batch_process(
        self, 
        input_dir: Union[str, Path], 
        output_dir: Optional[Union[str, Path]] = None,
        pattern: str = "*"
    ) -> List[Path]:
        """
        Process multiple images in a directory.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory for output images (optional)
            pattern: File pattern to match (default: "*")

        Returns:
            List of paths to processed images
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        if output_dir is None:
            output_dir = input_dir / "processed"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all image files
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in input_dir.glob(pattern) 
            if f.suffix.lower() in valid_extensions
        ]

        if not image_files:
            self.logger.warning(f"No image files found in {input_dir}")
            return []

        processed_files = []
        for img_file in image_files:
            try:
                output_path = output_dir / f"{img_file.stem}_no_bg.png"
                result_path = self.remove_background(img_file, output_path)
                processed_files.append(result_path)
            except Exception as e:
                self.logger.error(f"Failed to process {img_file}: {e}")
                continue

        self.logger.info(f"Processed {len(processed_files)}/{len(image_files)} images")
        return processed_files


def main():
    """Command-line interface for the background remover."""
    parser = argparse.ArgumentParser(
        description="Remove white/light backgrounds from images using flood fill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python background_remover.py input.png
  python background_remover.py input.png -o output.png
  python background_remover.py --batch ./images/ -od ./processed/
  python background_remover.py input.png --lower-diff 20 20 20 --no-crop
        """
    )

    parser.add_argument(
        "input", 
        help="Input image file or directory (for batch processing)"
    )
    
    parser.add_argument(
        "-o", "--output", 
        help="Output file path (for single image)"
    )
    
    parser.add_argument(
        "--batch", 
        action="store_true",
        help="Process all images in input directory"
    )
    
    parser.add_argument(
        "-od", "--output-dir", 
        help="Output directory (for batch processing)"
    )
    
    parser.add_argument(
        "--pattern", 
        default="*",
        help="File pattern for batch processing (default: *)"
    )
    
    parser.add_argument(
        "--lower-diff", 
        nargs=3, 
        type=int, 
        default=[10, 10, 10],
        metavar=("B", "G", "R"),
        help="Lower threshold for flood fill (BGR values, default: 10 10 10)"
    )
    
    parser.add_argument(
        "--upper-diff", 
        nargs=3, 
        type=int, 
        default=[10, 10, 10],
        metavar=("B", "G", "R"),
        help="Upper threshold for flood fill (BGR values, default: 10 10 10)"
    )
    
    parser.add_argument(
        "--start-points", 
        nargs="+", 
        type=str,
        help="Starting points for flood fill as 'x,y' pairs (e.g., '0,0' '100,50')"
    )
    
    parser.add_argument(
        "--no-crop", 
        action="store_true",
        help="Disable automatic cropping"
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse start points
    start_points = None
    if args.start_points:
        try:
            start_points = [
                tuple(map(int, point.split(','))) 
                for point in args.start_points
            ]
        except ValueError:
            print("Error: Invalid start points format. Use 'x,y' format.")
            sys.exit(1)

    # Create background remover
    remover = BackgroundRemover(
        lower_diff=tuple(args.lower_diff),
        upper_diff=tuple(args.upper_diff),
        start_points=start_points,
        auto_crop=not args.no_crop
    )

    try:
        if args.batch:
            # Batch processing
            processed = remover.batch_process(
                args.input, 
                args.output_dir, 
                args.pattern
            )
            print(f"Successfully processed {len(processed)} images")
        else:
            # Single image processing
            result = remover.remove_background(args.input, args.output)
            print(f"Successfully processed: {result}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()