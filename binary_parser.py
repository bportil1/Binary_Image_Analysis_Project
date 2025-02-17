import numpy as np
import struct
import scipy.stats
from PIL import Image 

def calculate_entropy(byte_array, window_size=256):
    """
    Calculate the Shannon entropy for a given byte sequence.
    A higher entropy value indicates randomness (e.g., compressed or encrypted data).
    """
    entropy_values = []
    for i in range(0, len(byte_array), window_size):
        chunk = byte_array[i:i + window_size]
        if len(chunk) == 0:
            continue
        _, counts = np.unique(chunk, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        entropy_values.append(entropy)
    return entropy_values

def detect_section_boundaries(byte_array):
    """
    Use entropy changes and common markers to determine possible section boundaries.
    """
    entropy_values = calculate_entropy(byte_array)
    boundaries = [0]  # Start at 0

    # Find sudden drops in entropy that may indicate section transitions
    for i in range(1, len(entropy_values)):
        if abs(entropy_values[i] - entropy_values[i - 1]) > 1.5:  # Threshold for change
            boundaries.append(i * 256)  # Convert window index to byte position

    boundaries.append(len(byte_array))  # End at the file size
    return boundaries, entropy_values

def classify_sections(byte_array, boundaries):
    """
    Classify sections based on entropy and known structures.
    """
    sections = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        segment = byte_array[start:end]
        
        entropy = scipy.stats.entropy(np.bincount(segment, minlength=256), base=2)
        
        if entropy < 3.0:
            section_type = "Header/Metadata (Low entropy)"
        elif entropy > 6.5:
            section_type = "High-Entropy (Possibly Encrypted/Compressed)"
        elif len(segment) > 100 and b'\x00' in segment[:20]:
            section_type = "Padding/Alignment"
        else:
            section_type = "Structured Data"
        
        sections.append((start, end, section_type, entropy))
    
    return sections

def parse_binary_file(file_path):
    """
    Parse the binary file and classify its sections heuristically.
    """
    with open(file_path, "rb") as f:
        byte_array = np.frombuffer(f.read(), dtype=np.uint8)

    boundaries, entropy_values = detect_section_boundaries(byte_array)
    sections = classify_sections(byte_array, boundaries)

    return sections, entropy_values

def compute_byte_histogram(byte_array):
    """
    Compute a histogram of byte frequencies.
    """
    histogram, _ = np.histogram(byte_array, bins=256, range=(0, 256))
    return histogram

def compute_fourier_transform(byte_array):
    """
    Compute the Fourier transform of the binary data.
    Used to detect periodic structures.
    """
    transformed = np.abs(fft(byte_array))  # Magnitude of FFT
    return transformed[:len(transformed) // 2]  

def generate_visualization(byte_array, sections, padding_size=10):
    """
    Generate an image representation of the binary file with padding between sections.
    
    Parameters:
    - byte_array: The original binary data.
    - sections: Parsed file sections with classifications.
    - padding_size: Number of pixels to insert between sections.
    
    Returns:
    - An image where different sections are visually distinguishable.
    """
    image_rows = []  # Store each row (section + padding)
    
    for start, end, section_type, _ in sections:
        segment = byte_array[start:end]

        # Normalize values for grayscale visualization (0-255)
        grayscale_segment = np.array(segment).astype(np.uint8).reshape(-1, 1)
        
        # Map section types to brightness levels
        if "Metadata" in section_type:
            grayscale_segment[:] = 50  # Dark gray
        elif "Structured Data" in section_type:
            grayscale_segment[:] = 150  # Medium gray
        elif "High-Entropy" in section_type:
            grayscale_segment[:] = 230  # Light gray
        elif "Padding" in section_type:
            grayscale_segment[:] = 0  # Black
        
        image_rows.append(grayscale_segment)
        
        # Add padding (white pixels)
        padding = np.full((padding_size, 1), 255, dtype=np.uint8)
        image_rows.append(padding)
    
    # Stack all sections vertically
    final_image = np.vstack(image_rows)
    
    # Convert to PIL image (expand horizontally for visibility)
    img = Image.fromarray(final_image.repeat(5, axis=1))  # Widen for better visibility
  
    return img

