# metadata_extractor.py
import json
import re
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import exifread
import piexif
import streamlit as st
import os
import subprocess

class ImageMetadataExtractor:
    """Utility class for extracting metadata from AI-generated images."""
    
    def __init__(self):
        self.ai_metadata_keys = {
            # Common AI generation metadata keys
            'prompt': ['prompt', 'positive_prompt', 'description', 'user_comment'],
            'negative_prompt': ['negative_prompt', 'negative'],
            'model': ['model', 'model_name', 'checkpoint', 'model_hash'],
            'sampler': ['sampler', 'sampling_method', 'scheduler'],
            'steps': ['steps', 'sampling_steps', 'num_inference_steps'],
            'cfg_scale': ['cfg_scale', 'guidance_scale', 'scale'],
            'seed': ['seed', 'noise_seed'],
            'size': ['size', 'width', 'height', 'resolution'],
            'software': ['software', 'generator', 'created_by'],
            'parameters': ['parameters', 'generation_data']
        }
    
    def extract_metadata(self, image_path):
        """Extract all available metadata from an image file."""
        try:
            metadata = {
                'file_info': self._get_file_info(image_path),
                'exif_data': {},
                'png_text': {},
                'ai_metadata': {},
                'raw_metadata': {}
            }
            
            # Extract EXIF data
            metadata['exif_data'] = self._extract_exif_data(image_path)
            
            # Extract PNG text chunks (common for AI-generated images)
            if str(image_path).lower().endswith('.png'):
                metadata['png_text'] = self._extract_png_text(image_path)
            
            # Extract Windows file properties
            metadata['windows_properties'] = self._extract_windows_properties(image_path)
            
            # Extract AI-specific metadata (AUTOMATIC1111, ComfyUI, etc.)
            with Image.open(image_path) as image:
                ai_metadata = self._extract_ai_metadata(image, metadata)
                if ai_metadata:
                    metadata['ai_metadata'] = ai_metadata
            
            # Store raw metadata for debugging
            metadata['raw_metadata'] = self._get_raw_metadata(image_path)
            
            # Parse additional AI-specific metadata and merge with existing
            additional_ai_metadata = self._parse_ai_metadata(metadata)
            if additional_ai_metadata:
                metadata['ai_metadata'].update(additional_ai_metadata)
            
            return metadata
            
        except Exception as e:
            st.error(f"Error extracting metadata from {image_path}: {str(e)}")
            return None
    
    def _get_file_info(self, image_path):
        """Get basic file information."""
        path = Path(image_path)
        stat = path.stat()
        
        with Image.open(image_path) as img:
            return {
                'filename': path.name,
                'size_bytes': stat.st_size,
                'format': img.format,
                'mode': img.mode,
                'dimensions': f"{img.width}x{img.height}",
                'width': img.width,
                'height': img.height
            }
    
    def _extract_exif_data(self, image_path):
        """Extract EXIF data from image using multiple methods."""
        exif_data = {}
        
        try:
            # Try with PIL first
            with Image.open(image_path) as img:
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        decoded_value = self._decode_exif_value(tag, value)
                        exif_data[tag] = decoded_value
                        
                # Also try getexif() method (newer PIL versions)
                elif hasattr(img, 'getexif'):
                    exif = img.getexif()
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        decoded_value = self._decode_exif_value(tag, value)
                        exif_data[tag] = decoded_value
                
                # Try to get info dict which might contain additional metadata
                if hasattr(img, 'info') and img.info:
                    for key, value in img.info.items():
                        if key not in exif_data:
                            exif_data[f"Info_{key}"] = str(value)
                        
        except Exception as e:
            print(f"PIL EXIF extraction failed: {e}")
            
        # Try with exifread as backup
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=True)
                for tag, value in tags.items():
                    if not tag.startswith('JPEGThumbnail'):
                        tag_clean = tag.replace('EXIF ', '').replace('Image ', '')
                        if tag_clean not in exif_data:
                            exif_data[tag_clean] = str(value)
        except Exception as e:
            print(f"exifread extraction failed: {e}")
            
        # Try with piexif for additional coverage
        try:
            exif_dict = piexif.load(str(image_path))
            for ifd_name, ifd in exif_dict.items():
                if isinstance(ifd, dict):
                    for tag_id, value in ifd.items():
                        try:
                            tag_name = piexif.TAGS[ifd_name][tag_id]["name"]
                            decoded_value = self._decode_exif_value(tag_name, value)
                            if tag_name not in exif_data:
                                exif_data[f"Piexif_{tag_name}"] = decoded_value
                        except (KeyError, TypeError):
                            continue
        except Exception as e:
            print(f"piexif extraction failed: {e}")
            
        return exif_data
    
    def _extract_windows_properties(self, image_path):
        """Extract Windows file properties including Comments."""
        properties = {}
        
        try:
            # Try using PowerShell to get extended file properties
            # Use a more comprehensive approach to find the Comments property
            ps_command = f'''
            $file = Get-Item "{image_path}"
            $shell = New-Object -ComObject Shell.Application
            $folder = $shell.Namespace($file.DirectoryName)
            $item = $folder.ParseName($file.Name)
            
            # Try multiple property indices that might contain comments/description
            $propertyIndices = 0..300
            foreach ($i in $propertyIndices) {{
                $propName = $folder.GetDetailsOf($null, $i)
                $propValue = $folder.GetDetailsOf($item, $i)
                if ($propValue -and $propValue.Trim() -ne "" -and $propName) {{
                    $cleanName = $propName -replace "[^a-zA-Z0-9 ]", ""
                    if ($cleanName -match "(Comment|Description|Title|Subject|Tag|Author|Keyword)" -or $i -in @(18,20,21,22,24,25,26,27,28)) {{
                        Write-Output "$cleanName ($i): $propValue"
                    }}
                }}
            }}
            '''
            
            result = subprocess.run(['powershell', '-Command', ps_command], 
                                  capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line and line.strip():
                        key, value = line.split(':', 1)
                        clean_key = key.strip().replace(' ', '_')
                        properties[f"Windows_{clean_key}"] = value.strip()
                        
        except Exception as e:
            print(f"Windows properties extraction failed: {e}")
            
        return properties

    def _extract_ai_metadata(self, image, metadata):
        """Extract AI-specific metadata from AUTOMATIC1111, ComfyUI, etc."""
        ai_data = {}
        
        # Priority 1: Check Windows properties for AI metadata (usually best formatted)
        if 'windows_properties' in metadata and metadata['windows_properties']:
            # Look for any Windows property that contains substantial AI metadata
            for key, value in metadata['windows_properties'].items():
                if value and len(str(value)) > 50:  # Substantial content
                    value_str = str(value)
                    
                    # Check if it looks like AI generation metadata (more comprehensive check)
                    value_lower = value_str.lower()
                    ai_keywords = ['steps:', 'sampler:', 'cfg scale:', 'seed:', 'negative prompt:', 'lora:', 
                                 'model:', 'denoising strength:', 'clip skip:', 'hires upscale:', 'version:']
                    
                    if any(keyword in value_lower for keyword in ai_keywords):
                        ai_data['user_comment'] = value_str
                        ai_data['source'] = f'Windows Properties ({key})'
                        return ai_data  # Return immediately with the best source
        
        # Priority 2: Check for AUTOMATIC1111 metadata in UserComment (if not found in Windows properties)
        if not ai_data and 'exif_data' in metadata and 'UserComment' in metadata['exif_data']:
            user_comment = metadata['exif_data']['UserComment']
            if user_comment and len(str(user_comment)) > 10:
                # Try to decode the UserComment properly
                decoded_comment = self._decode_ai_usercomment(user_comment)
                if decoded_comment:
                    ai_data['user_comment'] = decoded_comment
                    ai_data['source'] = 'EXIF UserComment'
        
        # Priority 3: Check PNG text chunks for AI metadata
        if not ai_data and 'png_text' in metadata:
            for key, value in metadata['png_text'].items():
                if key.lower() in ['parameters', 'workflow', 'prompt', 'negative_prompt']:
                    ai_data[key] = value
                    ai_data['source'] = 'PNG Text Chunk'
                    break
        
        # Priority 4: Check for ComfyUI workflow in PNG
        if not ai_data and hasattr(image, 'text') and image.text:
            for key, value in image.text.items():
                if 'workflow' in key.lower() or 'prompt' in key.lower():
                    ai_data[key] = value
                    ai_data['source'] = 'PNG Metadata'
                    break
        
        return ai_data

    def _decode_ai_usercomment(self, user_comment):
        """Decode AI-specific UserComment with multiple methods."""
        if not user_comment:
            return None
        
        # Method 1: Check if it's already properly formatted (like from Windows properties)
        if isinstance(user_comment, str) and len(user_comment) > 10:
            # Check if it contains mostly readable ASCII and proper formatting
            ascii_count = sum(1 for c in user_comment if ord(c) < 128)
            if ascii_count / len(user_comment) > 0.8:  # 80% ASCII
                # Check for typical AI metadata patterns
                if any(keyword in user_comment.lower() for keyword in ['steps:', 'sampler:', 'cfg scale:', 'seed:', 'negative prompt:']):
                    return user_comment  # Already properly formatted
        
        # Convert to string if it's not already
        comment_str = str(user_comment)
        
        # Method 2: Try standard UTF-16 decoding
        try:
            if isinstance(user_comment, str):
                bytes_value = user_comment.encode('latin-1')
                decoded = bytes_value.decode('utf-16le', errors='ignore').strip('\x00')
                if len(decoded) > 10:
                    return decoded
        except:
            pass
        
        return None
    


    def _decode_exif_value(self, tag, value):
        """Decode EXIF values, especially binary data like UserComment."""
        try:
            # Handle UserComment specifically
            if tag == 'UserComment':
                if isinstance(value, bytes):
                    # UserComment often starts with encoding info
                    if value.startswith(b'UNICODE\x00'):
                        # Remove UNICODE header and decode
                        return value[8:].decode('utf-16le', errors='ignore').strip('\x00')
                    elif value.startswith(b'ASCII\x00\x00\x00'):
                        # Remove ASCII header
                        return value[8:].decode('ascii', errors='ignore').strip('\x00')
                    else:
                        # Try multiple encodings
                        for encoding in ['utf-8', 'utf-16le', 'utf-16be', 'latin-1', 'cp1252']:
                            try:
                                decoded = value.decode(encoding, errors='ignore').strip('\x00')
                                if decoded and len(decoded) > 10:  # Reasonable length check
                                    return decoded
                            except:
                                continue
                        return value.decode('latin-1', errors='ignore').strip('\x00')
                elif isinstance(value, str):
                    # Already a string, try standard UTF-16 decoding if needed
                    try:
                        # Try encoding as latin-1 then decoding as UTF-16 for malformed strings
                        bytes_value = value.encode('latin-1')
                        decoded = bytes_value.decode('utf-16le', errors='ignore').strip('\x00')
                        if decoded and len(decoded) > 10:
                            return decoded
                    except:
                        pass
                    return value
            
            # Handle other binary data
            elif isinstance(value, bytes):
                # Try to decode as text if it looks like text
                if all(32 <= b <= 126 or b in [9, 10, 13] for b in value[:50]):  # Check if printable ASCII
                    return value.decode('ascii', errors='ignore').strip('\x00')
                else:
                    # Return hex representation for binary data
                    return f"[Binary data: {len(value)} bytes]"
            
            # Handle tuples (like GPS coordinates)
            elif isinstance(value, tuple):
                return ', '.join(str(v) for v in value)
            
            # Convert everything else to string
            else:
                return str(value)
                
        except Exception:
            # Fallback to string conversion
            return str(value)
    
    def _extract_png_text(self, image_path):
        """Extract text chunks from PNG files (common for AI metadata)."""
        png_text = {}
        
        try:
            with Image.open(image_path) as img:
                if hasattr(img, 'text'):
                    png_text.update(img.text)
                
                # Also check info attribute
                if hasattr(img, 'info'):
                    for key, value in img.info.items():
                        if isinstance(value, str):
                            png_text[key] = value
        except Exception:
            pass
        
        return png_text
    
    def _get_raw_metadata(self, image_path):
        """Get raw metadata for debugging purposes."""
        raw_data = {}
        
        try:
            with Image.open(image_path) as img:
                if hasattr(img, 'info'):
                    raw_data['pil_info'] = dict(img.info)
        except Exception:
            pass
        
        return raw_data
    
    def _parse_ai_metadata(self, metadata):
        """Parse and organize AI-specific metadata."""
        ai_data = {}
        all_metadata = {**metadata['exif_data'], **metadata['png_text']}
        
        # Look for common AI metadata patterns
        for category, possible_keys in self.ai_metadata_keys.items():
            for key in possible_keys:
                # Check exact matches (case-insensitive)
                for meta_key, meta_value in all_metadata.items():
                    if key.lower() == meta_key.lower():
                        ai_data[category] = meta_value
                        break
                
                # Check if key is contained in metadata key
                if category not in ai_data:
                    for meta_key, meta_value in all_metadata.items():
                        if key.lower() in meta_key.lower():
                            ai_data[category] = meta_value
                            break
        
        # Special handling for parameters field (common in Stable Diffusion)
        if 'parameters' in all_metadata:
            ai_data.update(self._parse_parameters_string(all_metadata['parameters']))
        
        # Look for JSON-formatted metadata
        for key, value in all_metadata.items():
            if self._is_json_string(value):
                try:
                    json_data = json.loads(value)
                    if isinstance(json_data, dict):
                        ai_data[f'json_{key}'] = json_data
                except:
                    pass
        
        return ai_data
    
    def _parse_parameters_string(self, params_string):
        """Parse Stable Diffusion style parameters string."""
        parsed = {}
        
        try:
            # Common pattern: "prompt\nNegative prompt: negative\nSteps: 20, Sampler: ..."
            lines = params_string.split('\n')
            
            if lines:
                # First line is usually the prompt
                parsed['prompt'] = lines[0].strip()
            
            for line in lines[1:]:
                line = line.strip()
                
                # Handle "Negative prompt:" line
                if line.startswith('Negative prompt:'):
                    parsed['negative_prompt'] = line.replace('Negative prompt:', '').strip()
                
                # Handle comma-separated parameters
                elif ':' in line and ',' in line:
                    parts = line.split(',')
                    for part in parts:
                        if ':' in part:
                            key, value = part.split(':', 1)
                            parsed[key.strip().lower()] = value.strip()
        except Exception:
            pass
        
        return parsed
    
    def _is_json_string(self, text):
        """Check if a string contains valid JSON."""
        if not isinstance(text, str):
            return False
        
        text = text.strip()
        return (text.startswith('{') and text.endswith('}')) or \
               (text.startswith('[') and text.endswith(']'))
    
    def format_metadata_for_display(self, metadata):
        """Format metadata for user-friendly display."""
        if not metadata:
            return "No metadata available"
        
        formatted_sections = []
        
        # File Information
        if metadata.get('file_info'):
            file_info = metadata['file_info']
            formatted_sections.append({
                'title': '📁 File Information',
                'content': {
                    'Filename': file_info.get('filename', 'Unknown'),
                    'Format': file_info.get('format', 'Unknown'),
                    'Dimensions': file_info.get('dimensions', 'Unknown'),
                    'File Size': self._format_file_size(file_info.get('size_bytes', 0)),
                    'Color Mode': file_info.get('mode', 'Unknown')
                }
            })
        
        # AI Generation Data
        if metadata.get('ai_metadata'):
            ai_data = metadata['ai_metadata']
            ai_content = {}
            
            # Organize AI metadata with better labels
            label_mapping = {
                'user_comment': 'AI Prompt/Parameters',
                'prompt': 'Prompt',
                'negative_prompt': 'Negative Prompt',
                'model': 'Model',
                'sampler': 'Sampler',
                'steps': 'Steps',
                'cfg_scale': 'CFG Scale',
                'seed': 'Seed',
                'software': 'Software',
                'source': 'Metadata Source'
            }
            
            for key, value in ai_data.items():
                if key.startswith('json_'):
                    # Handle JSON metadata
                    json_data = value
                    if isinstance(json_data, dict):
                        for json_key, json_value in json_data.items():
                            ai_content[f"{key.replace('json_', '').title()} - {json_key}"] = str(json_value)
                else:
                    label = label_mapping.get(key, key.replace('_', ' ').title())
                    ai_content[label] = str(value)
            
            if ai_content:
                formatted_sections.append({
                    'title': '🤖 AI Generation Data',
                    'content': ai_content
                })
        
        # Technical Metadata (EXIF)
        if metadata.get('exif_data'):
            exif_data = metadata['exif_data']
            # Filter out common technical EXIF data that's relevant
            relevant_exif = {}
            relevant_keys = [
                'DateTime', 'Software', 'Artist', 'Copyright', 'ImageDescription',
                'UserComment', 'XPComment', 'XPKeywords', 'XPSubject', 'XPTitle', 'XPAuthor',
                'Make', 'Model', 'Comment', 'Comments', 'Description', 'Title', 'Subject',
                'Keywords', 'Author', 'Creator', 'DocumentName', 'PageName', 'HostComputer'
            ]
            
            for key, value in exif_data.items():
                if any(relevant_key in key for relevant_key in relevant_keys):
                    # Filter out empty, meaningless, or binary values
                    if self._is_meaningful_value(value):
                        relevant_exif[key] = value
            
            # Debug: Show all EXIF data if no relevant data found
            if not relevant_exif and exif_data:
                debug_exif = {}
                for key, value in exif_data.items():
                    if self._is_meaningful_value(value):
                        debug_exif[f"DEBUG_{key}"] = str(value)[:100]  # Truncate for display
                
                if debug_exif:
                    formatted_sections.append({
                        'title': '🔍 All Available Metadata (Debug)',
                        'content': debug_exif
                    })
            elif relevant_exif:
                formatted_sections.append({
                    'title': '🔧 Technical Metadata',
                    'content': relevant_exif
                })
        
        # Windows Properties (Comments, etc.)
        if metadata.get('windows_properties'):
            windows_props = metadata['windows_properties']
            if windows_props:
                formatted_sections.append({
                    'title': '🪟 Windows Properties',
                    'content': windows_props
                })
        
        # PNG Text Data
        if metadata.get('png_text'):
            png_data = metadata['png_text']
            if png_data:
                formatted_sections.append({
                    'title': '📝 PNG Text Data',
                    'content': png_data
                })
        
        return formatted_sections
    
    def _is_meaningful_value(self, value):
        """Check if a metadata value is meaningful and should be displayed."""
        if not value:
            return False
        
        value_str = str(value).strip()
        
        # Filter out empty or whitespace-only values
        if not value_str:
            return False
        
        # Filter out common meaningless values
        meaningless_values = [
            '', '0', 'None', 'null', 'undefined', '[Binary data:', 
            '\x00', 'Unknown', 'N/A', 'Not Available'
        ]
        
        for meaningless in meaningless_values:
            if value_str.startswith(meaningless):
                return False
        
        # Filter out values that are just null bytes or control characters
        if all(ord(c) < 32 for c in value_str if c):
            return False
        
        # Filter out very long binary-looking strings
        if len(value_str) > 200 and all(c in '0123456789abcdefABCDEF\\x' for c in value_str[:50]):
            return False
        
        return True
    
    def _format_file_size(self, size_bytes):
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"