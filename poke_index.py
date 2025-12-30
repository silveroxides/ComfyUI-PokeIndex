import json
import os
import random
import requests
import torch
import numpy as np
from PIL import Image
import folder_paths
from comfy_api.latest import IO

# CDN URL for fusion sprites
IMAGE_CDN_URL = "https://ifd-spaces.sfo2.cdn.digitaloceanspaces.com/custom"


class PokeIndex(IO.ComfyNode):
    """
    ComfyUI node for loading Pokémon Infinite Fusion sprites.
    
    Data files (fusion_filenames.json, fusion_snippets.json) should be in the same
    directory as this node. Images are cached in ComfyUI's input/sprites folder.
    
    JSON structure:
    - fusion_filenames.json: {"1": "1_bulbasaur", "2": "1.1_bulbaizard", ...}
      Key = sequential index, Value = "fusionID_name"
    - fusion_snippets.json: {"1_bulbasaur": "description...", ...}
      Key = filename (matches value from filenames), Value = description
    """
    
    _filenames_cache = {}
    _snippets_cache = {}
    _filenames_path = os.path.join(os.path.dirname(__file__), "fusion_filenames.json")
    _snippets_path = os.path.join(os.path.dirname(__file__), "fusion_snippets.json")

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PokeIndex",
            display_name="Poke Index Loader",
            category="PokeIndex",
            inputs=[
                IO.Combo.Input("mode", options=["Random", "Index", "Search"], default="Random"),
                IO.Int.Input("index_id", default=1, min=1, max=999999),
                IO.String.Input("search_query", default=""),
                IO.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                IO.Boolean.Input("download_missing", default=True),
            ],
            outputs=[
                IO.Image.Output(display_name="Image"),
                IO.Mask.Output(display_name="Mask"),
                IO.String.Output(display_name="Filename"),
                IO.String.Output(display_name="Snippet"),
            ],
        )

    @classmethod
    def _load_data(cls):
        """Load JSON data files into cache if not already loaded."""
        if not cls._filenames_cache:
            if os.path.exists(cls._filenames_path):
                with open(cls._filenames_path, 'r', encoding='utf-8') as f:
                    cls._filenames_cache.update(json.load(f))
        
        if not cls._snippets_cache:
            if os.path.exists(cls._snippets_path):
                with open(cls._snippets_path, 'r', encoding='utf-8') as f:
                    cls._snippets_cache.update(json.load(f))

    @staticmethod
    def _download_image(fusion_id: str, target_path: str) -> bool:
        """
        Download a fusion sprite from the CDN.
        
        Args:
            fusion_id: The fusion ID (e.g., "1.6" or "1")
            target_path: Full path where the image should be saved
            
        Returns:
            True if download succeeded, False otherwise
        """
        url = f"{IMAGE_CDN_URL}/{fusion_id}.png"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://infinitefusiondex.com/"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                with open(target_path, 'wb') as f:
                    f.write(response.content)
                return True
            elif response.status_code == 404:
                # Not all fusions have custom sprites - this is expected
                pass
            else:
                print(f"[PokeIndex] Download failed ({response.status_code}): {url}")
        except Exception as e:
            print(f"[PokeIndex] Download error: {e}")
        return False

    @classmethod
    def execute(cls, mode, index_id, search_query, seed, download_missing) -> IO.NodeOutput:
        cls._load_data()
        
        # Select entry based on mode
        selected_key = None
        
        if mode == "Random":
            if cls._filenames_cache:
                random.seed(seed)
                keys = list(cls._filenames_cache.keys())
                selected_key = random.choice(keys)
        
        elif mode == "Index":
            # Direct lookup by string key, or positional if key not found
            str_key = str(index_id)
            if str_key in cls._filenames_cache:
                selected_key = str_key
            elif cls._filenames_cache:
                # Fallback to positional index
                keys = list(cls._filenames_cache.keys())
                idx = (index_id - 1) % len(keys)
                selected_key = keys[idx]
                
        elif mode == "Search":
            query = search_query.lower()
            for k, filename in cls._filenames_cache.items():
                # Search in filename and snippet
                snippet = cls._snippets_cache.get(filename, "")
                if query in filename.lower() or query in snippet.lower():
                    selected_key = k
                    break
        
        # Fallback if no selection made
        if selected_key is None:
            if cls._filenames_cache:
                selected_key = random.choice(list(cls._filenames_cache.keys()))
            else:
                return IO.NodeOutput(
                    torch.zeros((1, 64, 64, 3)),
                    torch.zeros((1, 64, 64)),
                    "No Data",
                    "No data loaded"
                )

        # Get filename and snippet
        # filename format: "1.6_bulbaizard" where "1.6" is the fusion_id
        filename = cls._filenames_cache.get(selected_key, "unknown")
        snippet = cls._snippets_cache.get(filename, "No description available")
        
        # Extract fusion_id from filename (everything before first underscore)
        fusion_id = filename.split('_')[0] if '_' in filename else filename
        
        # Setup sprite cache directory
        sprites_dir = os.path.join(folder_paths.get_input_directory(), "sprites")
        os.makedirs(sprites_dir, exist_ok=True)
        
        # Image path uses fusion_id, not full filename (CDN uses ID only)
        image_path = os.path.join(sprites_dir, f"{fusion_id}.png")
        
        # Download if missing and enabled
        if not os.path.exists(image_path) and download_missing:
            cls._download_image(fusion_id, image_path)
        
        # Load image
        image_tensor = None
        mask_tensor = None
        
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                
                # Extract alpha channel as mask (inverted: transparent=1.0, opaque=0.0)
                if 'A' in img.getbands():
                    alpha = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                    mask_tensor = torch.from_numpy(1.0 - alpha).unsqueeze(0)
                elif img.mode == 'P' and 'transparency' in img.info:
                    alpha = np.array(img.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                    mask_tensor = torch.from_numpy(1.0 - alpha).unsqueeze(0)
                
                # Convert to RGB tensor
                img_rgb = img.convert("RGB")
                img_np = np.array(img_rgb).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W, 3]
                
                # Default mask if none extracted
                if mask_tensor is None:
                    mask_tensor = torch.zeros((1, img_np.shape[0], img_np.shape[1]))
                    
            except Exception as e:
                print(f"[PokeIndex] Failed to load image: {e}")
        
        # Return black image if loading failed
        if image_tensor is None:
            image_tensor = torch.zeros((1, 64, 64, 3))
            mask_tensor = torch.zeros((1, 64, 64))
            
        return IO.NodeOutput(image_tensor, mask_tensor, filename, snippet)
