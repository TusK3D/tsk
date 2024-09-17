import os
from PIL import Image

def load_image_sequence(path, channel = "RGB"):
    """
    Load an Image seq from the specified path.
    
    Args:
    path(str): Path to the directory containing image sequence.
    channel(str): "L" or "RGB" 
    Returns:
    List[image] : list of image arrays (C,W,H)
    """
    image_files = sorted([f for f in os.listdir(path) if f.endswith(('png', 'jpg', 'jpeg', 'exr'))])
    images = []
    
    for image_file in image_files:
        image_path = os.path.join(path, image_file)
        image = Image.open(image_path).convert(channel)   #Convert Image to Luminance format(1,80,80)
        images.append(image)
    
    return images





    



    

    


        
        
