#!/usr/bin/env python3
"""
Download Sample Medical Images for Smart Health Diagnosis AI Demo
Educational purposes only - using publicly available medical images
"""

import requests
import os
from pathlib import Path

def download_medical_images():
    """Download sample medical images for demo purposes"""
    
    # Create images directory
    images_dir = Path("sample_medical_images")
    images_dir.mkdir(exist_ok=True)
    
    print("🏥 Downloading Sample Medical Images for Demo...")
    print("📋 These are for EDUCATIONAL PURPOSES ONLY")
    print()
    
    # Sample medical images from public datasets/sources
    medical_images = {
        "chest_xray_normal.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Chest_X-ray_%28PA%29_of_a_healthy_person.jpg/512px-Chest_X-ray_%28PA%29_of_a_healthy_person.jpg",
        "chest_xray_pneumonia.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Pneumonia_x-ray.jpg/512px-Pneumonia_x-ray.jpg",
        "skin_mole.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Melanoma.jpg/512px-Melanoma.jpg",
        "brain_mri.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/MRI_Head_5_slices.jpg/512px-MRI_Head_5_slices.jpg",
        "hand_xray.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/X-ray_of_normal_hand_by_dorsoplantar_projection.jpg/512px-X-ray_of_normal_hand_by_dorsoplantar_projection.jpg"
    }
    
    downloaded_count = 0
    
    for filename, url in medical_images.items():
        try:
            print(f"📥 Downloading {filename}...")
            
            # Download image
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save image
            image_path = images_dir / filename
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Downloaded: {filename}")
            downloaded_count += 1
            
        except Exception as e:
            print(f"❌ Failed to download {filename}: {str(e)}")
    
    print()
    print(f"🎉 Successfully downloaded {downloaded_count}/{len(medical_images)} medical images!")
    print(f"📁 Images saved in: {images_dir.absolute()}")
    print()
    print("🔍 Available images for demo:")
    for filename in medical_images.keys():
        image_path = images_dir / filename
        if image_path.exists():
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename} (failed)")
    
    print()
    print("⚠️  IMPORTANT DISCLAIMERS:")
    print("   • These images are for EDUCATIONAL DEMO purposes only")
    print("   • NOT for actual medical diagnosis or treatment")
    print("   • Images from public domain/Wikipedia Commons")
    print("   • Always consult healthcare professionals for medical advice")
    
    return images_dir

def create_sample_images_info():
    """Create info file about the sample images"""
    
    info_content = """# Sample Medical Images for Smart Health Diagnosis AI Demo

## 📋 Image Descriptions:

### 🫁 chest_xray_normal.jpg
- **Type:** Chest X-ray (PA view)
- **Condition:** Normal healthy chest
- **Use:** Demonstrate normal lung appearance
- **AI Analysis:** CNN can analyze lung fields, heart size, bone structure

### 🫁 chest_xray_pneumonia.jpg  
- **Type:** Chest X-ray showing pneumonia
- **Condition:** Pneumonia with lung consolidation
- **Use:** Show abnormal lung patterns
- **AI Analysis:** CNN can detect opacity and consolidation patterns

### 🔍 skin_mole.jpg
- **Type:** Dermatological image
- **Condition:** Skin lesion/mole
- **Use:** Demonstrate skin condition analysis
- **AI Analysis:** CNN can analyze color, shape, texture patterns

### 🧠 brain_mri.jpg
- **Type:** Brain MRI slices
- **Condition:** Normal brain anatomy
- **Use:** Show brain imaging analysis
- **AI Analysis:** CNN can analyze brain structure and anatomy

### 🖐️ hand_xray.jpg
- **Type:** Hand X-ray (dorsal view)
- **Condition:** Normal hand bones
- **Use:** Demonstrate bone structure analysis
- **AI Analysis:** CNN can analyze bone density, fractures, joint spaces

## 🎯 How to Use in Demo:

1. **Launch the Smart Health Diagnosis AI**
2. **Enter relevant symptoms** (e.g., cough, chest pain for chest X-ray)
3. **Upload one of these sample images**
4. **Select CNN (Medical Imaging) AI doctor**
5. **Watch AI analyze the medical image**

## ⚠️ IMPORTANT DISCLAIMERS:

- **EDUCATIONAL DEMO ONLY** - Not for actual medical diagnosis
- **Public domain images** from Wikipedia Commons
- **Always consult healthcare professionals** for medical advice
- **AI analysis is simulated** for educational purposes

## 📚 Educational Value:

Students will learn:
- How AI analyzes different types of medical images
- Visual pattern recognition in healthcare
- CNN applications in medical imaging
- Responsible AI development in healthcare

Built with ❤️ by Pravin Menghani - In love with Neural Networks!!
"""
    
    with open("sample_medical_images/README.md", "w") as f:
        f.write(info_content)
    
    print("📄 Created README.md with image descriptions")

if __name__ == "__main__":
    try:
        images_dir = download_medical_images()
        create_sample_images_info()
        
        print()
        print("🚀 Ready to use in Smart Health Diagnosis AI!")
        print("💡 Upload these images in the demo to see CNN medical image analysis")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 You can also manually download medical images from:")
        print("   • Wikipedia Commons medical images")
        print("   • Public medical datasets")
        print("   • Educational medical image repositories")
