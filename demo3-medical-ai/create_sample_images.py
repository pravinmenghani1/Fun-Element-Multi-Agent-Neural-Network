#!/usr/bin/env python3
"""
Create Sample Medical Images for Smart Health Diagnosis AI Demo
Generates simple sample images and provides download sources
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import os

def create_sample_medical_images():
    """Create simple sample medical images for demo purposes"""
    
    # Create images directory
    images_dir = Path("sample_medical_images")
    images_dir.mkdir(exist_ok=True)
    
    print("🏥 Creating Sample Medical Images for Demo...")
    print("📋 These are SIMULATED images for EDUCATIONAL PURPOSES ONLY")
    print()
    
    # 1. Create sample chest X-ray
    create_chest_xray_sample(images_dir)
    
    # 2. Create sample skin lesion
    create_skin_lesion_sample(images_dir)
    
    # 3. Create sample brain MRI
    create_brain_mri_sample(images_dir)
    
    # 4. Create sample hand X-ray
    create_hand_xray_sample(images_dir)
    
    print()
    print("🎉 Successfully created sample medical images!")
    print(f"📁 Images saved in: {images_dir.absolute()}")
    
    return images_dir

def create_chest_xray_sample(images_dir):
    """Create a simple chest X-ray sample"""
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    
    # Create chest outline
    chest = patches.Ellipse((0.5, 0.6), 0.8, 0.7, fill=False, edgecolor='white', linewidth=2)
    ax.add_patch(chest)
    
    # Add ribs
    for i in range(6):
        y_pos = 0.8 - i * 0.08
        rib_left = patches.Arc((0.3, y_pos), 0.4, 0.1, angle=0, theta1=0, theta2=180, 
                              edgecolor='lightgray', linewidth=1)
        rib_right = patches.Arc((0.7, y_pos), 0.4, 0.1, angle=0, theta1=0, theta2=180, 
                               edgecolor='lightgray', linewidth=1)
        ax.add_patch(rib_left)
        ax.add_patch(rib_right)
    
    # Add heart shadow
    heart = patches.Ellipse((0.45, 0.5), 0.25, 0.3, fill=True, facecolor='gray', alpha=0.3)
    ax.add_patch(heart)
    
    # Add lungs
    lung_left = patches.Ellipse((0.35, 0.6), 0.25, 0.5, fill=True, facecolor='darkgray', alpha=0.2)
    lung_right = patches.Ellipse((0.65, 0.6), 0.25, 0.5, fill=True, facecolor='darkgray', alpha=0.2)
    ax.add_patch(lung_left)
    ax.add_patch(lung_right)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor('black')
    ax.set_title('Sample Chest X-ray', color='white', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(images_dir / 'sample_chest_xray.png', facecolor='black', dpi=150)
    plt.close()
    
    print("✅ Created: sample_chest_xray.png")

def create_skin_lesion_sample(images_dir):
    """Create a simple skin lesion sample"""
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Create skin background
    ax.set_facecolor('#FDBCB4')  # Skin color
    
    # Add skin texture
    np.random.seed(42)
    x = np.random.uniform(0, 1, 1000)
    y = np.random.uniform(0, 1, 1000)
    colors = np.random.uniform(0.8, 1.0, 1000)
    ax.scatter(x, y, c=colors, s=1, alpha=0.3, cmap='Reds')
    
    # Add mole/lesion
    mole = patches.Circle((0.5, 0.5), 0.15, fill=True, facecolor='#8B4513', alpha=0.8)
    ax.add_patch(mole)
    
    # Add irregular border
    theta = np.linspace(0, 2*np.pi, 20)
    r = 0.15 + 0.02 * np.sin(5*theta)
    x_border = 0.5 + r * np.cos(theta)
    y_border = 0.5 + r * np.sin(theta)
    ax.plot(x_border, y_border, color='#654321', linewidth=2)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Sample Skin Lesion', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(images_dir / 'sample_skin_lesion.png', dpi=150)
    plt.close()
    
    print("✅ Created: sample_skin_lesion.png")

def create_brain_mri_sample(images_dir):
    """Create a simple brain MRI sample"""
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Create brain outline
    brain = patches.Circle((0.5, 0.5), 0.4, fill=True, facecolor='lightgray', alpha=0.8)
    ax.add_patch(brain)
    
    # Add brain structures
    # Ventricles
    ventricle_left = patches.Ellipse((0.42, 0.55), 0.08, 0.15, fill=True, facecolor='black', alpha=0.6)
    ventricle_right = patches.Ellipse((0.58, 0.55), 0.08, 0.15, fill=True, facecolor='black', alpha=0.6)
    ax.add_patch(ventricle_left)
    ax.add_patch(ventricle_right)
    
    # Brain stem
    stem = patches.Rectangle((0.48, 0.35), 0.04, 0.15, fill=True, facecolor='gray', alpha=0.7)
    ax.add_patch(stem)
    
    # Add some texture
    np.random.seed(42)
    for _ in range(50):
        x = np.random.uniform(0.15, 0.85)
        y = np.random.uniform(0.15, 0.85)
        if (x - 0.5)**2 + (y - 0.5)**2 < 0.16:  # Inside brain
            circle = patches.Circle((x, y), 0.01, fill=True, facecolor='darkgray', alpha=0.3)
            ax.add_patch(circle)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor('black')
    ax.set_title('Sample Brain MRI', color='white', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(images_dir / 'sample_brain_mri.png', facecolor='black', dpi=150)
    plt.close()
    
    print("✅ Created: sample_brain_mri.png")

def create_hand_xray_sample(images_dir):
    """Create a simple hand X-ray sample"""
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    
    # Create hand bones
    # Palm
    palm = patches.Rectangle((0.3, 0.2), 0.4, 0.3, fill=True, facecolor='lightgray', alpha=0.7)
    ax.add_patch(palm)
    
    # Fingers
    finger_positions = [0.32, 0.42, 0.52, 0.62]
    for i, x_pos in enumerate(finger_positions):
        # Finger bones (3 segments each)
        for j in range(3):
            y_pos = 0.5 + j * 0.12
            height = 0.1 if j < 2 else 0.08
            finger_bone = patches.Rectangle((x_pos, y_pos), 0.06, height, 
                                          fill=True, facecolor='lightgray', alpha=0.8)
            ax.add_patch(finger_bone)
    
    # Thumb
    thumb1 = patches.Rectangle((0.25, 0.35), 0.05, 0.08, fill=True, facecolor='lightgray', alpha=0.8)
    thumb2 = patches.Rectangle((0.22, 0.43), 0.05, 0.08, fill=True, facecolor='lightgray', alpha=0.8)
    ax.add_patch(thumb1)
    ax.add_patch(thumb2)
    
    # Wrist bones
    for i in range(8):
        x_pos = 0.3 + (i % 4) * 0.1
        y_pos = 0.1 + (i // 4) * 0.08
        wrist_bone = patches.Circle((x_pos, y_pos), 0.03, fill=True, facecolor='lightgray', alpha=0.6)
        ax.add_patch(wrist_bone)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor('black')
    ax.set_title('Sample Hand X-ray', color='white', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(images_dir / 'sample_hand_xray.png', facecolor='black', dpi=150)
    plt.close()
    
    print("✅ Created: sample_hand_xray.png")

def create_download_guide():
    """Create guide for downloading real medical images"""
    
    guide_content = """# 📥 How to Get Real Medical Images for Demo

## 🎯 Quick Option: Use Our Sample Images
We've created simple sample medical images in the `sample_medical_images/` folder:
- `sample_chest_xray.png` - Simulated chest X-ray
- `sample_skin_lesion.png` - Simulated skin lesion
- `sample_brain_mri.png` - Simulated brain MRI
- `sample_hand_xray.png` - Simulated hand X-ray

## 🌐 Download Real Medical Images (Educational Use)

### 1. **NIH Chest X-ray Dataset**
- **URL:** https://www.kaggle.com/datasets/nih-chest-xrays/data
- **Content:** 112,120 chest X-ray images
- **Format:** PNG files
- **Use:** Perfect for CNN medical imaging demo

### 2. **ISIC Skin Lesion Dataset**
- **URL:** https://www.isic-archive.com/
- **Content:** Dermoscopic images of skin lesions
- **Format:** JPEG files
- **Use:** Great for skin condition analysis

### 3. **Brain MRI Dataset**
- **URL:** https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
- **Content:** Brain MRI scans
- **Format:** JPEG/PNG files
- **Use:** Brain imaging analysis demo

### 4. **Hand X-ray Dataset**
- **URL:** https://www.kaggle.com/datasets/kmader/rsna-bone-age
- **Content:** Hand X-ray images
- **Format:** PNG files
- **Use:** Bone structure analysis

## 📱 Manual Download Steps:

### Option 1: Kaggle Datasets
1. Create free Kaggle account
2. Search for medical image datasets
3. Download ZIP files
4. Extract to `sample_medical_images/` folder

### Option 2: Medical Image Repositories
1. Visit medical image databases
2. Look for "educational use" or "public domain" images
3. Download individual images
4. Save as PNG/JPEG in `sample_medical_images/` folder

### Option 3: Wikipedia Commons
1. Go to https://commons.wikimedia.org/
2. Search for "medical imaging" or "X-ray"
3. Look for public domain medical images
4. Right-click and save images

## 🎯 Recommended Image Types for Demo:

### For CNN Analysis:
- **Chest X-rays** - Show lung patterns, heart size
- **Skin lesions** - Demonstrate color/texture analysis
- **Brain MRIs** - Show brain structure analysis
- **Bone X-rays** - Demonstrate fracture detection

### Image Requirements:
- **Format:** PNG, JPEG, or JPG
- **Size:** Any size (will be processed by AI)
- **Quality:** Clear, medical-grade preferred
- **Content:** Real medical images for best demo effect

## ⚠️ IMPORTANT DISCLAIMERS:

- **Educational Use Only** - Not for actual diagnosis
- **Respect Privacy** - Use only public/anonymized images
- **Copyright Compliance** - Ensure proper usage rights
- **Medical Ethics** - Always emphasize demo limitations

## 🚀 Using Images in Demo:

1. **Launch Smart Health Diagnosis AI**
2. **Enter relevant symptoms**
3. **Upload medical image**
4. **Select CNN (Medical Imaging) AI**
5. **Watch AI analyze the image**

Built with ❤️ by Pravin Menghani - In love with Neural Networks!!
"""
    
    with open("MEDICAL_IMAGE_DOWNLOAD_GUIDE.md", "w") as f:
        f.write(guide_content)
    
    print("📄 Created MEDICAL_IMAGE_DOWNLOAD_GUIDE.md")

if __name__ == "__main__":
    try:
        # Install matplotlib if not available
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        images_dir = create_sample_medical_images()
        create_download_guide()
        
        print()
        print("🚀 Sample medical images ready for Smart Health Diagnosis AI!")
        print("💡 You can now upload these images in the demo to see CNN analysis")
        print("📖 Check MEDICAL_IMAGE_DOWNLOAD_GUIDE.md for real medical images")
        
    except ImportError:
        print("❌ matplotlib not installed. Installing...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        print("✅ matplotlib installed. Please run the script again.")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Alternative: Manually download medical images from:")
        print("   • Kaggle medical datasets")
        print("   • Wikipedia Commons medical images")
        print("   • Public medical image repositories")
