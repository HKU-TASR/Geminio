#!/bin/bash

# ImageNet Pipeline Test Script
# This script tests the complete pipeline for ImageNet dataset:
# 1. VLM embedding preprocessing
# 2. Malicious model training  
# 3. Gradient inversion attack

set -e  # Exit on any error

# Configuration
SKIP_EMBEDDING=${SKIP_EMBEDDING:-false}
SKIP_TRAINING=${SKIP_TRAINING:-false}
SKIP_RECONSTRUCTION=${SKIP_RECONSTRUCTION:-false}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-embedding)
            SKIP_EMBEDDING=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-reconstruction)
            SKIP_RECONSTRUCTION=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --skip-embedding      Skip VLM embedding generation"
            echo "  --skip-training       Skip malicious model training"
            echo "  --skip-reconstruction Skip reconstruction testing"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "================================================"
echo "ImageNet Gradient Inversion Attack Pipeline"
echo "================================================"

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p malicious_models
mkdir -p results/imagenet
mkdir -p assets/private_samples

echo ""
echo "Step 1: VLM Embedding Preprocessing"
echo "-----------------------------------"

if [ "$SKIP_EMBEDDING" = true ]; then
    echo "⏩ Skipping VLM embedding generation (--skip-embedding)"
    echo "Expected files:"
    echo "  - ./data/imagenet-clip-test.pt"
    echo "  - ./data/imagenet-clip-meta.pt"
else
    echo "📋 Checking ImageNet dataset availability..."
    if [ ! -d "./data/imagenet" ]; then
        echo "❌ ImageNet dataset not found in ./data/imagenet/"
        echo "Please download and extract ImageNet validation set:"
        echo "  - ILSVRC2012_img_val.tar"
        echo "  - ILSVRC2012_devkit_t12.tar.gz"
        exit 1
    fi
    
    echo "🔄 Generating CLIP embeddings for ImageNet dataset..."
    python vlm-imagenet-embed.py
    
    echo "✅ VLM embeddings generated successfully!"
    echo "Generated files:"
    ls -la ./data/imagenet-clip-*.pt 2>/dev/null || echo "  (No embedding files found)"
fi

echo ""
echo "Step 2: Malicious Model Training"
echo "--------------------------------"

if [ "$SKIP_TRAINING" = true ]; then
    echo "⏩ Skipping malicious model training (--skip-training)"
    echo "Expected files:"
    echo "  - ./malicious_models/Any_jewelry.pt"
    echo "  - ./malicious_models/Any_human_faces.pt"  
    echo "  - ./malicious_models/Any_males_with_a_beard.pt"
    echo "  - ./malicious_models/Any_guns.pt"
    echo "  - ./malicious_models/Any_females_riding_a_horse.pt"
else
    echo "📋 Checking for existing malicious models..."
    existing_models=$(ls ./malicious_models/*.pt 2>/dev/null | wc -l)
    if [ "$existing_models" -eq 5 ]; then
        echo "✅ Pre-trained models already available (5/5):"
        ls -la ./malicious_models/*.pt
        echo "💡 To retrain models, remove them and run again"
    else
        echo "🔄 Training malicious models for different semantic queries..."
        python main_geminio-imagenet.py
        
        echo "✅ Model training completed!"
        echo "Generated models:"
        ls -la ./malicious_models/*.pt 2>/dev/null || echo "  (No model files found)"
    fi
fi

echo ""
echo "Step 3: Gradient Inversion Attack"
echo "---------------------------------"

if [ "$SKIP_RECONSTRUCTION" = true ]; then
    echo "⏩ Skipping reconstruction testing (--skip-reconstruction)"
else
    echo "📋 Checking private sample images..."
    if [ ! -d "./assets/private_samples" ] || [ -z "$(ls -A ./assets/private_samples 2>/dev/null)" ]; then
        echo "❌ No private sample images found in ./assets/private_samples/"
        echo "Please add images in format: {index}-{class}.png (e.g., 1-285.png)"
        exit 1
    fi
    
    sample_count=$(ls ./assets/private_samples/*.png 2>/dev/null | wc -l)
    echo "✅ Found $sample_count private sample images"
    
    echo ""
    echo "🔄 Testing baseline reconstruction..."
    python main_breaching-imagenet.py --baseline
    echo "✅ Baseline reconstruction completed!"
    
    echo ""
    echo "🔄 Testing query-guided reconstruction for 'Any jewelry?'..."
    python main_breaching-imagenet.py --geminio-query "Any jewelry?"
    echo "✅ Jewelry query reconstruction completed!"
    
    echo ""
    echo "🔄 Testing query-guided reconstruction for 'Any human faces?'..."
    python main_breaching-imagenet.py --geminio-query "Any human faces?"
    echo "✅ Human faces query reconstruction completed!"
    
    echo ""
    echo "📂 Generated reconstruction results:"
    for dir in ./results/*/; do
        if [ -d "$dir" ]; then
            echo "  📁 $dir"
            ls -la "$dir"*.jpg 2>/dev/null | sed 's/^/    /'
        fi
    done
fi
