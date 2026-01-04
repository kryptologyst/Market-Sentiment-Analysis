#!/usr/bin/env python3
"""Simple run script for market sentiment analysis."""

import sys
import subprocess
from pathlib import Path


def run_demo():
    """Run the Streamlit demo."""
    print("🚀 Starting Market Sentiment Analysis Demo...")
    print("⚠️  Remember: This is for research and educational purposes only!")
    print("📊 Opening Streamlit app...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "demo/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running demo: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
        return True
    
    return True


def run_training():
    """Run the training pipeline."""
    print("🏋️ Starting Market Sentiment Analysis Training...")
    print("⚠️  Remember: This is for research and educational purposes only!")
    
    try:
        subprocess.run([
            sys.executable, "scripts/train.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running training: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Training stopped by user")
        return True
    
    return True


def run_baseline_comparison():
    """Run baseline model comparison."""
    print("📊 Running Baseline Model Comparison...")
    print("⚠️  Remember: This is for research and educational purposes only!")
    
    try:
        subprocess.run([
            sys.executable, "scripts/train.py", "--baseline-only"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running baseline comparison: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Baseline comparison stopped by user")
        return True
    
    return True


def main():
    """Main entry point."""
    print("=" * 60)
    print("📈 MARKET SENTIMENT ANALYSIS")
    print("=" * 60)
    print()
    print("⚠️  IMPORTANT DISCLAIMER:")
    print("   This tool is for RESEARCH and EDUCATIONAL purposes only.")
    print("   It is NOT providing investment advice.")
    print("   Results may be inaccurate and should not be used for trading.")
    print()
    print("Available options:")
    print("1. 🚀 Run Interactive Demo (Streamlit)")
    print("2. 🏋️ Run Full Training Pipeline")
    print("3. 📊 Run Baseline Model Comparison")
    print("4. ❌ Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                return run_demo()
            elif choice == "2":
                return run_training()
            elif choice == "3":
                return run_baseline_comparison()
            elif choice == "4":
                print("👋 Goodbye!")
                return True
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
