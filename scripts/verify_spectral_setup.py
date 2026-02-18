#!/usr/bin/env python3
"""
Spectral Analysis Verification Script

Tests that all modules are installed and working correctly.
Run this before doing real analysis to ensure setup is complete.

Usage:
    python scripts/verify_spectral_setup.py
    
Exit codes:
    0 = All systems OK
    1 = Missing dependencies
    2 = Module import errors
    3 = Path issues
"""

import sys
import subprocess
from pathlib import Path

def check_venv():
    """Check if virtual environment is activated."""
    print("=" * 70)
    print("CHECKING VIRTUAL ENVIRONMENT")
    print("=" * 70)
    
    venv_path = Path(".venv/bin/python")
    if not venv_path.exists():
        print("❌ Virtual environment not found at .venv/")
        print("   Run: python3 -m venv .venv")
        return False
    
    print("✅ Virtual environment found")
    
    # Check if we're using it
    if not hasattr(sys, 'real_prefix') and not (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    ):
        print("⚠️  Virtual environment not activated")
        print("   Run: source .venv/bin/activate")
        return False
    
    print("✅ Virtual environment is activated")
    return True


def check_dependencies():
    """Check if all required packages are installed."""
    print("\n" + "=" * 70)
    print("CHECKING DEPENDENCIES")
    print("=" * 70)
    
    required_packages = {
        'cv2': 'opencv-python',
        'scipy': 'scipy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'decord': 'decord',
        'numpy': 'numpy',
        'torch': 'torch',
    }
    
    missing = []
    
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"✅ {package_name:20s} installed")
        except ImportError:
            print(f"❌ {package_name:20s} NOT installed")
            missing.append(package_name)
    
    if missing:
        print("\n" + "-" * 70)
        print("Missing packages. Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def check_modules():
    """Check if project modules can be imported."""
    print("\n" + "=" * 70)
    print("CHECKING PROJECT MODULES")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    src_root = project_root / "src"
    
    # Add to path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(src_root))
    
    modules_to_check = [
        ('info_rates', 'Core package'),
        ('info_rates.analysis', 'Analysis subpackage'),
        ('info_rates.sampling', 'Sampling subpackage'),
        ('info_rates.data', 'Data subpackage'),
    ]
    
    errors = []
    
    for module_name, description in modules_to_check:
        try:
            __import__(module_name)
            print(f"✅ {module_name:35s} ({description})")
        except Exception as e:
            print(f"❌ {module_name:35s} ({description})")
            errors.append((module_name, str(e)))
    
    if errors:
        print("\n" + "-" * 70)
        print("Module import errors:")
        for module, error in errors:
            print(f"  {module}: {error}")
        return False
    
    return True


def check_spectral_analysis():
    """Deep check: can we import and instantiate spectral analysis?"""
    print("\n" + "=" * 70)
    print("CHECKING SPECTRAL ANALYSIS MODULE")
    print("=" * 70)
    
    try:
        from info_rates.analysis.spectral_analysis import (
            OpticalFlowExtractor,
            TemporalFFT,
            SpectralAnalyzer,
            SpectralMetrics
        )
        print("✅ Imported OpticalFlowExtractor")
        print("✅ Imported TemporalFFT")
        print("✅ Imported SpectralAnalyzer")
        print("✅ Imported SpectralMetrics")
        return True
    except Exception as e:
        print(f"❌ Failed to import spectral analysis: {e}")
        return False


def check_paths():
    """Check if all required directories and files exist."""
    print("\n" + "=" * 70)
    print("CHECKING PATHS AND FILES")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    
    paths = [
        (project_root / "src" / "info_rates", "Source directory"),
        (project_root / "scripts", "Scripts directory"),
        (project_root / "evaluations", "Evaluations directory"),
        (project_root / "docs", "Documentation directory"),
        (project_root / "scripts" / "run_spectral_analysis.py", "Main script"),
        (project_root / "scripts" / "analysis" / "spectral_correlation.py", "Analysis script"),
        (project_root / "src" / "info_rates" / "analysis" / "spectral_analysis.py", "Core module"),
    ]
    
    all_ok = True
    
    for path, description in paths:
        if path.exists():
            print(f"✅ {description:40s} at {path.relative_to(project_root)}")
        else:
            print(f"❌ {description:40s} NOT found at {path.relative_to(project_root)}")
            all_ok = False
    
    return all_ok


def test_demo_mode():
    """Test that demo mode works (quick validation)."""
    print("\n" + "=" * 70)
    print("TESTING DEMO MODE (No videos needed)")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "evaluations" / "spectral_verify_test"
    
    print(f"\nRunning: python scripts/run_spectral_analysis.py --output-dir {output_dir}")
    print("(This should complete in <10 seconds with synthetic data)\n")
    
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "run_spectral_analysis.py"),
                "--output-dir", str(output_dir)
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Demo mode completed successfully")
            
            # Check outputs
            outputs = [
                "correlation_analysis.json",
                "spectral_validation_summary.csv",
                "01_spectral_profiles.png",
                "02_correlation_scatter.png",
                "03_sensitivity_tiers_spectral.png"
            ]
            
            all_present = True
            for output_file in outputs:
                output_path = output_dir / output_file
                if output_path.exists():
                    size_kb = output_path.stat().st_size / 1024
                    print(f"✅ Generated {output_file:40s} ({size_kb:.1f} KB)")
                else:
                    print(f"❌ Missing {output_file}")
                    all_present = False
            
            return all_present
        else:
            print(f"❌ Demo mode failed with exit code {result.returncode}")
            print("\nStdout:")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print("\nStderr:")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False
    
    except subprocess.TimeoutExpired:
        print("❌ Demo mode timed out (>60 seconds)")
        return False
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False


def main():
    """Run all checks and report status."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "SPECTRAL ANALYSIS VERIFICATION" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    
    checks = [
        ("Virtual Environment", check_venv),
        ("Dependencies", check_dependencies),
        ("Project Modules", check_modules),
        ("Spectral Analysis", check_spectral_analysis),
        ("Paths & Files", check_paths),
        ("Demo Mode Test", test_demo_mode),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ Error in {check_name}: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}\n")
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {check_name}")
    
    if passed == total:
        print("\n" + "🎉 " * 10)
        print("ALL CHECKS PASSED - Spectral Analysis is ready to use!")
        print("🎉 " * 10)
        print("\nQuick start:")
        print("  python scripts/run_spectral_analysis.py \\")
        print("    --sensitivity-csv evaluations/ucf101/vivit/vivit_per_class.csv \\")
        print("    --dataset-root data/UCF101_data/UCF-101 \\")
        print("    --output-dir evaluations/spectral_analysis")
        return 0
    else:
        print("\n❌ Some checks failed. See above for details.")
        print("\nFor help:")
        print("  1. Check docs/SPECTRAL_QUICK_REFERENCE.md")
        print("  2. Ensure virtual environment is activated: source .venv/bin/activate")
        print("  3. Install missing dependencies: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
