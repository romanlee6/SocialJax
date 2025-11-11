"""
Quick test script to verify benchmark setup before running full benchmark.

This script runs a minimal test with just one model and few steps to ensure
everything is configured correctly.
"""

import os
import sys

def test_environment():
    """Test that environment is properly configured."""
    print("\n" + "="*80)
    print("TESTING BENCHMARK SETUP")
    print("="*80 + "\n")
    
    # Test 1: Check API key
    print("1. Checking API key...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("   ❌ FAIL: OPENAI_API_KEY not set")
        print("   Please set it: export OPENAI_API_KEY='your-key-here'")
        return False
    else:
        print(f"   ✓ PASS: API key is set (length: {len(api_key)})")
    
    # Test 2: Check base URL (optional)
    print("\n2. Checking API base URL...")
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        print(f"   ✓ Custom base URL set: {base_url}")
    else:
        print("   ℹ Using default OpenAI endpoint")
    
    # Test 3: Check dependencies
    print("\n3. Checking dependencies...")
    dependencies = {
        'jax': 'jax',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'PIL': 'pillow',
        'openai': 'openai'
    }
    
    all_deps_ok = True
    for module, package_name in dependencies.items():
        try:
            __import__(module)
            print(f"   ✓ {package_name} installed")
        except ImportError:
            print(f"   ❌ {package_name} NOT installed")
            print(f"      Install with: pip install {package_name}")
            all_deps_ok = False
    
    if not all_deps_ok:
        return False
    
    # Test 4: Check SocialJax
    print("\n4. Checking SocialJax...")
    try:
        import socialjax
        print(f"   ✓ SocialJax imported successfully")
    except ImportError as e:
        print(f"   ❌ Cannot import SocialJax: {e}")
        return False
    
    # Test 5: Check coins_llm_simulation
    print("\n5. Checking coins_llm_simulation module...")
    try:
        from coins_llm_simulation import CoinGame, LLMAgent, ActionParser
        print(f"   ✓ coins_llm_simulation imported successfully")
    except ImportError as e:
        print(f"   ❌ Cannot import coins_llm_simulation: {e}")
        return False
    
    # Test 6: Check benchmark_llms
    print("\n6. Checking benchmark_llms module...")
    try:
        from benchmark_llms import BenchmarkRunner, BenchmarkLogger
        print(f"   ✓ benchmark_llms imported successfully")
    except ImportError as e:
        print(f"   ❌ Cannot import benchmark_llms: {e}")
        return False
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED - Environment is ready!")
    print("="*80 + "\n")
    
    return True


def run_quick_test():
    """Run a quick 3-step simulation with one model."""
    print("\n" + "="*80)
    print("RUNNING QUICK TEST SIMULATION")
    print("="*80)
    print("This will run a 3-step simulation with gpt-5-mini to verify everything works.\n")
    
    from benchmark_llms import BenchmarkRunner
    
    # Run minimal benchmark
    runner = BenchmarkRunner(
        models=["gpt-5-mini"],  # Just one fast model
        num_steps=3,            # Just 3 steps
        seed=42,
        base_output_dir="./test_benchmark_output",
        temperature=0.7
    )
    
    try:
        runner.run_benchmark()
        
        print("\n" + "="*80)
        print("✓ QUICK TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nYour setup is working correctly! You can now run the full benchmark:")
        print("  python benchmark_llms.py")
        print("\nor use the shell script:")
        print("  ./run_benchmark.sh")
        print("\n" + "="*80 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        print("\nPlease check the error message above and fix any issues.")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test benchmark setup before running full benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Only test environment, skip running simulation"
    )
    
    args = parser.parse_args()
    
    # Run environment tests
    env_ok = test_environment()
    
    if not env_ok:
        print("\n❌ Environment test failed. Please fix the issues above before proceeding.\n")
        sys.exit(1)
    
    # Run quick simulation test unless skipped
    if not args.skip_simulation:
        sim_ok = run_quick_test()
        if not sim_ok:
            sys.exit(1)
    else:
        print("\nℹ Skipping simulation test (--skip-simulation flag used)\n")
        print("To test with actual simulation, run without --skip-simulation")
    
    sys.exit(0)


if __name__ == "__main__":
    main()

