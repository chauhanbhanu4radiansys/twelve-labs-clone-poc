"""
Test script to verify video download functionality
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.embedding.download import download_from_url


def test_download():
    """Test the download function with a sample URL"""
    
    # Test URL from run_example.py
    test_url = "https://vfxai-storage.s3.us-east-1.amazonaws.com/dev/raw/69255973bb09ecdc9e888a87?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjELX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDivAMedSUqjYt3Dp8Utj8VmwOhK7spMzRxu%2FQamOXmvgIgRg9kgL%2FXIO9Q3IKFK7jMnFzeVYKWb59KYKmQhdd6ICkq0AMIfhAAGgw1NDY4ODAzNTgzODciDEZ9JQieTi6U8OcKZCqtA4qICzoz5l4xM49V2aveYSURliNJ98fKgnmzXr8KdLT945Vg8oejGae%2Bqp%2BE8jPli4o4yPr3xwPQTCE%2FGT3VRIV%2FXc1z%2BXb%2F9RiOkhvOlGaHNkJHjmCtGql6hijYy0K2X2Rq%2BwxvyxR11ZG41vXfGimcd5Lc4jf7whiOmTKeSKDtczGILtNGbZ8WBpw87UvHY8uArKJm%2F1K0WO48BJZwQD8hppXaI12x0w1rnBQNrcEKQyVh8cFO2tcvPTnc3Q9PT6IX%2FAe3mtloP%2Fk%2Fv9XV%2BDdLstztZG7k7%2BO7344SILQMi9yNIm198pKxl15oh9I4FaFnTT8IeXUrtl1jTf%2BarmTSrYmpHU6ZXD1OHIQ5BshsslCgI31%2FGMzFUvIgf0grC3pTY1EYo%2FJQ8kwceofpkp0uGDtyCKvg6UwJXNU1B9fWlhzRj9nwjQDteVRiV4Yo4A3%2FY4pQYVf29TOd5jYXBIfEpHammuZGStAznhbHjAKaqvezojI2I1RPWP2z2NuUu30GAcon%2B4FUMTmsggu9989jqSAl7f9OKumipFFQaToIF3koDAyyPfrA3fj3BjCfiJrJBjreAgyRvTN5XGCI7hOE6W2H0Jq%2BuT0Ug8fqB9%2BdNOF3e05cAsStb92CxCKcPEe1HKE0boaYEYVAAELTNHop%2BCqNs4NBzgd3pElC91hcWgAhLtYCtCth8BVE1fH%2BrfCJvr9J5PcJSfZ4lN9wFUig%2BLB47zwT7rSgPV660fC%2FSAERCSGy0N%2F%2BOabP5muPA9CPr3ZGLGx%2BB8hRkKrRhz7pJ1EMB5nbNqRt7v4YybsjafRxwnnt%2F2yk4QyR4YeSKfgV45h7dNLOLL98uBG0vWpfq%2BlxYlh0iA3A7mcZMKAAK5WsNUoO4P%2B4Ezt0m3VEr%2BFIwPV8C1yFUHNiKlZK8GxPf1njoTK0wFjVRdZNorjJDSdEBjsl7rrTfsrrGOtwpWTQINq8i0T9eoEJOmpLaqL0M0mCyGh1RbVE8G8i7hMpz600OB3Klg%2BmnH%2Fnk65HmwfJ%2FIYlwvgivg445oI2Fr7mDwh2&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAX6VE4A7ZYRN2M4GB%2F20251126%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251126T043900Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=afeba486d92cb11cb1aba4a16c3b104b4259a85f463331665f33c5d698c19d0f"
    
    print("=" * 80)
    print("Testing Video Download")
    print("=" * 80)
    print(f"URL: {test_url[:100]}...")
    print()
    
    def progress(msg):
        print(f"[PROGRESS] {msg}")
    
    try:
        result = download_from_url(test_url, progress_callback=progress)
        
        if result:
            print(f"\n✓ Download successful!")
            print(f"  File: {result}")
            print(f"  Size: {os.path.getsize(result) / (1024*1024):.2f} MB")
            
            # Clean up
            try:
                os.remove(result)
                print(f"  Cleaned up temporary file")
            except:
                print(f"  Warning: Could not remove temporary file")
        else:
            print(f"\n✗ Download failed!")
            return 1
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(test_download())

