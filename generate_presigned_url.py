#!/usr/bin/env python3
"""
Standalone script to generate AWS S3 presigned URLs for file uploads.
This script can run in complete isolation from the rest of the codebase.

================================================================================
HOW TO RUN THIS FILE:
================================================================================

1. Make sure you have boto3 installed:
   pip install boto3
   OR
   conda install boto3

2. Ensure AWS credentials are configured (via AWS CLI, environment variables, 
   or IAM role):
   aws configure
   OR
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret

3. Run the script with required arguments:
   python generate_presigned_url.py <bucket> <directory> <filename> [options]

4. Examples:
   # Basic usage:
   python generate_presigned_url.py vfxai-storage local/video_Summarization/ 20251216T140000.json
   
   # With custom region:
   python generate_presigned_url.py vfxai-storage local/video_Summarization/ file.json --region us-west-2
   
   # With custom expiration (1 hour):
   python generate_presigned_url.py vfxai-storage local/video_Summarization/ file.json --expires 3600
   
   # With conda environment:
   conda run -n clone12labs python generate_presigned_url.py vfxai-storage local/video_Summarization/ file.json

5. View help:
   python generate_presigned_url.py --help

================================================================================
ARGUMENTS:
================================================================================
  bucket      : S3 bucket name (required)
  directory   : Directory path in S3, e.g., "local/video_Summarization/" (required)
  filename    : Name of the file to upload (required)
  --region    : AWS region (optional, default: us-east-1)
  --expires   : URL expiration in seconds (optional, default: 1800 = 30 minutes)

================================================================================
OUTPUT:
================================================================================
The script will output a presigned URL that can be used to upload files via PUT request.
The URL uses AWS Signature Version 4 and expires after the specified time.

================================================================================
"""

import boto3
from botocore.config import Config
import sys
import argparse


def create_presigned_url_upload(bucket, file_path, region='us-east-1', expires_in=1800):
    """
    Creates a presigned URL for uploading a file to S3 using AWS Signature Version 4.
    
    Args:
        bucket: S3 bucket name
        file_path: S3 object key (file path in bucket)
        region: AWS region (default: us-east-1)
        expires_in: URL expiration time in seconds (default: 1800 = 30 minutes)
        
    Returns:
        Presigned URL string for uploading (PUT request)
    """
    # Configure boto3 to use signature version 4
    config = Config(
        signature_version='s3v4',
        region_name=region
    )
    s3 = boto3.client('s3', config=config, region_name=region)
    
    response = s3.generate_presigned_url(
        'put_object',
        Params={'Bucket': bucket, 'Key': file_path},
        ExpiresIn=expires_in
    )
    
    return response


def main():
    """Main function to generate presigned URL with specified parameters."""
    parser = argparse.ArgumentParser(
        description='Generate AWS S3 presigned URL for file uploads',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s vfxai-storage local/video_Summarization/ 20251216T140000.json
  %(prog)s my-bucket path/to/dir/ file.json --region us-west-2 --expires 3600
        """
    )
    
    parser.add_argument('bucket', help='S3 bucket name')
    parser.add_argument('directory', help='Directory path in S3 (e.g., local/video_Summarization/)')
    parser.add_argument('filename', help='File name to upload')
    parser.add_argument('--region', default='us-east-1', help='AWS region (default: us-east-1)')
    parser.add_argument('--expires', type=int, default=1800, help='URL expiration time in seconds (default: 1800 = 30 minutes)')
    
    args = parser.parse_args()
    
    # Normalize directory path (ensure it ends with /)
    directory = args.directory.rstrip('/') + '/'
    
    # Construct full S3 path
    s3_path = f"{directory}{args.filename}"
    
    try:
        # Generate presigned URL
        presigned_url = create_presigned_url_upload(
            bucket=args.bucket,
            file_path=s3_path,
            region=args.region,
            expires_in=args.expires
        )
        
        print("=" * 80)
        print("PRESIGNED URL FOR UPLOAD (PUT)")
        print("=" * 80)
        print(f"\nBucket: {args.bucket}")
        print(f"Path: {s3_path}")
        print(f"Region: {args.region}")
        print(f"Expires in: {args.expires} seconds ({args.expires // 60} minutes)")
        print("\n" + "-" * 80)
        print("URL:")
        print("-" * 80)
        print(presigned_url)
        print("-" * 80)
        print("\nUsage examples:")
        print(f"  curl -X PUT --upload-file {args.filename} \"{presigned_url}\"")
        print("\n  Python:")
        print(f"    import requests")
        print(f"    with open('{args.filename}', 'rb') as f:")
        print(f"        response = requests.put('{presigned_url}', data=f)")
        print("=" * 80)
        
        return presigned_url
        
    except Exception as e:
        print(f"Error generating presigned URL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
