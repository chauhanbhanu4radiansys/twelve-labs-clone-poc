# MinIO Docker Container Information

## Status
✅ MinIO is running in Docker

## Connection Details

### MinIO API (S3-compatible)
- **URL:** http://localhost:9000
- **Access Key:** `minioadmin`
- **Secret Key:** `minioadmin`

### MinIO Console (Web UI)
- **URL:** http://localhost:9001
- **Username:** `minioadmin`
- **Password:** `minioadmin`

## Docker Commands

### View MinIO Status
```bash
docker ps | grep minio
```

### View MinIO Logs
```bash
docker logs -f minio
```

### Stop MinIO
```bash
docker stop minio
```

### Start MinIO (if stopped)
```bash
docker start minio
```

### Restart MinIO
```bash
docker restart minio
```

### Remove MinIO Container
```bash
docker stop minio
docker rm minio
```

## Data Storage
- **Data Directory:** `~/minio-data`
- **Mounted Volume:** `/data` inside container

## Configuration in app-code-ref.py
The MinIO configuration in your `app-code-ref.py` matches these settings:
- Endpoint: `localhost:9000`
- Access Key: `minioadmin`
- Secret Key: `minioadmin`
- Buckets: `videos` and `transcripts` (will be created automatically when first used)

## Next Steps
1. Access MinIO Console at http://localhost:9001 to create buckets manually (optional)
2. Or let the app create buckets automatically when you upload videos
3. Start your Streamlit app: `streamlit run app-code-ref.py --server.address 0.0.0.0`

