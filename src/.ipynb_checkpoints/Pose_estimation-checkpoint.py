import torch
import intel_extension_for_pytorch as ipex
from ultralytics import YOLO
import minio
import psycopg
from psycopg.rows import TupleRow
from dotenv import load_dotenv
import os
import pickle
import uuid
# Load the environment variables from the .env file
load_dotenv() 

def get_postgres_client():
    return psycopg.connect(
        host = os.getenv("POSTGRES_HOST"),
        port = os.getenv("POSTGRES_PORT"),
        dbname = os.getenv("POSTGRES_DB"),
        user = os.getenv("POSTGRES_USER"),
        password = os.getenv("POSTGRES_PASSWORD")
    )

def get_minio_client():
    return minio.Minio(
        endpoint = os.getenv("MINIO_URL") + ":" + os.getenv("MINIO_PORT"),
        access_key = os.getenv("MINIO_API_ACCESSKEY"),
        secret_key = os.getenv("MINIO_API_SECRETKEY"),
        secure = False)

def get_minio_url(name: str):
    url = get_minio_client().presigned_get_object("highjump", name)
    return url

def process_video(model:YOLO, url: str):
    postgres_client = get_postgres_client()
    cursor = postgres_client.cursor()
    generator = model.predict(url, stream = True, stream_buffer = True)
    for idx, result in enumerate(generator):
        print("""
            INSERT INTO video_yolo_pose(id, video_name, frame_idx, position)
            VALUES(%s, %s, %s, %s)
            """,
            (uuid.uuid4(), url, idx, pickle.dumps(result.keypoints.data)))
        cursor.execute("""
            INSERT INTO video_yolo_pose(id, video_name, frame_idx, position)
            VALUES(%s, %s, %s, %s)
            """,
            (uuid.uuid4(), url, idx, pickle.dumps(result.keypoints.data)))
        

model_path = './yolo11x-pose_openvino_model/'
model = YOLO(model_path)
minio_client = get_minio_client()

video_list_generator = minio_client.list_objects('highjump', recursive=True, prefix='raw_data')
video_list = [i for i in video_list_generator]
process_video(model, get_minio_url(video_list[0].object_name))