import socket
import json
from confluent_kafka import Producer
from googleapiclient.discovery import build

# Kafka settings
BROKER = 'localhost:9092'
TOPIC = 'youtube_topic'
YOUTUBE_API_KEY = 'AIzaSyC-_iYv1qcNxI5yxUFOXNR5NIwqpp-qyhQ'


def create_kafka_producer(broker):
    conf = {
        'bootstrap.servers': broker,
        'client.id': socket.gethostname()
    }
    producer = Producer(conf)
    return producer


def get_video_details(youtube, video_id):
    video_request = youtube.videos().list(part="snippet", id=video_id)
    video_response = video_request.execute()
    title = video_response["items"][0]["snippet"]["title"]

    total_likes = 0
    page_token = None
    page_count = 0
    while page_count < 5:
        comment_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=page_token
        )
        comment_response = comment_request.execute()
        for item in comment_response.get("items", []):
            total_likes += item["snippet"]["topLevelComment"]["snippet"]["likeCount"]

        page_token = comment_response.get("nextPageToken")
        if not page_token:
            break
        page_count += 1

    return title, total_likes


def publish_youtube_analysis():
    producer = create_kafka_producer(BROKER)
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

    num_videos = int(input("How many videos do you want to search (around 1-5)? "))
    num_videos = min(max(num_videos, 1), 5)

    video_ids = []
    for i in range(num_videos):
        video_id = input(f"Input the number {i + 1} video ID: ")
        video_ids.append(video_id.strip())

    if not video_ids:
        print("Didn't detect any videos")
        return

    for video_id in video_ids:
        title, total_likes = get_video_details(youtube, video_id)
        if title is not None:
            message = {'video_id': video_id, 'title': title, 'total_comment_likes': total_likes}
            message_json = json.dumps(message)
            producer.produce(TOPIC, key=video_id, value=message_json)
            print(f"Detected '{title}' \n and the likes number is: {total_likes}")

    producer.flush()
    print("All data sent to Kafka。")


if __name__ == "__main__":
    publish_youtube_analysis()